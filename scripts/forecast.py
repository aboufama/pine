"""Joint scaling fit L(N, D) on the phase-1 GPU runs and the body-only projection along a compute-optimal path.

    python scripts/forecast.py          # reads demo/data.json (D.gpu), writes D.gpu.forecast back

Points: the model-size ladder (all data) and the data ladder (2XL on 1/64, 1/16, 1/4 of the data).
Two forms, both with an irreducible term E:
  additive  L = E + A N^-a + B D^-b            (Hoffmann et al. 2022)
  coupled   L = E + ((Nc/N)^(a/b) + Dc/D)^b      (Kaplan et al. 2020, plus E)
Fitted in log space; the spread comes from leave-one-out refits of both forms.
Projection: N = 20 tokens per parameter at one token per frame, 20 frames a second (3,600 parameters per hour), never below
the largest model trained (so the path starts at today's best run).
Text reference: the Chinchilla fit (Hoffmann et al. 2022, App. D.2) and Epoch's replication, evaluated on the same
parameter path with D = 20 N tokens, as a ratio to its value at our largest model.
Generalization point: where the projected error on trained-on people falls to today's error divided by the stranger gap
(error on held-out people / error on held-out takes of trained people, every HiPHI phase-2 run).
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo", "data.json")
PER_H = 72000 / 20


def additive(p, N, D):
    lE, lA, a, lB, b = p
    return np.exp(lE) + np.exp(lA) * N ** -a + np.exp(lB) * D ** -b


def coupled(p, N, D):
    lE, lNc, a, lDc, b = p
    return np.exp(lE) + ((np.exp(lNc) / N) ** (a / b) + np.exp(lDc) / D) ** b


INITS = {
    additive: [[np.log(e), np.log(A), a, np.log(B), b] for e in (1e-4, 0.01, 0.03) for A in (0.1, 1.0) for a in (0.05, 0.2)
               for B in (0.1, 1.0) for b in (0.3, 0.6)],
    coupled: [[np.log(e), np.log(nc), a, np.log(dc), b] for e in (1e-4, 0.01, 0.03) for nc in (1e-3, 1.0, 1e3) for a in (0.05, 0.1)
              for dc in (1.0, 50.0) for b in (0.2, 0.5)],
}
LB = {additive: [np.log(1e-6), -20, 0.0, -20, 0.0], coupled: [np.log(1e-6), -40, 0.005, -20, 0.01]}
UB = {additive: [np.log(0.2), 20, 2.0, 20, 2.0], coupled: [np.log(0.2), 40, 2.0, 20, 2.0]}


def fit(f, N, D, L):
    best = None
    for p0 in INITS[f]:
        try:
            r = least_squares(lambda p: np.log(f(p, N, D)) - np.log(L), p0, bounds=(LB[f], UB[f]), max_nfev=20000)
        except ValueError:
            continue
        if np.isfinite(r.cost) and (best is None or r.cost < best.cost):
            best = r
    return best.x, float(np.sqrt(2 * best.cost / len(L)))


def chin(N, E=1.69, A=406.4, a=0.34, B=410.7, b=0.28):
    return E + A / N ** a + B / (20 * N) ** b


def main():
    D = json.load(open(DEMO))
    G = D["gpu"]
    pts = [(r["params"], r["train_hours"], r["val"], r["size"] + " all") for r in G["ladder"]]
    pts += [(r["params"], r["train_hours"], r["val"], r["size"] + f" {r['fraction']}") for r in G["data_ladder"] if r["fraction"] < 1]
    N = np.array([p[0] for p in pts], float); H = np.array([p[1] for p in pts], float); L = np.array([p[2] for p in pts], float)
    best_now = float(L.min())
    n_now = float(N[np.argmin(L)]); h_now = float(H[np.argmin(L)])
    npath = lambda h: np.maximum(PER_H * np.asarray(h, float), n_now)
    path_h = [1e3, 1e4, 1e5, 1e6, 1e7]
    grid = np.exp(np.linspace(np.log(h_now), np.log(3e7), 60))
    hr = [r for r in G["phase2"]["runs"] if (r.get("split") or {}).get("experiment") == "HiPHI"]
    gaps = np.array([r["test"] / r["val"] for r in hr])
    gap = float(gaps.mean())
    curves = []
    out = dict(points=[dict(params=p[0], hours=p[1], loss=p[2], tag=p[3]) for p in pts], now=best_now, per_hour_params=PER_H,
               path_hours=path_h, forms={})
    allproj = {h: [] for h in path_h}
    floors = []
    for f in (additive, coupled):
        name = f.__name__
        p, rmse = fit(f, N, H, L)
        proj = {h: float(f(p, npath(h), h)) for h in path_h}
        curves.append(f(p, npath(grid), grid))
        loo = []
        for i in range(len(L)):
            m = np.arange(len(L)) != i
            q, _ = fit(f, N[m], H[m], L[m])
            loo.append(dict(drop=pts[i][3], E=float(np.exp(q[0])), a=float(q[2]), b=float(q[4]), proj={str(h): float(f(q, npath(h), h)) for h in path_h}))
            curves.append(f(q, npath(grid), grid))
            for h in path_h:
                allproj[h].append(float(f(q, npath(h), h)))
            floors.append(float(np.exp(q[0])))
        for h in path_h:
            allproj[h].append(proj[h])
        floors.append(float(np.exp(p[0])))
        out["forms"][name] = dict(params=[float(x) for x in p], E=float(np.exp(p[0])), exp_N=float(p[2]), exp_D=float(p[4]), rmse_log=rmse,
                                  proj={str(h): proj[h] for h in path_h}, loo=loo)
        print(f"{name:8s} E={np.exp(p[0]):.4f} a={p[2]:.3f} b={p[4]:.3f} rmse(log)={rmse:.3f}  " +
              "  ".join(f"{h:.0e}:{best_now / proj[h]:.2f}x" for h in path_h))
    out["range"] = {str(h): dict(lo=min(v), hi=max(v), x_lo=best_now / max(v), x_hi=best_now / min(v)) for h, v in allproj.items()}
    out["floor"] = dict(lo=min(floors), hi=max(floors))
    C = np.array(curves)
    out["curve"] = dict(hours=grid.tolist(), lo=C.min(0).tolist(), hi=C.max(0).tolist(), mid=np.median(C, 0).tolist())
    for nm in ("additive", "coupled"):
        lo_a = [l["a"] for l in out["forms"][nm]["loo"]]; lo_b = [l["b"] for l in out["forms"][nm]["loo"]]
        out["forms"][nm]["exp_N_range"] = [min(lo_a), max(lo_a)]; out["forms"][nm]["exp_D_range"] = [min(lo_b), max(lo_b)]
        print(nm, "exp_N LOO", [round(x, 3) for x in (min(lo_a), max(lo_a))], "exp_D LOO", [round(x, 3) for x in (min(lo_b), max(lo_b))])
    # text on the same parameter path
    epoch = dict(E=1.8172, A=482.01, a=0.3478, B=2085.43, b=0.3658)
    out["text"] = dict(n_now=n_now, chinchilla=(chin(npath(grid)) / chin(n_now)).tolist(), epoch=(chin(npath(grid), **epoch) / chin(n_now, **epoch)).tolist(),
                       x_at_1e7=dict(chinchilla=float(chin(n_now) / chin(npath(1e7))), epoch=float(chin(n_now, **epoch) / chin(npath(1e7), **epoch))),
                       x_at_1e6=dict(chinchilla=float(chin(n_now) / chin(npath(1e6))), epoch=float(chin(n_now, **epoch) / chin(npath(1e6), **epoch))))
    print("text on the same path:", out["text"]["x_at_1e6"], out["text"]["x_at_1e7"])
    # generalization point: first hour where the curve reaches now / gap, for the band edges and the gap range
    def cross(c, g):
        t = best_now / g
        i = np.argmax(np.asarray(c) <= t)
        if c[i] > t:
            return None
        x0, x1, y0, y1 = np.log(grid[i - 1]), np.log(grid[i]), np.log(c[i - 1]), np.log(c[i])
        return float(np.exp(x0 + (np.log(t) - y0) * (x1 - x0) / (y1 - y0)))
    out["gap"] = dict(mean=gap, lo=float(gaps.min()), hi=float(gaps.max()), n=len(gaps))
    out["gen_point"] = dict(early=cross(C.min(0), gap), mid=cross(np.median(C, 0), gap), late=cross(C.max(0), gap))
    print("gap", out["gap"], "generalization point", out["gen_point"])
    for h in path_h:
        r = out["range"][str(h)]
        print(f"{h:>10.0f} h  N={PER_H * h / 1e9:7.3f}B  loss {r['lo']:.4f}-{r['hi']:.4f}  => {r['x_lo']:.2f}-{r['x_hi']:.2f}x lower than {best_now:.4f}")
    print("floor E range", out["floor"])
    G["forecast"] = out
    json.dump(D, open(DEMO, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
