"""Pre-register what 100 hours should buy: fit L(N, D) = E + A N^-a + B D^-b on every CPU run, bootstrap, predict."""
import glob, json, os, time
import numpy as np
from scipy.optimize import curve_fit
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
rs = [json.load(open(p)) for p in glob.glob(os.path.join(RUNS, "run_*.json"))]
N = np.array([r["params"] for r in rs], float); D = np.array([r["train_hours"] for r in rs], float)
f = lambda X, E, A, a, B, b: E + A * X[0] ** -a + B * X[1] ** -b
def fit(y):
    best = None
    for a0 in (0.3, 0.8, 1.4):
        for b0 in (0.2, 0.6, 1.0):
            try:
                p, _ = curve_fit(f, (N, D), y, p0=(0.5 * y.min(), 50.0, a0, 0.05, b0), bounds=([0, 0, 0.01, 0, 0.01], [y.min(), np.inf, 3, np.inf, 3]), maxfev=40000)
                r = float(((np.log(f((N, D), *p)) - np.log(y)) ** 2).sum())
                if best is None or r < best[1]: best = (p, r)
            except Exception: pass
    return best[0]
out = {"registered_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "form": "L(N, D) = E + A N^-alpha + B D^-beta", "fit_on": len(rs), "predictions": {}}
rng = np.random.default_rng(0)
sizes = {"S": 114236, "M": 356860, "L": 821500, "2XL": 4921084}
for split in ("val", "test"):
    W = [np.array(r[split]["per_window"]) for r in rs]
    y0 = np.array([w.mean() for w in W]); p0 = fit(y0)
    boots = []
    for _ in range(300):
        y = np.array([w[rng.integers(0, len(w), len(w))].mean() for w in W])
        try: boots.append(fit(y))
        except Exception: pass
    res = {"params": dict(zip(["E", "A", "alpha", "B", "beta"], map(float, p0)))}
    for name, n in sizes.items():
        for h in (12.0, 100.0, 617.0):
            v = [f((np.array([n]), np.array([h])), *b)[0] for b in boots]
            res[f"{name}@{h:g}h"] = dict(loss=float(f((np.array([n]), np.array([h])), *p0)[0]), lo=float(np.percentile(v, 10)), hi=float(np.percentile(v, 90)))
    out["predictions"][split] = res
json.dump(out, open(os.path.join(RUNS, "prediction_hours.json"), "w"), indent=1)
for split, res in out["predictions"].items():
    print(split, {k: round(v, 3) for k, v in res["params"].items()})
    for k in ("S@12h", "S@100h", "L@12h", "L@100h", "2XL@100h", "2XL@617h"):
        print(f"   {k:10s} {res[k]['loss']:.4f} [{res[k]['lo']:.4f}, {res[k]['hi']:.4f}]")
