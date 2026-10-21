"""Ideas from Physical Intelligence's papers, tested on the motion model under one fixed budget.

  direct4  baseline: predict the next 4 frames (200 ms) directly                (what the sweep used)
  resid4   the same, as offsets from the current frame                          (zero output = hold the pose)
  chunk16  action chunking (pi0): predict 16 frames (800 ms), execute 8, re-plan
  dct16    FAST-style (pi0-FAST): the 16-frame chunk as its 6 lowest DCT coefficients per channel

Every variant is size S (114k params), 900 steps, same data, seed and masking. All are scored the same way:
free-running forecasts on 200 windows from people never seen in training, joint-angle error in degrees,
against simply holding the last pose.
"""
from __future__ import annotations
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine.model_np import MotionTransformer, AdamW    # noqa: E402
from pine.features import MotionData                 # noqa: E402
from pine import features as F, train_np             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "pi_pass"); os.makedirs(OUT, exist_ok=True)


def dct_basis(H, k):
    n = np.arange(H)[:, None]; j = np.arange(k)[None]
    B = np.cos(np.pi * (n + 0.5) * j / H) * np.sqrt(2.0 / H); B[:, 0] /= np.sqrt(2.0)
    return B.astype(np.float32)                                   # (H, k), orthonormal columns


class Objective:
    def __init__(self, name, H, residual, k=None, execute=1):
        self.name, self.H, self.residual, self.k, self.execute = name, H, residual, k, execute
        self.B = dct_basis(H, k) if k else None
        self.rows = k or H
        self.out_dim = self.rows * F.D

    def targets(self, w, T):
        fut = np.stack([w[:, 1 + j:T + 1 + j] for j in range(self.H)], 2)          # (B, T, H, D)
        if self.residual:
            fut = fut - w[:, :T, None, :]
        if self.B is not None:
            fut = np.einsum("hk,bthd->btkd", self.B, fut)
        return fut.reshape(w.shape[0], T, -1).astype(np.float32)

    def decode(self, y, last):                                                       # -> (B, H, D)
        c = y.reshape(len(y), self.rows, F.D)
        fut = np.einsum("hk,bkd->bhd", self.B, c) if self.B is not None else c
        return fut + last[:, None] if self.residual else fut


VARIANTS = [Objective("direct4", 4, False, execute=1), Objective("resid4", 4, True, execute=1),
            Objective("chunk16", 16, True, execute=8), Objective("dct16", 16, True, k=6, execute=8)]


def inputs(w, T, mask):
    x = w[:, :T] * mask[:, None, :]
    return np.concatenate([x, np.broadcast_to(mask[:, None, :], x.shape)], -1).astype(np.float32)


def rollout(m, obj, W, ctx=32, hor=40, execute=None):
    execute = execute or obj.execute
    seq = W[:, :ctx].copy(); ones = np.ones((len(W), F.D), np.float32)
    while seq.shape[1] < ctx + hor:
        win = seq[:, -32:]
        y, _ = m.forward(inputs(win[:, :, :], win.shape[1], ones))
        seq = np.concatenate([seq, obj.decode(y[:, -1], win[:, -1])[:, :execute]], 1)
    return seq[:, ctx:ctx + hor]


def score(data, m, obj, W, ctx=32, hor=40):
    std = data.std[:23]; truth = W[:, ctx:ctx + hor, :23] * std
    hold = np.repeat(W[:, ctx - 1:ctx, :23] * std, hor, 1)
    rmse = lambda a, k: float(np.degrees(np.sqrt(((a[:, k] - truth[:, k]) ** 2).mean())))
    res = {}
    for ex in sorted({1, obj.execute}):
        p = rollout(m, obj, W, ctx, hor, ex)[..., :23] * std
        res[f"execute_{ex}"] = {f"{s}s": rmse(p, int(s * 20) - 1) for s in (0.25, 0.5, 1.0, 2.0)}
        res[f"execute_{ex}"]["mean_2s"] = float(np.degrees(np.sqrt(((p - truth) ** 2).mean())))
    res["hold_pose"] = {f"{s}s": rmse(hold, int(s * 20) - 1) for s in (0.25, 0.5, 1.0, 2.0)}
    res["hold_pose"]["mean_2s"] = float(np.degrees(np.sqrt(((hold - truth) ** 2).mean())))
    y, _ = m.forward(inputs(W[:, :ctx], ctx, np.ones((len(W), F.D), np.float32)))   # from real frames
    fut = obj.decode(y[:, -1], W[:, ctx - 1]) * data.std
    res["from_real_200ms"] = float(np.degrees(np.sqrt(((fut[:, 3, :23] - W[:, ctx + 3, :23] * data.std[:23]) ** 2).mean())))
    return res


def train(data, obj, size="S", steps=900, B=64, T=32, seed=0, log=print):
    d, L, H = train_np.SIZES[size]
    tr = F.Windows(data, data.subset("train"), T, obj.H, seed=seed)
    va = F.Windows(data, data.subset("val"), T, obj.H).fixed(512, seed=7)
    m = MotionTransformer(2 * F.D, obj.out_dim, d, L, H, T, seed=seed)
    lr = min(3e-3, 0.02 / math.sqrt(d)); opt = AdamW(m.p, lr=lr)
    rng = np.random.default_rng(seed + 1); warm = max(20, steps // 20)
    best, t0 = None, time.time()
    vmask = np.ones((len(va), F.D), np.float32)
    for step in range(1, steps + 1):
        w = tr.sample(B)
        inp = inputs(w, T, F.make_masks(B, "mixed", rng)); tgt = obj.targets(w, T)
        y, c = m.forward(inp)
        g = m.backward(2.0 * (y - tgt) / y.size, c)
        cur = lr * (step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm))))
        opt.step(m.p, g, cur)
        if step % 150 == 0 or step == steps:
            vl = 0.0
            for i in range(0, len(va), 128):
                b = va[i:i + 128]; yy, _ = m.forward(inputs(b, T, vmask[:len(b)]))
                vl += float(((yy - obj.targets(b, T)) ** 2).mean()) * len(b)
            vl /= len(va)
            if best is None or vl < best[0]:
                best = (vl, step, {k: v.copy() for k, v in m.p.items()})
            log(f"  {obj.name} step {step} val {vl:.4f} {time.time()-t0:.0f}s")
    m.p = best[2]
    return m, dict(best_val=best[0], best_step=best[1], wall_s=time.time() - t0, params=int(m.n_params))


def main():
    data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
    W = F.Windows(data, data.subset("test"), 32 + 40 + 16 - 4, 4).fixed(200, seed=21)
    results = {}
    for obj in VARIANTS:
        p = os.path.join(OUT, f"{obj.name}.json")
        if os.path.exists(p):
            results[obj.name] = json.load(open(p)); continue
        print(f"== {obj.name}", flush=True)
        m, rec = train(data, obj, log=lambda s: print(s, flush=True))
        rec.update(score(data, m, obj, W), name=obj.name, horizon_frames=obj.H, residual=obj.residual,
                   dct_coeffs=obj.k, execute=obj.execute)
        json.dump(rec, open(p, "w"), indent=1); results[obj.name] = rec
        e = rec[f"execute_{obj.execute}"]
        print(f"   {obj.name}: 0.5s {e['0.5s']:.1f}  1s {e['1.0s']:.1f}  2s {e['2.0s']:.1f}  mean {e['mean_2s']:.1f} deg | "
              f"hold {rec['hold_pose']['mean_2s']:.1f} | 200ms from real {rec['from_real_200ms']:.1f}", flush=True)
    json.dump(results, open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
