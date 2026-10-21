"""One pretraining run on CPU with the NumPy reference model. Returns a JSON-able record."""
from __future__ import annotations
import math, time
import numpy as np
from .model_np import MotionTransformer, AdamW
from . import features as F

SIZES = {  # name: (d_model, layers, heads)
    "2XS": (16, 2, 2), "XS": (32, 2, 2), "S": (64, 2, 4), "M": (96, 3, 4), "L": (128, 4, 4), "XL": (192, 5, 6),
    # GPU ladder (gpu/train_torch.py)
    "2XL": (256, 6, 8), "3XL": (384, 8, 8), "4XL": (512, 10, 8), "5XL": (768, 12, 12),
}


def evaluate(model, val_w, T, K, rng_seed=0):
    out = {}
    for mode in ("full", "exo"):
        m = F.make_masks(len(val_w), mode, np.random.default_rng(rng_seed))
        losses, se_arm, se_arm_base, n = [], 0.0, 0.0, 0
        for i in range(0, len(val_w), 128):
            w = val_w[i:i + 128]
            inp, tgt = F.batch(w, T, K, m[i:i + 128])
            y, _ = model.forward(inp)
            losses.append(((y - tgt) ** 2).mean(axis=(1, 2)))
            if mode == "exo":   # horizon-1 error on the joints the suit never saw, in degrees
                hid = [j for j in F.HIDDEN if j in F.ANGLE_DIMS]
                std = model._std[hid]
                err = (y[..., hid] - tgt[..., hid]) * std
                se_arm += float((np.degrees(err) ** 2).sum()); n += err.size
                se_arm_base += float((np.degrees(tgt[..., hid] * std) ** 2).sum())   # predict the average pose
        pw = np.concatenate(losses)
        out[mode] = float(pw.mean())
        if mode == "full":
            out["per_window"] = np.round(pw, 5).tolist()
        if mode == "exo":
            out["hidden_joint_rmse_deg"] = math.sqrt(se_arm / n)
            out["hidden_joint_rmse_deg_meanpose"] = math.sqrt(se_arm_base / n)
    return out


def run(data, size="S", fraction=1.0, steps=1200, B=64, T=32, K=4, lr=None, seed=0, eval_every=150,
        log=print, val_windows=512):
    d, L, H = SIZES[size]
    tr_ids = data.subset("train", fraction, seed=seed)
    trw = F.Windows(data, tr_ids, T, K, seed=seed)
    vaw = F.Windows(data, data.subset("val"), T, K).fixed(val_windows, seed=7)
    tew = F.Windows(data, data.subset("test"), T, K).fixed(val_windows, seed=11)
    model = MotionTransformer(2 * F.D, K * F.D, d, L, H, T, seed=seed)
    model._std = data.std
    lr = lr or min(3e-3, 0.02 / math.sqrt(d))
    opt = AdamW(model.p, lr=lr)
    rng = np.random.default_rng(seed + 1)
    warm = max(20, steps // 20)
    curve, best, t0 = [], None, time.time()
    for step in range(1, steps + 1):
        w = trw.sample(B)
        inp, tgt = F.batch(w, T, K, F.make_masks(B, "mixed", rng))
        y, cache = model.forward(inp)
        loss = float(((y - tgt) ** 2).mean())
        g = model.backward(2.0 * (y - tgt) / y.size, cache)
        frac = step / steps
        cur = lr * (step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm))))
        gn = opt.step(model.p, g, cur)
        if step % eval_every == 0 or step == steps:
            ev = evaluate(model, vaw, T, K)
            curve.append(dict(step=step, train=loss, val=ev["full"], val_exo=ev["exo"],
                              hidden_rmse=ev["hidden_joint_rmse_deg"], sec=time.time() - t0))
            if best is None or ev["full"] < best["val"]["full"]:
                best = dict(step=step, val=ev, test=evaluate(model, tew, T, K),
                            weights={k: v.copy() for k, v in model.p.items()})
            ev = {k: v for k, v in ev.items()}
            log(f"  {size} f={fraction:<6} step {step:5d} train {loss:.4f} val {ev['full']:.4f} exo {ev['exo']:.4f} "
                f"hidden {ev['hidden_joint_rmse_deg']:.1f}deg  {time.time()-t0:.0f}s")
    tokens = steps * B * T
    rec = dict(size=size, d_model=d, layers=L, heads=H, params=int(model.n_params),
               params_nonembed=int(L * 12 * d * d), fraction=fraction, train_frames=trw.frames,
               train_hours=trw.frames / 20 / 3600, steps=steps, batch=B, context=T, horizon=K, lr=lr,
               tokens=tokens, flops=6.0 * model.n_params * tokens, epochs=tokens / max(1, trw.frames),
               best_step=best["step"], val=best["val"], test=best["test"], curve=curve,
               wall_s=time.time() - t0)
    return rec, best["weights"]
