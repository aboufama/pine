"""Exposure-bias fixes for the chunked motion model, scored exactly like scripts/pi_pass.py.

Both ideas sit on top of pi_pass's best variant (chunk16: 16-frame residual chunks, execute 8, re-plan) with
everything else identical: size S, 900 steps, seed 0, the same training batches and suit masks (corruption
draws from its own random stream), the same checkpoint rule and the same 200 held-out windows.
pi_pass.score is imported and used unchanged. `--check` proves that with no corruption this loop
reproduces pi_pass.train bit for bit.

  noisyhist  Synthetic drift on the history. On half of the training windows, a random suffix of the
             context (4-32 frames) gets smooth drift (an integrated random walk, so it ramps up the way a
             forecast error does) plus a little jitter. Drift size at the last context frame is
             U(0, 2) x 0.15 sqrt(n) std units, capped at 1.0; this model family's own forecast error is
             ~0.4 of the joints' spread at 200 ms and ~0.75 at 1-2 s. Targets stay the clean truth, written
             as offsets from the corrupted frame, so the model learns to pull a drifting history back.
             No extra compute. (Input-noise "training noise" from Learning to Simulate, ICML 2020; noise-
             augmented context in GameNGen, ICLR 2025, and History-Guided Video Diffusion, ICML 2025.)
  selfhist   Two-forward scheduled sampling with the model's own predictions. For a share of the windows
             (0 for the first 150 steps, ramped to 25% by step 450) a no-gradient pass predicts a chunk from
             the real frames up to T-1-n, the last n context frames (n ~ U{1..16}, one n per step) become
             that open-loop prediction, and the gradient pass trains on this history against the clean truth.
             ~8% extra compute. (MotionStreamer, ICCV 2025; DART, ICLR 2025.)
  none       Identical to pi_pass chunk16.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np

ROOT = "/home/claude/pine-engine"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import pi_pass                                   # noqa: E402  objective, inputs, rollout, score: reused as is
from pine.model_np import MotionTransformer, AdamW   # noqa: E402
from pine.features import MotionData             # noqa: E402
from pine import features as F, train_np         # noqa: E402

OUT = os.path.join(ROOT, "runs", "research"); os.makedirs(OUT, exist_ok=True)
BASE = next(o for o in pi_pass.VARIANTS if o.name == "chunk16")


def targets_from(obj, w, base, T):
    """Clean future frames, as offsets from `base` (the frames the model actually saw)."""
    fut = np.stack([w[:, 1 + j:T + 1 + j] for j in range(obj.H)], 2)
    if obj.residual:
        fut = fut - base[:, :T, None, :]
    if obj.B is not None:
        fut = np.einsum("hk,bthd->btkd", obj.B, fut)
    return fut.reshape(w.shape[0], T, -1).astype(np.float32)


def noisy_history(w, T, rng, p=0.5, umax=2.0, rate=0.15, cap=1.0, nmin=4, jitter=0.1, mask=None):
    """Only channels the suit mask observes are corrupted: hidden channels are zeroed in the input anyway,
    and corrupting them would only add unpredictable noise to their residual targets."""
    B, D = w.shape[0], w.shape[2]
    ctx = w[:, :T].copy()
    sel = rng.random(B) < p
    n = rng.integers(nmin, T + 1, size=B)                                  # corrupted suffix length
    end = np.minimum(rng.uniform(0, umax, size=B) * rate * np.sqrt(n), cap) * sel   # drift std at last frame
    on = (np.arange(T)[None] >= (T - n)[:, None]).astype(np.float32)[..., None]
    acc = rng.standard_normal((B, T, D)).astype(np.float32) * on
    drift = np.cumsum(np.cumsum(acc, 1), 1)                                # smooth, grows ~ t^1.5 from 0
    nn = n.astype(np.float64)
    drift /= np.sqrt(nn * (nn + 1) * (2 * nn + 1) / 6).astype(np.float32)[:, None, None]   # unit std at the end
    jit = rng.standard_normal((B, T, D)).astype(np.float32) * on * jitter
    corr = (drift + jit) * end.astype(np.float32)[:, None, None]
    if mask is not None:
        corr *= mask[:, None, :]
    ctx += corr
    return ctx


def self_history(m, obj, w, T, mask, rng, frac):
    """The last n context frames of a random subset become the model's own open-loop chunk, predicted
    from the real frames before them (n ~ U{1..H}, one n per step, so the no-gradient pass only runs over
    the T-n real frames it needs). Observed channels only (see noisy_history)."""
    B = w.shape[0]; ctx = w[:, :T].copy()
    k = int(round(frac * B))
    if k == 0:
        return ctx
    sel = np.sort(rng.choice(B, k, replace=False))
    n = int(rng.integers(1, obj.H + 1))
    a = T - 1 - n                                                    # last real frame
    y, _ = m.forward(pi_pass.inputs(w[sel], a + 1, mask[sel]))       # no gradient taken from this pass
    fut = obj.decode(y[:, -1], w[sel, a])[:, :n]                     # (k, n, D)
    ctx[sel, a + 1:] = np.where(mask[sel][:, None, :] > 0, fut, ctx[sel, a + 1:])
    return ctx


def train(data, obj, variant, size="S", steps=900, B=64, T=32, seed=0, log=print,
          p_noise=0.5, f_self=0.25, start=150, ramp=300):
    d, L, H = train_np.SIZES[size]
    tr = F.Windows(data, data.subset("train"), T, obj.H, seed=seed)
    va = F.Windows(data, data.subset("val"), T, obj.H).fixed(512, seed=7)
    m = MotionTransformer(2 * F.D, obj.out_dim, d, L, H, T, seed=seed)
    lr = min(3e-3, 0.02 / math.sqrt(d)); opt = AdamW(m.p, lr=lr)
    rng = np.random.default_rng(seed + 1); warm = max(20, steps // 20)
    aug = np.random.default_rng(seed + 1000)     # own stream, so batches and masks match pi_pass exactly
    best, t0, c0 = None, time.time(), time.process_time()
    vmask = np.ones((len(va), F.D), np.float32)
    for step in range(1, steps + 1):
        w = tr.sample(B)
        mask = F.make_masks(B, "mixed", rng)
        if variant == "noisyhist":
            ctx = noisy_history(w, T, aug, p_noise, mask=mask)
        elif variant == "selfhist":     # DART-style: plain training first, then ramp the self-history share
            ctx = self_history(m, obj, w, T, mask, aug, f_self * min(1.0, max(0.0, (step - start) / ramp)))
        else:
            ctx = w[:, :T]
        inp = pi_pass.inputs(ctx, T, mask); tgt = targets_from(obj, w, ctx, T)
        y, c = m.forward(inp)
        g = m.backward(2.0 * (y - tgt) / y.size, c)
        cur = lr * (step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm))))
        opt.step(m.p, g, cur)
        if step % 150 == 0 or step == steps:
            vl = 0.0
            for i in range(0, len(va), 128):
                b = va[i:i + 128]; yy, _ = m.forward(pi_pass.inputs(b, T, vmask[:len(b)]))
                vl += float(((yy - obj.targets(b, T)) ** 2).mean()) * len(b)
            vl /= len(va)
            if best is None or vl < best[0]:
                best = (vl, step, {k: v.copy() for k, v in m.p.items()})
            log(f"  {variant} step {step} val {vl:.4f} {time.time()-t0:.0f}s")
    m.p = best[2]
    return m, dict(best_val=best[0], best_step=best[1], wall_s=time.time() - t0,
                   cpu_s=time.process_time() - c0, params=int(m.n_params))


def boot_ci(per_window_sq, n=2000, seed=0):
    r = np.random.default_rng(seed); k = len(per_window_sq)
    s = np.sqrt(per_window_sq[r.integers(0, k, (n, k))].mean(1))
    return [float(np.degrees(np.percentile(s, 2.5))), float(np.degrees(np.percentile(s, 97.5)))]


def score_with_detail(data, m, obj, W, ctx=32, hor=40):
    """pi_pass.score unchanged, plus per-window errors (caught from its own rollouts) for intervals."""
    caught, orig = {}, pi_pass.rollout
    def keep(m_, o_, W_, c_=32, h_=40, ex=None):
        out = orig(m_, o_, W_, c_, h_, ex); caught[ex or o_.execute] = out; return out
    pi_pass.rollout = keep
    try:
        res = pi_pass.score(data, m, obj, W, ctx, hor)
    finally:
        pi_pass.rollout = orig
    std = data.std[:23]; truth = W[:, ctx:ctx + hor, :23] * std
    hold = np.repeat(W[:, ctx - 1:ctx, :23] * std, hor, 1)
    hsq = ((hold - truth) ** 2).mean((1, 2))
    detail = {}
    for ex, seq in caught.items():
        p = seq[..., :23] * std
        sq = ((p - truth) ** 2).mean((1, 2))
        r = np.random.default_rng(1); idx = r.integers(0, len(sq), (2000, len(sq)))
        diff = np.degrees(np.sqrt(sq[idx].mean(1))) - np.degrees(np.sqrt(hsq[idx].mean(1)))
        detail[f"execute_{ex}"] = dict(mean_2s_ci95=boot_ci(sq),
                                       minus_hold_ci95=[float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
                                       per_window_mse_rad2=np.round(sq, 6).tolist())
    detail["hold_pose"] = dict(mean_2s_ci95=boot_ci(hsq), per_window_mse_rad2=np.round(hsq, 6).tolist())
    return res, detail


def check(data):
    """--check: with no corruption this loop must reproduce pi_pass.train bit for bit (3 steps)."""
    m1, _ = pi_pass.train(data, BASE, steps=3, log=lambda s: None)
    m2, _ = train(data, BASE, "none", steps=3, log=lambda s: None)
    diff = max(float(np.abs(m1.p[k] - m2.p[k]).max()) for k in m1.p)
    w = F.Windows(data, data.subset("train"), 32, 16, seed=0).sample(8)
    tdiff = float(np.abs(BASE.targets(w, 32) - targets_from(BASE, w, w[:, :32], 32)).max())
    print(f"max weight difference vs pi_pass.train after 3 steps: {diff:.3g}; target difference {tdiff:.3g}")
    return diff == 0.0 and tdiff == 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["noisyhist", "selfhist", "none"], default="noisyhist")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=None,
                    help="default 0. The control ('none') defaults to 1: at seed 0 it is bit-identical to pi_pass "
                         "chunk16 (see --check), so a second seed is the run that measures run-to-run noise.")
    a = ap.parse_args()
    seed = a.seed if a.seed is not None else (1 if a.variant == "none" else 0)
    tag = a.variant if seed == 0 else f"{a.variant}_s{seed}"
    t0 = time.time(); c0 = time.process_time()
    data = MotionData("/home/claude/data/canon", f"/home/claude/data/{F.ROBOT}")
    if a.check:
        sys.exit(0 if check(data) else 1)
    W = F.Windows(data, data.subset("test"), 32 + 40 + 16 - 4, 4).fixed(200, seed=21)   # as pi_pass.main
    print(f"== chunk16 + {a.variant}, seed {seed}", flush=True)
    m, rec = train(data, BASE, a.variant, seed=seed, log=lambda s: print(s, flush=True))
    ts = time.time()
    res, detail = score_with_detail(data, m, BASE, W)
    rec.update(res, name=f"chunk16+{a.variant}", seed=seed, base="chunk16", horizon_frames=BASE.H,
               residual=BASE.residual, execute=BASE.execute, score_s=time.time() - ts, total_s=time.time() - t0,
               total_cpu_s=time.process_time() - c0, detail=detail)
    json.dump(rec, open(os.path.join(OUT, f"{tag}.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, f"weights_{tag}.npz"), **m.p)
    for ex in (1, BASE.execute):
        e, dd = rec[f"execute_{ex}"], detail[f"execute_{ex}"]
        print(f"   execute {ex}: 0.5s {e['0.5s']:.2f}  1s {e['1.0s']:.2f}  2s {e['2.0s']:.2f}  mean {e['mean_2s']:.2f} deg "
              f"(95% {dd['mean_2s_ci95'][0]:.2f}-{dd['mean_2s_ci95'][1]:.2f}; vs hold {dd['minus_hold_ci95'][0]:+.2f} to "
              f"{dd['minus_hold_ci95'][1]:+.2f})", flush=True)
    print(f"   hold {rec['hold_pose']['mean_2s']:.2f} | 200ms from real {rec['from_real_200ms']:.2f} | "
          f"train {rec['wall_s']:.0f}s score {rec['score_s']:.0f}s total {rec['total_s']:.0f}s "
          f"(cpu {rec['total_cpu_s']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
