"""Gate: the PyTorch model must equal the NumPy reference before any GPU time is spent on training.

    python gpu/parity.py            exit 0 = identical, exit 1 = mismatch (a table says where), exit 2 = cannot run

Checks, float64 on CPU unless noted (tolerance: max relative error < 1e-6, relative to the reference tensor's scale):
  1. parameter names, shapes, count and initialisation (pine.model_np init copied in, round trip exact)
  2. loss and every parameter gradient on the same batch, built by pine.features.batch from random windows and
     mixed masks; for several sizes, initial and perturbed weights, full and shorter context
  3. AdamW: one step with clip 1.0 and one step where clipping is active (decoupled weight decay on 2-D params
     except pos, bias correction, b1 0.9, b2 0.95, eps 1e-8); parameter updates compared
  4. evaluation metrics (gpu.model_torch.evaluate_torch vs pine.train_np.evaluate), the LR schedule, the batch
     builder, and the mixed-mask distribution
  5. on a CUDA device, if present: fp32 loss and gradients (TF32 off, tolerance 1e-3) and a bf16 autocast loss (5e-2)
"""
from __future__ import annotations
import math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402,F401  (sets PINE_* defaults, puts the repo root on sys.path)
import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402
except ImportError:
    print("parity: torch is not installed; cannot check the GPU model", flush=True)
    sys.exit(2)

from pine import features as F                                   # noqa: E402
from pine.model_np import MotionTransformer, AdamW, count_params  # noqa: E402
from pine.train_np import SIZES, evaluate as evaluate_np         # noqa: E402
import model_torch as MT                                         # noqa: E402

TOL = 1e-6
ROWS: list[tuple] = []          # (check, item, value, limit, ok)


def rel_err(a, b) -> float:
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return float("inf")
    scale = max(float(np.abs(b).max()) if b.size else 0.0, 1e-30)
    return float(np.abs(a - b).max()) / scale if a.size else 0.0


def record(check: str, item: str, value: float, limit: float = TOL) -> bool:
    ok = bool(np.isfinite(value) and value < limit)
    ROWS.append((check, item, value, limit, ok))
    return ok


def np_model(size: str, seed: int, perturb: bool) -> MotionTransformer:
    d, L, H = SIZES[size]
    m = MotionTransformer(2 * F.D, 4 * F.D, d, L, H, 32, seed=seed, dtype=np.float64)
    if perturb:             # make every gain, bias and table distinct so a swapped parameter cannot hide
        r = np.random.default_rng(seed + 100)
        for k, v in m.p.items():
            v += (0.05 if v.ndim == 1 else 0.01) * r.standard_normal(v.shape)
    return m


def torch_twin(ref: MotionTransformer, attn: str, device="cpu", dtype=torch.float64) -> MT.MotionTransformerTorch:
    t = MT.MotionTransformerTorch(ref.din, ref.dout, ref.d, ref.L, ref.H, ref.T, attn=attn).to(device=device, dtype=dtype)
    t.load_from_numpy(ref.p)
    return t


def make_batch(B: int, T: int, K: int, seed: int):
    r = np.random.default_rng(seed)
    w = r.standard_normal((B, T + K, F.D)).astype(np.float32)
    mask = F.make_masks(B, "mixed", r)
    inp, tgt = F.batch(w, T, K, mask)
    return w, mask, inp, tgt


def check_structure():
    for size in ("XS", "S", "M", "XL"):
        d, L, H = SIZES[size]
        ref = MotionTransformer(2 * F.D, 4 * F.D, d, L, H, 32, seed=3)
        t = MT.MotionTransformerTorch(2 * F.D, 4 * F.D, d, L, H, 32)
        names_np = {k: tuple(v.shape) for k, v in ref.p.items()}
        ok = names_np == t.expected_shapes()
        record("structure", f"{size} names+shapes", 0.0 if ok else float("inf"))
        record("structure", f"{size} n_params", 0.0 if t.n_params == ref.n_params == count_params(2 * F.D, 4 * F.D, d, L, 32) else float("inf"))
        t.load_from_numpy(ref.p)
        back = t.to_numpy(np.float32)
        record("structure", f"{size} init round trip", max(rel_err(back[k], ref.p[k]) for k in ref.p))
        # decay groups: 2-D params except pos, exactly pine.model_np.AdamW's rule
        want = {k for k, v in ref.p.items() if v.ndim == 2 and k != "pos"}
        got = {n for n, p in t.named_parameters() if MT.decays(n, p)}
        record("structure", f"{size} weight-decay set", 0.0 if want == got else float("inf"))


def check_grads_and_step(size: str, seed: int, perturb: bool, attn: str, T_use: int = 32, B: int = 12):
    tag = f"{size} seed{seed}{' perturbed' if perturb else ''} {attn} T={T_use}"
    ref = np_model(size, seed, perturb)
    t = torch_twin(ref, attn)
    K = 4
    _, _, inp, tgt = make_batch(B, T_use, K, seed + 7)
    x64, t64 = inp.astype(np.float64), tgt.astype(np.float64)

    # forward + backward, NumPy
    y, cache = ref.forward(x64)
    loss_np = float(((y - t64) ** 2).mean())
    g_np = ref.backward(2.0 * (y - t64) / y.size, cache)
    # forward + backward, torch
    xt, tt = torch.from_numpy(x64), torch.from_numpy(t64)
    yt = t(xt)
    loss_t = ((yt - tt) ** 2).mean()
    loss_t.backward()
    record("forward", f"{tag} output", rel_err(yt.detach().numpy(), y))
    record("loss", tag, abs(float(loss_t) - loss_np) / max(abs(loss_np), 1e-30))
    tp = dict(t.named_parameters())
    worst = (0.0, "")
    for k in ref.p:
        g = tp[k].grad
        e = rel_err(g.numpy(), g_np[k]) if g is not None else float("inf")
        record("grad", f"{tag} {k}", e)
        worst = max(worst, (e, k))

    # AdamW: step 1 with clip 1.0, step 2 with clipping forced on (clip = half the gradient norm)
    opt_np = AdamW(ref.p, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1)
    opt_t = MT.make_optimizer(t, lr=1e-3, wd=0.1, b1=0.9, b2=0.95, eps=1e-8, fused=False)
    params_t = [p for _, p in t.named_parameters()]
    for step, clip in ((1, 1.0), (2, None)):
        if step == 2:   # fresh gradients at the new weights
            y, cache = ref.forward(x64)
            g_np = ref.backward(2.0 * (y - t64) / y.size, cache)
            opt_t.zero_grad(set_to_none=True)
            yt = t(xt); ((yt - tt) ** 2).mean().backward()
            gn = math.sqrt(sum(float((g ** 2).sum()) for g in g_np.values()))
            clip = 0.5 * gn
        before_np = {k: v.copy() for k, v in ref.p.items()}
        before_t = {k: p.detach().clone() for k, p in t.named_parameters()}
        gn_np = opt_np.step(ref.p, g_np, 1e-3, clip=clip)
        gn_t = float(MT.clip_grads(params_t, clip))
        MT.set_lr(opt_t, 1e-3)
        opt_t.step()
        record("adamw", f"{tag} step{step} grad-norm", abs(gn_t - gn_np) / max(gn_np, 1e-30))
        for k in ref.p:
            d_np = ref.p[k] - before_np[k]
            d_t = (tp[k].detach() - before_t[k]).numpy()
            record("adamw", f"{tag} step{step} {k}", rel_err(d_t, d_np))
        record("adamw", f"{tag} step{step} clip active={'yes' if gn_np > clip else 'no'}", 0.0)
    return worst


def check_eval_and_utils():
    # evaluation metrics on the same weights and windows
    ref = np_model("S", 5, True)
    t = torch_twin(ref, "sdpa")
    r = np.random.default_rng(9)
    val_w = r.standard_normal((300, 36, F.D)).astype(np.float64)
    ref._std = np.abs(r.standard_normal(F.D)) + 0.1
    e_np = evaluate_np(ref, val_w, 32, 4)
    e_t = MT.evaluate_torch(t, val_w, ref._std, 32, 4, batch_size=128)
    for k in ("full", "exo", "hidden_joint_rmse_deg", "hidden_joint_rmse_deg_meanpose"):
        record("evaluate", k, abs(e_t[k] - e_np[k]) / max(abs(e_np[k]), 1e-30))
    record("evaluate", "per_window (rounded 5 dp)", float(np.abs(np.array(e_t["per_window"]) - np.array(e_np["per_window"])).max()), 1.01e-5)
    record("evaluate", "keys", 0.0 if list(e_t) == list(e_np) else float("inf"))
    # LR schedule: the expression in pine.train_np.run, verbatim
    for steps in (40, 900, 20000):
        lr, warm = 1e-3, max(20, steps // 20)
        worst = 0.0
        for step in range(1, steps + 1):
            ref_lr = lr * (step / warm if step <= warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / (steps - warm))))
            worst = max(worst, abs(MT.lr_at(step, steps, lr) - ref_lr) / lr)
        record("schedule", f"lr_at, {steps} steps", worst, 1e-12)
    # batch builder
    w, mask, inp, tgt = make_batch(16, 32, 4, 21)
    ti, tt = MT.batch_torch(torch.from_numpy(w), 32, 4, torch.from_numpy(mask))
    record("batch", "inputs", float(np.abs(ti.numpy() - inp).max()), 1e-12)
    record("batch", "targets", float(np.abs(tt.numpy() - tgt).max()), 1e-12)
    # masks: deterministic modes exactly, mixed mode in distribution (200k windows each)
    for mode in ("full", "exo"):
        a = MT.masks_torch(4, mode).numpy(); b = F.make_masks(4, mode, np.random.default_rng(0))
        record("masks", mode, float(np.abs(a - b).max()), 1e-12)
    n = 200_000
    g = torch.Generator().manual_seed(0)
    a = MT.masks_torch(n, "mixed", generator=g).numpy()
    b = F.make_masks(n, "mixed", np.random.default_rng(0))
    record("masks", "mixed: per-channel keep rate", float(np.abs(a.mean(0) - b.mean(0)).max()), 0.01)
    full_a, full_b = (a.min(1) == 1).mean(), (b.min(1) == 1).mean()
    exo_pat = np.ones(F.D); exo_pat[F.HIDDEN] = 0
    exo_a, exo_b = (a == exo_pat).all(1).mean(), (b == exo_pat).all(1).mean()
    record("masks", "mixed: share fully observed", abs(full_a - full_b), 0.01)
    record("masks", "mixed: share EXO pattern", abs(exo_a - exo_b), 0.01)


def check_cuda():
    if not torch.cuda.is_available():
        ROWS.append(("cuda", "no CUDA device: skipped", 0.0, 0.0, True))
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    ref = np_model("M", 4, True)
    _, _, inp, tgt = make_batch(32, 32, 4, 33)
    x64, t64 = inp.astype(np.float64), tgt.astype(np.float64)
    y, cache = ref.forward(x64)
    loss_np = float(((y - t64) ** 2).mean())
    g_np = ref.backward(2.0 * (y - t64) / y.size, cache)
    t = torch_twin(ref, "sdpa", device=dev, dtype=torch.float32)
    xt, tt = torch.from_numpy(inp).to(dev), torch.from_numpy(tgt).to(dev)
    loss = ((t(xt) - tt) ** 2).mean()
    loss.backward()
    record("cuda", "fp32 loss", abs(float(loss) - loss_np) / loss_np, 1e-3)
    tp = dict(t.named_parameters())
    record("cuda", "fp32 worst gradient", max(rel_err(tp[k].grad.double().cpu().numpy(), g_np[k]) for k in ref.p), 1e-3)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lb = float(((t(xt).float() - tt) ** 2).mean())
    record("cuda", "bf16 autocast loss", abs(lb - loss_np) / loss_np, 5e-2)
    record("cuda", f"device {torch.cuda.get_device_name(0)}", 0.0)


def main() -> int:
    t0 = time.time()
    torch.manual_seed(0)
    print(f"parity: torch {torch.__version__}, numpy {np.__version__}, robot {F.ROBOT}, D={F.D}", flush=True)
    sections = [("structure", check_structure)]
    for size, seed, perturb, attn, T_use in (("S", 0, False, "sdpa", 32), ("S", 1, True, "sdpa", 32),
                                            ("S", 1, True, "math", 32), ("M", 2, True, "sdpa", 32),
                                            ("XL", 3, True, "sdpa", 32), ("S", 4, True, "sdpa", 20)):
        sections.append((f"grads {size}", lambda a=(size, seed, perturb, attn, T_use): check_grads_and_step(*a)))
    sections += [("evaluate/utils", check_eval_and_utils), ("cuda", check_cuda)]
    for name, fn in sections:
        try:
            fn()
        except Exception as e:      # a crash is a failure, with the reason in the table
            ROWS.append((name, f"EXCEPTION {type(e).__name__}: {e}", float("inf"), 0.0, False))
    bad = [r for r in ROWS if not r[4]]
    width = max(len(r[1]) for r in ROWS)
    print(f"{'check':<10} {'item':<{width}} {'max rel err':>12} {'limit':>9}  ok", flush=True)
    for chk, item, v, lim, ok in ROWS:
        if not ok or os.environ.get("PINE_PARITY_VERBOSE"):
            print(f"{chk:<10} {item:<{width}} {v:>12.3e} {lim:>9.1e}  {'ok' if ok else 'FAIL'}", flush=True)
    by = {}
    for chk, _, v, lim, ok in ROWS:
        a = by.setdefault(chk, [0, 0, 0.0]); a[0] += 1; a[1] += (not ok)
        if np.isfinite(v) and lim > 0:
            a[2] = max(a[2], v / lim)
    for chk, (n, nbad, worst) in by.items():
        print(f"  {chk:<10} {n:4d} checks, {nbad} failed, worst {worst:.2e} of the limit", flush=True)
    if bad:
        print(f"PARITY FAILED: {len(bad)} of {len(ROWS)} checks ({time.time() - t0:.0f}s)", flush=True)
        return 1
    print(f"PARITY OK: {len(ROWS)} checks, max relative error < {TOL:g} in float64 ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
