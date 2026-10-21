"""One pretraining run on GPU(s): the PyTorch twin of pine/train_np.run, with the same record schema.

    torchrun --standalone --nproc_per_node=<gpus> gpu/train_torch.py --size 2XL --fraction 1.0 --steps 20000
    python gpu/train_torch.py --size S --steps 200            (one GPU, no torchrun)

Same as the CPU run: pine.train_np.SIZES, T=32 context, K=4 horizon, mixed masking (50% full, 30% EXO, 20% random
channel groups), MSE on normalised features, LR min(3e-3, 0.02/sqrt(d)) with warmup max(20, steps//20) and cosine to
10%, AdamW (b1 0.9, b2 0.95, wd 0.1 on 2-D params except pos), global-norm clip 1.0, NumPy initialisation, train-only
normalisation, the same train subset rule for fractions, val/test windows drawn with pine.features.Windows (seeds 7/11).
Different on purpose: global batch (default 512 windows), bf16 autocast, data-parallel over GPUs, more val windows.

Writes $PINE_RUNS/run_<size>_<fraction:.4f>.json (rank 0) with exactly pine.train_np.run's keys plus robot/device/data
info, and optionally weights_<size>.npz (NumPy dict, loadable into pine.model_np.MotionTransformer).
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402  (PINE_* defaults; repo root on sys.path)
from common import log, write_json, die  # noqa: E402
import numpy as np  # noqa: E402

try:
    import torch  # noqa: E402
    import torch.distributed as dist  # noqa: E402
    import torch.nn.functional as Fn  # noqa: E402
except ImportError:
    print("train_torch: torch is not installed", flush=True)
    sys.exit(2)

from pine import features as F            # noqa: E402
from pine.train_np import SIZES            # noqa: E402
import model_torch as MT                  # noqa: E402
import objectives as OB                   # noqa: E402


# ------------------------------------------------------------------------------------------------ data
class ShardData:
    """Duck-types pine.features.MotionData (clips, lens, starts, split, X, mean, std) from gpu/build_shards.py output,
    so pine.features.Windows and MotionData.subset are used unchanged."""
    subset = F.MotionData.subset
    hours = F.MotionData.hours

    def __init__(self, shard_dir: str):
        paths = {k: os.path.join(shard_dir, k) for k in ("features.npy", "clips.json", "stats.json")}
        for k, p in paths.items():
            if not os.path.exists(p):
                die(f"missing {p}: run gpu/build_shards.py first")
        st = json.load(open(paths["stats.json"]))
        if st.get("robot") != F.ROBOT:
            die(f"shards are for robot {st.get('robot')!r} but PINE_ROBOT={F.ROBOT!r}")
        if st.get("names") != list(F.NAMES) or int(st.get("D", -1)) != F.D:
            die("shard channel names differ from pine.features.NAMES; rebuild the shards")
        if not st.get("normalized", False):
            die("features.npy must hold train-normalised features (stats.json normalized=true)")
        cl = json.load(open(paths["clips.json"]))
        clips = cl["clips"] if isinstance(cl, dict) else cl
        self.clips = clips
        self.lens = np.array([int(c["length"]) for c in clips], dtype=np.int64)
        self.starts = np.array([int(c["start"]) for c in clips], dtype=np.int64)
        self.split = np.array([c["split"] for c in clips])
        if len(clips) == 0:
            die("clips.json is empty")
        if not np.array_equal(self.starts, np.concatenate([[0], np.cumsum(self.lens)[:-1]])):
            die("clips.json start/length are not contiguous")
        self.X = np.load(paths["features.npy"], mmap_mode="r")
        if self.X.ndim != 2 or self.X.shape[1] != F.D or self.X.shape[0] != int(self.lens.sum()):
            die(f"features.npy shape {self.X.shape} does not match clips.json ({int(self.lens.sum())} frames x {F.D})")
        self.mean = np.asarray(st["mean"], np.float64)
        self.std = np.asarray(st["std"], np.float64)
        self.stats = st


def train_starts(data: ShardData, wins: "F.Windows") -> np.ndarray:
    """Every valid window start of the training windows: sampling one uniformly is exactly Windows.sample's law
    (clip chosen with p proportional to its number of starts, offset uniform within it)."""
    ns = wins.n_starts.astype(np.int64)
    base = np.repeat(data.starts[wins.ids], ns)
    within = np.arange(int(ns.sum()), dtype=np.int64) - np.repeat(np.cumsum(ns) - ns, ns)
    return base + within


# ------------------------------------------------------------------------------------------------ chunk objective
@torch.no_grad()
def evaluate_chunk(model, obj, Bm, val_w, std, T: int, batch_size: int = 512) -> dict:
    """pine.train_np.evaluate's keys for the dct16mix objective: 'full'/'exo' are MSE on its coefficient targets
    (full mask / EXO mask); the hidden-joint error is the horizon-1 frame decoded from the coefficients, in degrees."""
    m = MT.unwrap(model)
    dev, dt = m.win.device, m.win.dtype
    hid = [j for j in F.HIDDEN if j in F.ANGLE_DIMS]
    std_h = torch.as_tensor(np.asarray(std, np.float64)[hid], device=dev)
    out = {}
    with torch.autocast(device_type=dev.type, enabled=False):
        for mode in ("full", "exo"):
            mask = torch.as_tensor(F.make_masks(1, mode, np.random.default_rng(0))[0], device=dev, dtype=dt)
            losses, se, se_base, n = [], 0.0, 0.0, 0
            for i in range(0, len(val_w), batch_size):
                w = torch.as_tensor(val_w[i:i + batch_size], device=dev).to(dt)
                x = w[:, :T] * mask[None, None, :]
                inp = torch.cat([x, mask[None, None, :].expand_as(x)], -1)
                y = m(inp)
                tgt = obj.targets(torch, Bm, w, x, T)
                losses.append(((y - tgt) ** 2).mean(dim=(1, 2)).double().cpu())
                if mode == "exo":
                    f1 = obj.first_frame(torch, Bm, y, x)
                    err = (f1[..., hid] - w[:, 1:T + 1, hid]).double() * std_h
                    se += float((torch.rad2deg(err) ** 2).sum()); n += err.numel()
                    se_base += float((torch.rad2deg(w[:, 1:T + 1, hid].double() * std_h) ** 2).sum())
            pw = torch.cat(losses).numpy()
            out[mode] = float(pw.mean())
            if mode == "full":
                out["per_window"] = np.round(pw, 5).tolist()
            if mode == "exo":
                out["hidden_joint_rmse_deg"] = math.sqrt(se / n)
                out["hidden_joint_rmse_deg_meanpose"] = math.sqrt(se_base / n)
    return out


class NumpyFacade:
    """Lets scripts/pi_pass.score drive a torch model unchanged: forward(x) -> (y, None), fp32, no autocast."""

    def __init__(self, model):
        self.m = MT.unwrap(model)

    @torch.no_grad()
    def forward(self, x):
        dev = self.m.win.device
        with torch.autocast(device_type=dev.type, enabled=False):
            y = self.m(torch.as_tensor(np.asarray(x, np.float32), device=dev).to(self.m.win.dtype))
        return y.double().cpu().numpy(), None


def pi_pass_scores(data, model, objective: str, out_dir: str, size: str) -> dict:
    """scripts/pi_pass.score on its own 200 held-out test windows (seed 21): free-running forecasts, joint-angle error
    in degrees, vs holding the last pose. Also scores the next-4-frame model of the same size, if its weights exist."""
    import importlib
    pi = importlib.import_module("scripts.pi_pass")
    W = np.asarray(F.Windows(data, data.subset("test"), 32 + 40 + 16 - 4, 4).fixed(200, seed=21), np.float32)
    res = {}
    if objective == OB.NAME:   # with every channel measured, dct16mix decodes exactly like pi_pass's residual dct16
        res[objective] = pi.score(data, NumpyFacade(model), pi.Objective("dct16", OB.H, True, k=OB.K_COEF,
                                                                           execute=OB.EXECUTE), W)
    wp = os.path.join(out_dir, f"weights_{size}.npz")
    if os.path.exists(wp):
        d, L, H = SIZES[size]
        base = MT.MotionTransformerTorch(2 * F.D, 4 * F.D, d, L, H, 32).to(MT.unwrap(model).win.device)
        base.load_from_numpy(dict(np.load(wp)))
        res["next4_baseline"] = pi.score(data, NumpyFacade(base), pi.Objective("direct4", 4, False, execute=1), W)
    return res


# ------------------------------------------------------------------------------------------------ distributed
def setup_dist():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            die("WORLD_SIZE > 1 needs CUDA (NCCL)")
        torch.cuda.set_device(local)
        dist.init_process_group("nccl")
    return world, rank, local


def all_mean(x: float, world: int, dev) -> float:
    if world == 1:
        return x
    t = torch.tensor([x], device=dev, dtype=torch.float64)
    dist.all_reduce(t)
    return float(t) / world


def any_flag(flag: bool, world: int, dev) -> bool:
    if world == 1:
        return flag
    t = torch.tensor([1.0 if flag else 0.0], device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return bool(t.item() > 0)


# ------------------------------------------------------------------------------------------------ run
def run(a) -> int:
    world, rank, local = setup_dist()
    is0 = rank == 0
    say = (lambda m: log(m, "train")) if is0 else (lambda m: None)
    if a.size not in SIZES:
        die(f"unknown size {a.size!r}; known: {', '.join(SIZES)}")
    d, L, H = SIZES[a.size]
    T, K = a.context, a.horizon
    chunk = a.objective == OB.NAME
    obj = OB.ChunkObjective(F.D) if chunk else None
    Kw = obj.H if chunk else K                       # future frames each window carries
    dout = obj.out_dim if chunk else K * F.D
    if a.batch % world:
        die(f"global batch {a.batch} is not divisible by {world} processes")
    b_local = a.batch // world
    if a.device == "cuda" and not torch.cuda.is_available():
        die("CUDA is not available (use --device cpu for a debug run)")
    dev = torch.device(f"cuda:{local}" if a.device == "cuda" else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False          # evaluation stays true fp32
    torch.backends.cudnn.allow_tf32 = False
    bf16 = a.precision == "bf16" and dev.type == "cuda"
    if bf16 and not torch.cuda.is_bf16_supported():
        say("WARNING: this GPU has no bf16; training in fp32"); bf16 = False

    t_load = time.time()
    data = ShardData(a.shards)
    tr_ids = data.subset("train", a.fraction, seed=a.seed)
    trw = F.Windows(data, tr_ids, T, Kw, seed=a.seed)
    va_ids, te_ids = data.subset("val"), data.subset("test")
    if len(trw.ids) == 0 or len(va_ids) == 0 or len(te_ids) == 0:
        die(f"empty split: train {len(trw.ids)} val {len(va_ids)} test {len(te_ids)} clips")
    vaw = np.asarray(F.Windows(data, va_ids, T, Kw).fixed(a.val_windows, seed=7), np.float32)
    tew = np.asarray(F.Windows(data, te_ids, T, Kw).fixed(a.val_windows, seed=11), np.float32)
    starts = torch.from_numpy(train_starts(data, trw)).to(dev)
    X = torch.from_numpy(np.ascontiguousarray(data.X)).to(dev)            # float16, normalised
    ar = torch.arange(T + Kw, device=dev)
    say(f"{a.size} f={a.fraction}: {len(trw.ids)} train clips, {trw.frames:,} frames ({trw.frames/20/3600:.2f} h), "
        f"{len(starts):,} window starts, {world} x {b_local} windows/step, data load {time.time()-t_load:.0f}s")

    model = MT.MotionTransformerTorch.like_numpy(2 * F.D, dout, d, L, H, T, seed=a.seed).to(dev)
    Bm = torch.as_tensor(obj.B, device=dev) if chunk else None
    if chunk and is0:   # the objective's torch path must equal scripts/pi_pass before any time is spent on it
        chk = OB.check_against_pi_pass(obj, torch, Bm, to_lib=lambda z: torch.as_tensor(z, device=dev),
                                       to_np=lambda z: z.detach().double().cpu().numpy())
        bad = {k: v for k, v in chk.items() if not v < 1e-4}
        if bad:
            die(f"dct16mix objective does not match scripts/pi_pass: {bad}")
        say(f"objective {a.objective}{' + noisy history' if a.noisy_history else ''}: matches scripts/pi_pass "
            f"(max diff {max(chk.values()):.1e})")
    gen_aug = torch.Generator(device=dev).manual_seed(a.seed * 1000 + 1017 + rank)
    n_params = model.n_params
    lr = a.lr or min(3e-3, 0.02 / math.sqrt(d))
    opt = MT.make_optimizer(model, lr=lr, wd=0.1, b1=0.9, b2=0.95, eps=1e-8)
    params = [p for p in model.parameters()]
    net = model
    if world > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local], broadcast_buffers=False)
    nets = {"eager": net}
    if a.compile:
        try:
            nets["compiled"] = torch.compile(net)
        except Exception as e:  # noqa: BLE001
            say(f"WARNING: torch.compile failed ({e}); running eager"); a.compile = False
    gen = torch.Generator(device=dev).manual_seed(a.seed * 1000 + 17 + rank)

    def forward(inp):
        if a.compile:
            try:
                return nets["compiled"](inp)
            except Exception as e:  # noqa: BLE001  (compile errors surface on the first call)
                if step_fn.calls > 1:
                    raise
                say(f"WARNING: compiled model failed on the first step ({type(e).__name__}: {e}); running eager")
                a.compile = False
        return nets["eager"](inp)

    def step_fn(step: int) -> torch.Tensor:
        step_fn.calls += 1
        s = starts[torch.randint(len(starts), (b_local,), generator=gen, device=dev)]
        w = X[s[:, None] + ar[None]].float()                                  # (b, T+K, D)
        mask = MT.masks_torch(b_local, "mixed", generator=gen, device=dev)
        if chunk:
            ctx = w[:, :T]
            if a.noisy_history:
                ctx = OB.noisy_history_torch(ctx, mask, gen_aug)
            x = ctx * mask[:, None, :]
            inp = torch.cat([x, mask[:, None, :].expand_as(x)], -1)
            tgt = obj.targets(torch, Bm, w, x, T)
        else:
            inp, tgt = MT.batch_torch(w, T, K, mask)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=bf16):
            y = forward(inp)
        loss = Fn.mse_loss(y.float(), tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        MT.clip_grads(params, a.clip)
        MT.set_lr(opt, MT.lr_at(step, a.steps, lr))
        opt.step()
        return loss.detach()

    step_fn.calls = 0

    def evaluate(windows):
        if chunk:
            return evaluate_chunk(model, obj, Bm, windows, data.std, T)
        return MT.evaluate_torch(model, windows, data.std, T, K)
    eval_every = a.eval_every or max(1, a.steps // 20)
    curve, best, t0 = [], None, time.time()
    train_s, t_mark, stopped, step = 0.0, time.time(), None, 0
    progress = os.path.join(a.out, "progress.json")
    for step in range(1, a.steps + 1):
        loss_t = step_fn(step)
        if step == 1 and a.compile:
            say("compiled first step done")
        last = step == a.steps
        if step % 50 == 0 or last or step % eval_every == 0:
            late = a.deadline > 0 and time.time() > a.deadline
            if any_flag(late, world, dev) and not last:
                stopped = "deadline"
        if step % eval_every == 0 or last or stopped:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            train_s += time.time() - t_mark
            loss = all_mean(float(loss_t), world, dev)
            if not math.isfinite(loss):
                die(f"{a.size}: training loss is {loss} at step {step}")
            if is0:
                ev = evaluate(vaw)
                curve.append(dict(step=step, train=loss, val=ev["full"], val_exo=ev["exo"],
                                  hidden_rmse=ev["hidden_joint_rmse_deg"], sec=time.time() - t0))
                if best is None or ev["full"] < best["val"]["full"]:
                    best = dict(step=step, val=ev, test=evaluate(tew),
                                weights=model.to_numpy(np.float32) if (a.save_weights or chunk) else None)
                tps = step * a.batch * T / max(train_s, 1e-9)
                say(f"  {a.size} f={a.fraction:<8} step {step:6d}/{a.steps} train {loss:.4f} val {ev['full']:.4f} "
                    f"exo {ev['exo']:.4f} hidden {ev['hidden_joint_rmse_deg']:.1f}deg  {tps/1e6:.2f}M tok/s  "
                    f"{time.time()-t0:.0f}s")
                write_json(progress, dict(size=a.size, fraction=a.fraction, step=step, steps=a.steps, train=loss,
                                          val=ev["full"], tokens_per_s=tps, sec=time.time() - t0, at=time.time()))
            if world > 1:
                dist.barrier()
            t_mark = time.time()
        if stopped:
            say(f"  {a.size}: stopping at step {step} of {a.steps} ({stopped})")
            break
    steps_done = step
    if is0:
        tokens = steps_done * a.batch * T
        rec = dict(size=a.size, d_model=d, layers=L, heads=H, params=int(n_params),
                   params_nonembed=int(L * 12 * d * d), fraction=a.fraction, train_frames=trw.frames,
                   train_hours=trw.frames / 20 / 3600, steps=steps_done, batch=a.batch, context=T, horizon=Kw, lr=lr,
                   tokens=tokens, flops=6.0 * n_params * tokens, epochs=tokens / max(1, trw.frames),
                   best_step=best["step"], val=best["val"], test=best["test"], curve=curve,
                   wall_s=time.time() - t0)
        rec.update(
            trainer="gpu/train_torch.py", robot=F.ROBOT, steps_planned=a.steps, stopped_early=stopped,
            val_windows=a.val_windows, seed=a.seed,
            data=dict(shards=a.shards, hours_total=data.stats.get("hours"), hours_by_split=data.stats.get("hours_by_split"),
                      sources=sorted((data.stats.get("per_source") or {}).keys())),
            device=dict(type=dev.type, name=torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                        count=world, torch=torch.__version__, cuda=torch.version.cuda, precision="bf16" if bf16 else "fp32",
                        compile=bool(a.compile), tokens_per_s=steps_done * a.batch * T / max(train_s, 1e-9),
                        train_seconds=train_s,
                        peak_mem_gb=(torch.cuda.max_memory_allocated(dev) / 1e9) if dev.type == "cuda" else None))
        os.makedirs(a.out, exist_ok=True)
        name = f"run_{a.size}_{a.fraction:.4f}"
        tag = ""
        if chunk:   # an extra experiment: never named run_*.json, so ladder consumers and fits cannot pick it up
            tag = f"_{a.objective}" + ("_noisyhist" if a.noisy_history else "")
            name = f"extra_{a.size}_{a.fraction:.4f}{tag}"
            rec.update(objective=dict(name=a.objective, chunk_frames=obj.H, dct_coeffs=obj.k, execute=obj.execute,
                                      targets="offsets from the masked input frame (measured: residual, masked: absolute)",
                                      noisy_history=bool(a.noisy_history)))
            best_w = best["weights"] or model.to_numpy(np.float32)
            final = model.to_numpy(np.float32)
            model.load_from_numpy(best_w)           # score the best checkpoint, like pi_pass
            try:
                rec["pi_pass_score"] = pi_pass_scores(data, model, a.objective, a.out, a.size)
            except Exception as e:  # noqa: BLE001  (the score is extra; the record is still written)
                rec["pi_pass_score"] = dict(error=f"{type(e).__name__}: {e}"[:500])
            model.load_from_numpy(final)
        else:
            rec["objective"] = dict(name="next4")
        if a.save_weights and best["weights"] is not None:
            wname = (f"weights_{a.size}.npz" if a.fraction == 1.0 else f"weights_{a.size}_{a.fraction:.4f}.npz")
            wname = wname.replace(".npz", f"{tag}.npz")
            tmp = os.path.join(a.out, wname + ".tmp.npz")
            np.savez(tmp, **best["weights"])
            os.replace(tmp, os.path.join(a.out, wname))
            rec["weights"] = wname
        write_json(os.path.join(a.out, name + ".json"), rec, indent=None)
        say(f"   {a.size} f={a.fraction}: params {n_params:,} best val {rec['val']['full']:.4f} test {rec['test']['full']:.4f} "
            f"hidden {rec['val']['hidden_joint_rmse_deg']:.1f} deg, {rec['wall_s']:.0f}s, "
            f"{rec['device']['tokens_per_s']/1e6:.2f}M tok/s -> {name}.json")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", required=True)
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=512, help="global batch in windows")
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=0, help="default: steps // 20")
    ap.add_argument("--val-windows", type=int, default=2048)
    ap.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (off by default)")
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--deadline", type=float, default=0.0, help="unix time; stop early (and still write the record)")
    ap.add_argument("--save-weights", action="store_true")
    ap.add_argument("--objective", choices=("next4", OB.NAME), default="next4",
                    help="next4 = the CPU recipe (default, used by the whole ladder); dct16mix = 16-frame chunks as 6 "
                         "DCT coefficients, residual on measured / absolute on masked channels (one extra run)")
    ap.add_argument("--noisy-history", action="store_true", help="dct16mix only: noisy-history context augmentation")
    ap.add_argument("--shards", default=common.SHARDS)
    ap.add_argument("--out", default=common.RUNS)
    a = ap.parse_args(argv)
    if not 0.0 < a.fraction <= 1.0:
        ap.error("--fraction must be in (0, 1]")
    if a.steps < 1 or a.batch < 1:
        ap.error("--steps and --batch must be positive")
    if a.context != 32:
        ap.error("the CPU reference and the record schema use T=32")
    if a.noisy_history and a.objective != OB.NAME:
        ap.error("--noisy-history is only defined for --objective dct16mix")
    return a


if __name__ == "__main__":
    sys.exit(run(parse()))
