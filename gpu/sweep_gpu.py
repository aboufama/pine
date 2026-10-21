"""GPU scaling sweep with a pre-registered prediction and a budget guard.

    python gpu/sweep_gpu.py              (on the pod, after gpu/build_shards.py; run_all.sh does this)
    python gpu/sweep_gpu.py --dry-run    (print the schedule and the budget estimate, train nothing)

1. Model ladder (gpu/plan_lib.LADDER minus its largest size) on all training data.
2. Fit L(N) = E + A N^-alpha with pine.scaling.bootstrap_prediction and write $PINE_RUNS/prediction.json, timestamped,
   BEFORE the largest size starts training. An existing prediction is never rewritten, and none is written after the
   fact if the largest run already exists.
3. Train the largest size. 4. Data ladder: DATA_SIZE on fractions of the training hours (+ its full-data run).
5. Write fits.json (prediction vs observed, L(D)) and sweep_status.json.

Budget guard: PINE_MAX_GPU_HOURS (default 8) counts GPU-hours since run_all.sh started (PINE_T0) plus
PINE_BUDGET_SLACK_H (default 0.25 h for pod start, upload and packing). A run is not started if its estimate (from the
tokens/s measured on earlier runs, or the plan's 25%-MFU model before the first measurement) would cross the cap,
and every run gets a hard deadline at the cap so it stops early and still writes its record.
Runs are resumable: a run whose record exists is skipped.
"""
from __future__ import annotations
import argparse, glob, json, math, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
from common import log, write_json, env_float, env_int  # noqa: E402
import plan_lib as P  # noqa: E402
from pine import features as F  # noqa: E402
from pine.model_np import count_params  # noqa: E402
from pine.train_np import SIZES  # noqa: E402

SAY = lambda m: log(m, "sweep")  # noqa: E731
LAUNCH_OVERHEAD_S = 45          # python + torch import + shard load, on top of the measured wall tokens/s


def n_gpus() -> int:
    v = os.environ.get("PINE_GPUS", "").strip()
    if v:
        return max(1, int(v))
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        if n:
            return n
    except (OSError, subprocess.SubprocessError):
        pass
    return max(1, int(os.environ.get("RUNPOD_GPU_COUNT", "1") or 1))


def gpu_spec() -> dict:
    """Peak dense BF16 TFLOPS of this GPU from the plan table (nvidia-smi names equal RunPod's gpuTypeIds); an unknown
    GPU of a known family gets that family's lowest entry, anything else a deliberately low 100 TFLOPS."""
    name = ""
    try:
        name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30).stdout.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    key = " ".join(name.upper().split())
    for g in P.GPUS:
        if " ".join(g["id"].upper().split()) == key:
            return dict(g, detected=name)
    for fam in ("H100", "H200", "A100", "L40S"):
        if fam in key:
            cands = [g for g in P.GPUS if fam in g["id"].upper()] or [g for g in P.GPUS if "H100" in g["id"].upper()]
            return dict(min(cands, key=lambda g: g["bf16_tflops"]), detected=name)
    return dict(id=name or "unknown", name=name or "unknown", bf16_tflops=100.0, usd_per_hr=0.0, detected=name)


class Budget:
    def __init__(self, cap_h: float, gpus: int, t0: float, slack_h: float, reserve_s: float = 300.0):
        self.cap_h, self.gpus, self.t0, self.slack_h, self.reserve_s = cap_h, gpus, t0, slack_h, reserve_s

    def used_h(self, now: float | None = None) -> float:
        return ((now or time.time()) - self.t0) / 3600.0 * self.gpus + self.slack_h

    def left_s(self, now: float | None = None) -> float:
        return (self.cap_h - self.used_h(now)) / self.gpus * 3600.0

    def deadline(self) -> float:
        """Unix time at which training must stop so packing up still fits under the cap."""
        return self.t0 + ((self.cap_h - self.slack_h) / self.gpus) * 3600.0 - self.reserve_s

    def allows(self, est_s: float, now: float | None = None) -> bool:
        return est_s + self.reserve_s <= self.left_s(now)


def estimate_seconds(params: int, steps: int, batch: int, measured: list[dict], spec: dict, gpus: int) -> float:
    """Wall seconds for one run. With measurements: the smallest measured model at least this big runs at its own
    tokens/s; a bigger model is scaled down by the parameter ratio (compute-bound, conservative)."""
    tokens = steps * batch * P.T
    if measured:
        ge = sorted([m for m in measured if m["params"] >= params], key=lambda m: m["params"])
        ref = ge[0] if ge else max(measured, key=lambda m: m["params"])
        tps = ref["wall_tps"] * min(1.0, ref["params"] / params)
        return tokens / max(tps, 1.0) + LAUNCH_OVERHEAD_S
    return P.run_seconds(params, spec, gpus, steps, batch)


def run_path(out: str, size: str, frac: float) -> str:
    return os.path.join(out, f"run_{size}_{frac:.4f}.json")


def load_run(out: str, size: str, frac: float):
    p = run_path(out, size, frac)
    try:
        return json.load(open(p)) if os.path.exists(p) else None
    except (OSError, ValueError):
        return None


def measured_from(rec: dict) -> dict | None:
    try:
        return dict(params=int(rec["params"]), wall_tps=float(rec["tokens"]) / max(float(rec["wall_s"]), 1e-9))
    except (KeyError, TypeError, ValueError):
        return None


def register_prediction(out: str, ladder_recs: list[dict], size: str, steps: int) -> dict | None:
    path = os.path.join(out, "prediction.json")
    if os.path.exists(path):
        SAY(f"prediction already registered: {path} (kept as is)")
        return json.load(open(path))
    if load_run(out, size, 1.0):
        SAY(f"NOT registering a prediction: run_{size}_1.0000.json already exists (it would not be a prediction)")
        return None
    pts = [r for r in ladder_recs if r and not r.get("stopped_early") and r.get("val", {}).get("per_window")]
    if len(pts) < 3:
        SAY(f"NOT registering a prediction: only {len(pts)} complete ladder runs (need 3)")
        return None
    from pine import scaling   # scipy
    d, L, H = SIZES[size]
    n_new = count_params(2 * F.D, P.K * F.D, d, L, P.T)
    pts = sorted(pts, key=lambda r: r["params"])
    bp = scaling.bootstrap_prediction([r["params"] for r in pts], [r["val"]["per_window"] for r in pts], n_new)
    pred = dict(size=size, params=n_new, steps=steps, registered_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                registered_unix=time.time(), fit_on=[r["size"] for r in pts], form="L(N) = E + A * N^-alpha",
                robot=F.ROBOT, metric="val full-observation MSE (normalised features), best checkpoint", **bp)
    write_json(path, pred)
    SAY(f"PRE-REGISTERED: {size} ({n_new:,} params) val loss {bp['loss']:.4f} [{bp['lo']:.4f}, {bp['hi']:.4f}] "
        f"fit on {pred['fit_on']} -> {path}")
    return pred


def write_fits(out: str) -> None:
    recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out, "run_*.json")))]
    pred = json.load(open(os.path.join(out, "prediction.json"))) if os.path.exists(os.path.join(out, "prediction.json")) else None
    res = dict(robot=F.ROBOT, model=None, data=None)
    if pred:
        lad = sorted([r for r in recs if r["fraction"] == 1.0 and r["size"] in pred["fit_on"]], key=lambda r: r["params"])
        obs = next((r for r in recs if r["size"] == pred["size"] and r["fraction"] == 1.0), None)
        res["model"] = dict(form=pred["form"], E=pred["E"], A=pred["A"], alpha=pred["alpha"], fit_on=pred["fit_on"],
                            registered_at=pred["registered_at"],
                            points=[dict(size=r["size"], params=r["params"], loss=r["val"]["full"]) for r in lad],
                            prediction=dict(size=pred["size"], params=pred["params"], loss=pred["loss"], lo=pred["lo"], hi=pred["hi"]),
                            observed=dict(size=obs["size"], params=obs["params"], loss=obs["val"]["full"],
                                          inside_interval=bool(pred["lo"] <= obs["val"]["full"] <= pred["hi"]),
                                          stopped_early=obs.get("stopped_early")) if obs else None)
    dl = sorted([r for r in recs if r["size"] == P.DATA_SIZE and not r.get("stopped_early")], key=lambda r: r["train_hours"])
    if len(dl) >= 3:
        try:
            from pine import scaling
            import numpy as np
            x = [r["train_hours"] for r in dl]; y = [r["val"]["full"] for r in dl]
            E, B, b = scaling.fit(x, y)
            hrs = np.geomspace(min(x), 10000, 40)
            res["data"] = dict(form="L(D) = E + B * D^-beta", E=float(E), B=float(B), beta=float(b), size=P.DATA_SIZE,
                               points=[dict(hours=h, loss=l, fraction=r["fraction"]) for h, l, r in zip(x, y, dl)],
                               curve=[dict(hours=float(h), loss=float(scaling.law(h, E, B, b))) for h in hrs])
        except Exception as e:  # noqa: BLE001
            SAY(f"L(D) fit failed: {e}")
    write_json(os.path.join(out, "fits.json"), res)


def train_cmd(size: str, frac: float, steps: int, batch: int, gpus: int, deadline: float, save_weights: bool,
              out: str) -> list[str]:
    script = os.path.join(HERE, "train_torch.py")
    args = ["--size", size, "--fraction", repr(frac), "--steps", str(steps), "--batch", str(batch),
            "--val-windows", str(env_int("PINE_VAL_WINDOWS", P.VAL_WINDOWS)), "--deadline", f"{deadline:.0f}",
            "--out", out, "--shards", common.SHARDS]
    if save_weights:
        args.append("--save-weights")
    if os.environ.get("PINE_COMPILE", "0") == "1":
        args.append("--compile")
    if gpus > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}", script] + args
    return [sys.executable, script] + args


def extra_run(a, plan, status, steps, batch, gpus, spec, budget, measured) -> dict:
    """One extra run after the ladder (it never feeds the fits): the largest size that finished, trained with the
    dct16mix objective + noisy history, scored with scripts/pi_pass.score. Written as extra_<size>_1.0000_*.json."""
    done = [r for r in status if r.get("status") in ("ok", "done") and r["fraction"] == 1.0]
    if a.dry_run:
        done = [r for r in status if r["fraction"] == 1.0]
    if not done:
        SAY("extra run skipped: no finished full-data run")
        return dict(role="extra", status="skipped_no_base")
    base = max(done, key=lambda r: r["params"])
    size = base["size"]
    d, L, H = SIZES[size]
    import objectives as OB
    params = count_params(2 * F.D, OB.K_COEF * F.D, d, L, P.T)
    est = estimate_seconds(params, steps, batch, measured, spec, gpus) * 1.15 + 120     # longer windows + pi_pass score
    row = dict(role="extra", size=size, fraction=1.0, objective=OB.NAME, noisy_history=True, params=params,
               est_seconds=round(est))
    path = os.path.join(a.out, f"extra_{size}_1.0000_{OB.NAME}_noisyhist.json")
    if a.dry_run:
        SAY(f"  plan {'extra':<12} {size:<4} f={1.0:<9.4g} {params:>11,} params  est {est/60:6.1f} min  ({OB.NAME} + noisy history)")
        return row
    if os.path.exists(path):
        SAY(f"skip extra {size} {OB.NAME}: done"); row["status"] = "done"; return row
    if not budget.allows(est):
        SAY(f"BUDGET: not starting the extra {OB.NAME} run on {size} (est {est/60:.0f} min, {budget.left_s()/60:.0f} min left)")
        row["status"] = "skipped_budget"; return row
    cmd = train_cmd(size, 1.0, steps, batch, gpus, budget.deadline(), False, a.out) + \
        ["--objective", OB.NAME, "--noisy-history"]
    SAY(f"== extra {size} {OB.NAME} + noisy history: {params:,} params, est {est/60:.0f} min")
    t_run = time.time()
    try:
        rc = subprocess.run(cmd, timeout=max(600.0, budget.deadline() - time.time() + 900.0)).returncode
    except subprocess.TimeoutExpired:
        rc = "timeout"
    row.update(status="ok" if os.path.exists(path) else f"failed ({rc})", wall_s=round(time.time() - t_run))
    if os.path.exists(path):
        rec = json.load(open(path))
        sc = rec.get("pi_pass_score", {})
        try:
            e = sc[OB.NAME][f"execute_{OB.EXECUTE}"]; b = sc.get("next4_baseline", {}).get("execute_1", {})
            SAY(f"   extra {size}: free-running mean 2 s {e['mean_2s']:.1f} deg (execute {OB.EXECUTE}); next-4 model "
                f"{b.get('mean_2s', float('nan')):.1f}; hold pose {sc[OB.NAME]['hold_pose']['mean_2s']:.1f}")
        except (KeyError, TypeError):
            SAY(f"   extra {size}: pi_pass score unavailable ({str(sc)[:200]})")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=common.RUNS)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    stats_p = os.path.join(common.SHARDS, "stats.json")
    if not os.path.exists(stats_p):
        common.die(f"{stats_p} is missing: run gpu/build_shards.py first")
    stats = json.load(open(stats_p))
    train_frames = int(stats["frames_by_split"]["train"])
    batch = env_int("PINE_BATCH", P.BATCH)
    steps = env_int("PINE_STEPS", 0) or P.steps_for(train_frames, batch)
    gpus = n_gpus()
    spec = gpu_spec()
    t0 = env_float("PINE_T0", time.time())
    budget = Budget(env_float("PINE_MAX_GPU_HOURS", P.DEFAULT_MAX_GPU_HOURS), gpus, t0, env_float("PINE_BUDGET_SLACK_H", 0.25))
    plan = P.runs(SIZES)
    SAY(f"sweep: robot {F.ROBOT}, {stats.get('hours', 0):.1f} h of motion ({train_frames/72000:.1f} h train), "
        f"{steps} steps x {batch} windows, {gpus} x {spec['name']}, cap {budget.cap_h} GPU-h, "
        f"used so far {budget.used_h():.2f} GPU-h")
    measured = [m for m in (measured_from(r) for r in (load_run(a.out, x["size"], x["fraction"]) for x in plan) if r) if m]
    status = []
    for item in plan:
        size, frac, role = item["size"], item["fraction"], item["role"]
        d, L, H = SIZES[size]
        params = count_params(2 * F.D, P.K * F.D, d, L, P.T)
        est = estimate_seconds(params, steps, batch, measured, spec, gpus)
        row = dict(item, params=params, est_seconds=round(est))
        if a.dry_run:
            SAY(f"  plan {role:<12} {size:<4} f={frac:<9.4g} {params:>11,} params  est {est/60:6.1f} min")
            status.append(row); continue
        if load_run(a.out, size, frac):
            SAY(f"skip {size} f={frac}: done"); row["status"] = "done"; status.append(row); continue
        if role == "predicted":
            ladder = [load_run(a.out, x["size"], 1.0) for x in plan if x["role"] == "model_ladder"]
            register_prediction(a.out, ladder, size, steps)
        if not budget.allows(est):
            SAY(f"BUDGET: not starting {size} f={frac} (est {est/60:.0f} min, {budget.left_s()/60:.0f} min left "
                f"under PINE_MAX_GPU_HOURS={budget.cap_h})")
            row["status"] = "skipped_budget"; status.append(row); write_json(os.path.join(a.out, "sweep_status.json"), status)
            continue
        keep = role == "predicted" or (size == P.DATA_SIZE and frac == 1.0)
        cmd = train_cmd(size, frac, steps, batch, gpus, budget.deadline(), keep, a.out)
        SAY(f"== {role} {size} f={frac}: {params:,} params, est {est/60:.0f} min, {budget.left_s()/60:.0f} min left")
        t_run = time.time()
        try:
            rc = subprocess.run(cmd, timeout=max(600.0, budget.deadline() - time.time() + 900.0)).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
        rec = load_run(a.out, size, frac)
        row.update(status="ok" if rec else f"failed ({rc})", wall_s=round(time.time() - t_run))
        if rec:
            m = measured_from(rec)
            if m:
                measured.append(m)
            if rec.get("stopped_early"):
                row["status"] = "stopped_early"
        else:
            SAY(f"run {size} f={frac} FAILED (exit {rc}); continuing with the next run")
        status.append(row)
        write_json(os.path.join(a.out, "sweep_status.json"), status)
    if os.environ.get("PINE_EXTRA", "1") != "0":
        status.append(extra_run(a, plan, status, steps, batch, gpus, spec, budget, measured))
        if not a.dry_run:
            write_json(os.path.join(a.out, "sweep_status.json"), status)
    if not a.dry_run:
        write_fits(a.out)
        write_json(os.path.join(a.out, "sweep_status.json"), status)
        ok = sum(1 for r in status if r.get("status") in ("ok", "done"))
        SAY(f"sweep finished: {ok}/{len(status)} runs complete, {budget.used_h():.2f} GPU-h used of {budget.cap_h}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
