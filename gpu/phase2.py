"""Phase 2 on the pod: does a new person's motion get predicted better by more people or by more hours?

Runs after phase 1 (gpu/sweep_gpu.py) on the same volume. Two datasets name their performers, so whole people can be
held out: HiPHI (132 performers, already in the phase-1 shards) and BONES-SEED (522 actors, added by gpu/seed_data.py,
which phase 2 starts in the background and which writes $WORKSPACE/data/shards_all = phase-1 shards + SEED).

  split    per dataset: performers with crc32(subject) % 10 == 3 are TEST (never trained on), 1 in 20 takes of the
           others is VAL (new takes of seen people), the rest is that dataset's pool
  sweeps   H hours split evenly across P performers of one dataset, every run the same size (2XL) and the same
           number of steps, scored on that dataset's held-out performers:
             HiPHI  P = 4, 8, 16, 32, 64 (H = 16 h, or less if too few performers have H/4)  + hours at P = 32
             SEED   P = 8, 16, 32, 64, 128, 256, 400 (H from the 8th-best actor)            + hours at P = 128
           first draw of each sweep, then more draws while the budget lasts
  E3       everything except held-out performers (HiPHI + SEED pools and the other sources' training takes):
           the CPU recipe vs dct16mix + noisy history at 2XL, then the FLAGSHIP: 5XL (~85M parameters) with the new
           recipe for the rest of the budget, scored on held-out HiPHI performers
Results: $PINE_RUNS/phase2/*.json and summary.json.
"""
from __future__ import annotations
import glob, json, os, subprocess, sys, time, zlib
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402

SAY = lambda m: common.log(m, "phase2")  # noqa: E731
OUT = os.path.join(common.RUNS, "phase2"); SPLITS = os.path.join(OUT, "splits")
SH1 = common.SHARDS
SHALL = os.path.join(os.path.dirname(common.SHARDS), "shards_all")
FPS, MIN_LEN, TOK_PER_STEP, SWEEP_SIZE = 20.0, 36, 512 * 32, "2XL"
DEADLINE = float(os.environ.get("PINE_PHASE2_DEADLINE", time.time() + 4 * 3600))


# ----------------------------------------------------------------------------------------------- speed and budget
def run_records():
    for p in glob.glob(os.path.join(common.RUNS, "run_*.json")) + glob.glob(os.path.join(OUT, "phase2_*.json")):
        try:
            yield json.load(open(p))
        except Exception:  # noqa: BLE001
            pass


def tps_for(size: str) -> float:
    """Tokens/s for `size`: a measured run of that size, else the largest measured run scaled by parameter count."""
    sys.path.insert(0, os.path.dirname(HERE))
    from pine.train_np import SIZES
    from pine.model_np import count_params
    import pine.features as F
    d, L, H = SIZES[size]
    n = count_params(2 * F.D, 4 * F.D, d, L, 32)
    best = None
    for r in run_records():
        t = float((r.get("device") or {}).get("tokens_per_s") or 0)
        if t <= 0:
            continue
        if r.get("size") == size:
            return t
        if best is None or r["params"] > best[0]:
            best = (r["params"], t)
    return best[1] * best[0] / n if best else 0.0


def steps_for(size, seconds, lo, hi):
    t = tps_for(size)
    return int(min(hi, max(lo, seconds * t / TOK_PER_STEP))) if t else lo


STEPS = int(os.environ.get("PINE_PHASE2_STEPS", 0)) or steps_for(SWEEP_SIZE, 150, 3000, 20000)   # ~2.5 min per run
left = lambda: DEADLINE - time.time()


# ----------------------------------------------------------------------------------------------- splits
def load_clips(shards):
    cl = json.load(open(os.path.join(shards, "clips.json")))
    return cl["clips"] if isinstance(cl, dict) else cl


def base_split(cs, source):
    """For one dataset: test = held-out performers, val = 1 in 20 takes of the others, pool = the rest; else unused."""
    out = []
    for c in cs:
        if c["source"] != source or int(c["length"]) < MIN_LEN or str(c["subject"]).endswith("unknown"):
            out.append("unused")
        elif zlib.crc32(str(c["subject"]).encode()) % 10 == 3:
            out.append("test")
        elif zlib.crc32(str(c["clip"]).encode()) % 20 == 0:
            out.append("val")
        else:
            out.append("pool")
    return out


def pool_hours(cs, base):
    per = defaultdict(int)
    for c, s in zip(cs, base):
        if s == "pool":
            per[c["subject"]] += int(c["length"])
    return sorted((v / FPS / 3600 for v in per.values()), reverse=True)


def draw(cs, base, P, hours, seed):
    """P pool performers, hours/P each (last take trimmed). Returns (split, length, people) or None."""
    rng = np.random.default_rng(1000 + seed)
    per = defaultdict(list)
    for i, (c, s) in enumerate(zip(cs, base)):
        if s == "pool":
            per[c["subject"]].append(i)
    share = int(round(hours * 3600 * FPS / P))
    eligible = sorted(p for p, ids in per.items() if sum(int(cs[i]["length"]) for i in ids) >= share)
    if len(eligible) < P:
        return None
    people = list(rng.choice(eligible, size=P, replace=False))
    split = ["unused" if s == "pool" else s for s in base]
    length = [int(c["length"]) for c in cs]
    for p in people:
        need = share
        for i in rng.permutation(per[p]):
            if need <= 0:
                break
            n = min(int(cs[i]["length"]), need)
            if n < MIN_LEN:
                continue
            split[i] = "train"; length[i] = n; need -= n
    if "train" not in split:
        return None
    return split, length, [str(p) for p in people]


# ----------------------------------------------------------------------------------------------- training
def train(name, split, length, meta, shards, size=SWEEP_SIZE, steps=None, extra=()):
    if glob.glob(os.path.join(OUT, f"phase2_{name}_{size}*.json")):
        SAY(f"skip {name} (done)"); return True
    if left() < 12 * 60:
        SAY(f"not starting {name}: {left()/60:.0f} min left"); return False
    path = os.path.join(SPLITS, name + ".json")
    json.dump(dict(name=name, split=split, length=length, meta=meta), open(path, "w"))
    cmd = [sys.executable, os.path.join(HERE, "train_torch.py"), "--size", size, "--steps", str(steps or STEPS),
           "--split-file", path, "--shards", shards, "--out", OUT, "--deadline", str(DEADLINE - 5 * 60), *extra]
    SAY(f"== {name} ({size}, {steps or STEPS} steps): " + json.dumps({k: v for k, v in meta.items() if k != 'people'}))
    t0 = time.time(); r = subprocess.run(cmd)
    SAY(f"   {name} exited {r.returncode} after {(time.time()-t0)/60:.1f} min")
    summarize()
    return r.returncode == 0


def summarize():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "phase2_*.json"))):
        try:
            r = json.load(open(p))
        except Exception:  # noqa: BLE001
            continue
        m = (r.get("split") or {}).get("meta") or {}
        rows.append(dict(name=(r.get("split") or {}).get("name"), size=r.get("size"), params=r.get("params"),
                         dataset=m.get("dataset"), experiment=m.get("experiment"), performers=m.get("performers"),
                         hours=m.get("hours"), seed=m.get("seed"), new_people=r["test"]["full"],
                         new_people_exo=r["test"].get("exo"), new_people_hidden_deg=r["test"].get("hidden_joint_rmse_deg"),
                         seen_people_new_takes=r["val"]["full"], objective=(r.get("objective") or {}).get("name", "next4"),
                         pi_pass=r.get("pi_pass_score"), train_hours=r.get("train_hours"), steps=r.get("steps"),
                         best_step=r.get("best_step"), wall_s=r.get("wall_s")))
    common.write_json(os.path.join(OUT, "summary.json"),
                      dict(updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), sweep_size=SWEEP_SIZE,
                           sweep_steps=STEPS, seed_status=seed_status(), rows=rows))
    return rows


def seed_status():
    for f, st in (("SEED_OK", "ok"), ("SEED_FAILED", "failed")):
        p = os.path.join(SHALL, f)
        if os.path.exists(p):
            return dict(status=st, detail=open(p).read()[:500])
    return dict(status="running" if os.path.exists(os.path.join(common.LOGS, "seed.log")) else "not started")


def wait_for_seed(max_wait_s):
    t0 = time.time()
    while time.time() - t0 < max_wait_s and seed_status()["status"] == "running" and left() > 45 * 60:
        time.sleep(60)
    return seed_status()["status"] == "ok"


# ----------------------------------------------------------------------------------------------- sweeps
def sweep_plan(source, cs, people_list, hours_P, label, seed):
    base = base_split(cs, source)
    hrs = pool_hours(cs, base)
    n_test = len({c["subject"] for c, s in zip(cs, base) if s == "test"})
    if n_test < 3 or len(hrs) < people_list[0] * 2:
        SAY(f"{label}: too few performers (pool {len(hrs)}, held out {n_test}); skipped"); return []
    P0 = people_list[0]
    H = 0.95 * (min(16.0, P0 * hrs[P0 - 1]) if source == "hiphi" else P0 * hrs[P0 - 1])
    H = float(f"{H:.2g}") if float(f"{H:.2g}") <= H else float(f"{H:.3g}") * 0.99
    if seed == 0:
        SAY(f"{label}: {len(hrs)} performers in the pool ({sum(hrs):.0f} h), {n_test} held out; H = {H:g} h")
    runs = [(f"{label}_P{P}_H{H:g}_s{seed}", P, H) for P in people_list if P <= len(hrs)]
    if seed == 0:
        runs += [(f"{label}_P{hours_P}_H{h:.3g}_s0", hours_P, float(f"{h:.3g}")) for h in (H / 4, H * 4)]
    return [(name, P, h, base) for name, P, h in runs]


def run_sweep(source, cs, shards, people_list, hours_P, label, seed):
    for name, P, h, base in sweep_plan(source, cs, people_list, hours_P, label, seed):
        d = draw(cs, base, P, h, seed)
        if d is None:
            SAY(f"skip {name}: fewer than {P} performers with {h/P:.2f} h"); continue
        split, length, people = d
        ok = train(name, split, length, dict(dataset=source, experiment=label, performers=P, hours=h, seed=seed,
                                             people=people), shards)
        if not ok and left() < 12 * 60:
            return False
    return True


def main():
    os.makedirs(SPLITS, exist_ok=True)
    seed_proc = None
    if os.environ.get("PINE_SKIP_SEED") != "1" and seed_status()["status"] != "ok":
        SAY("starting gpu/seed_data.py in the background (BONES-SEED download, retargeting, merge)")
        seed_proc = subprocess.Popen([sys.executable, os.path.join(HERE, "seed_data.py")],
                                     stdout=open(os.path.join(common.LOGS, "seed.log"), "a"), stderr=subprocess.STDOUT)
    cs1 = load_clips(SH1)
    SAY(f"phase 2: {left()/3600:.2f} h budget, sweep runs {SWEEP_SIZE} x {STEPS} steps")
    run_sweep("hiphi", cs1, SH1, [4, 8, 16, 32, 64], 32, "HiPHI", 0)
    if wait_for_seed(40 * 60):
        run_sweep("seed", load_clips(SHALL), SHALL, [8, 16, 32, 64, 128, 256, 400], 128, "SEED", 0)
    else:
        SAY(f"SEED not available for the sweep: {seed_status()}")
    for s in (1, 2):
        if left() < 100 * 60:
            break
        run_sweep("hiphi", cs1, SH1, [4, 8, 16, 32, 64], 32, "HiPHI", s)
        if seed_status()["status"] == "ok" and left() > 100 * 60:
            run_sweep("seed", load_clips(SHALL), SHALL, [8, 16, 32, 64, 128, 256, 400], 128, "SEED", s)
    # E3: everything except held-out performers
    use_all = seed_status()["status"] == "ok"
    shards = SHALL if use_all else SH1
    cs = load_clips(shards)
    bh, bs = base_split(cs, "hiphi"), base_split(cs, "seed")
    split = []
    for c, a, b in zip(cs, bh, bs):
        if a == "test":
            split.append("test")                       # scored on held-out HiPHI performers
        elif a == "val":
            split.append("val")
        elif a == "pool" or b == "pool" or (c["source"] not in ("hiphi", "seed") and c.get("split") == "train"):
            split.append("train" if int(c["length"]) >= MIN_LEN else "unused")
        else:
            split.append("unused")                     # held-out SEED actors and other sources' val/test stay out
    length = [int(c["length"]) for c in cs]
    frames = sum(l for s_, l in zip(split, length) if s_ == "train")
    meta = dict(dataset="all" if use_all else "phase1+hiphi", experiment="E3", hours=round(frames / FPS / 3600, 1), seed=0)
    steps_b = steps_for(SWEEP_SIZE, 8 * 60, STEPS, 30000)
    for name, extra in (("E3_recipe_next4", ("--save-weights",)), ("E3_recipe_dct16mix", ("--objective", "dct16mix", "--noisy-history"))):
        if left() < 45 * 60:
            SAY(f"skip {name}: under 45 min left"); break
        train(name, split, length, dict(meta, recipe=name.split("_")[-1]), shards, steps=steps_b, extra=extra)
    if left() > 25 * 60:
        size = os.environ.get("PINE_FLAGSHIP_SIZE", "5XL")
        t = tps_for(size)
        by_time = (left() - 12 * 60) * 0.8 * t / TOK_PER_STEP if t else 20000
        steps_f = int(max(4000, min(by_time, 30 * frames / TOK_PER_STEP)))
        SAY(f"flagship {size}: {steps_f} steps on {frames/FPS/3600:.0f} h, {left()/3600:.2f} h left, ~{t:,.0f} tok/s")
        train("E3_flagship", split, length, dict(meta, recipe="dct16mix", flagship=True), shards, size=size,
              steps=steps_f, extra=("--objective", "dct16mix", "--noisy-history", "--save-weights"))
    if seed_proc and seed_proc.poll() is None:
        SAY("stopping the SEED job (budget used)"); seed_proc.terminate()
    rows = summarize()
    SAY(f"phase 2 done: {len(rows)} runs; SEED {seed_status()['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
