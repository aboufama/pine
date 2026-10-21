"""Phase 2 on the pod: does a new person's motion get predicted better by more people or by more hours?

Runs after phase 1 (gpu/sweep_gpu.py) on the same shards, inside the GPU-hour budget phase 1 left. HiPHI has 132
performers, so whole people can be held out:

  split      HiPHI performers with crc32(subject) % 10 == 3 are TEST (never trained on); 1 in 20 takes of the other
             performers is VAL (new takes, seen people); everything else is the training pool. Other sources are unused.
  E1 people  H hours of training data split evenly across P performers, P = 4, 8, 16, 32, 64 (2 draws for P <= 16)
  E2 hours   32 performers, H/4 and 4H hours (with E1's 32-performer run that is three points)
  E3 recipe  all of the training pool, the 2XL size: the CPU recipe vs dct16mix + noisy history (phase-1 extra run's
             recipe), on this held-out-people split

E1 and E2 runs are all the same size (2XL) and the same number of steps (about 2.5 min each at phase 1's measured
throughput), scored on the held-out performers. H is the largest of 16, 8, 4 hours for which at least 8 performers
each have H/4 hours. E3 splits whatever budget is left between its two runs. Results: $PINE_RUNS/phase2/*.json and summary.json.
"""
from __future__ import annotations
import glob, json, os, subprocess, sys, time, zlib
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402

SAY = lambda m: common.log(m, "phase2")  # noqa: E731
OUT = os.path.join(common.RUNS, "phase2"); SPLITS = os.path.join(OUT, "splits")
SIZE, FPS, MIN_LEN, TOK_PER_STEP = "2XL", 20.0, 36, 512 * 32
DEADLINE = float(os.environ.get("PINE_PHASE2_DEADLINE", time.time() + 4 * 3600))


def throughput() -> float:
    """Training tokens per second of size SIZE, measured by phase 1 (0 if unknown)."""
    for p in glob.glob(os.path.join(common.RUNS, f"*_{SIZE}_*.json")):
        try:
            tps = float((json.load(open(p)).get("device") or {}).get("tokens_per_s") or 0)
            if tps > 0:
                return tps
        except Exception:  # noqa: BLE001
            pass
    return 0.0


def steps_for(seconds: float, lo: int, hi: int) -> int:
    tps = throughput()
    return int(min(hi, max(lo, seconds * tps / TOK_PER_STEP))) if tps else lo


STEPS = int(os.environ.get("PINE_PHASE2_STEPS", 0)) or steps_for(150, 3000, 20000)   # about 2.5 min of training per run


def clips():
    cl = json.load(open(os.path.join(common.SHARDS, "clips.json")))
    return cl["clips"] if isinstance(cl, dict) else cl


def base_split(cs):
    """test = held-out HiPHI performers, val = 1 in 20 takes of the others, pool = the rest; non-HiPHI unused."""
    out = []
    for c in cs:
        if c["source"] != "hiphi" or int(c["length"]) < MIN_LEN:
            out.append("unused")
        elif zlib.crc32(str(c["subject"]).encode()) % 10 == 3:
            out.append("test")
        elif zlib.crc32(str(c["clip"]).encode()) % 20 == 0:
            out.append("val")
        else:
            out.append("pool")
    return out


def draw(cs, base, P, hours, seed):
    """P pool performers, hours/P each (last take trimmed). Returns (split, length) lists or None if too few people."""
    rng = np.random.default_rng(1000 + seed)
    per = defaultdict(list)
    for i, (c, s) in enumerate(zip(cs, base)):
        if s == "pool":
            per[c["subject"]].append(i)
    share = int(round(hours * 3600 * FPS / P))
    have = {p: sum(int(cs[i]["length"]) for i in ids) for p, ids in per.items()}
    eligible = sorted(p for p in per if have[p] >= share)
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
    return split, length, [str(p) for p in people]


def pick_hours(cs, base):
    per = defaultdict(int)
    for c, s in zip(cs, base):
        if s == "pool":
            per[c["subject"]] += int(c["length"])
    hrs = sorted((v / FPS / 3600 for v in per.values()), reverse=True)
    for H in (16.0, 8.0, 4.0, 2.0):
        if sum(h >= H / 4 for h in hrs) >= 8:
            return H, hrs
    return 1.0, hrs


def train(name, split, length, meta, extra=(), steps=None):
    rec_glob = os.path.join(OUT, f"phase2_{name}_{SIZE}*.json")
    if glob.glob(rec_glob):
        SAY(f"skip {name} (done)"); return True
    path = os.path.join(SPLITS, name + ".json")
    json.dump(dict(name=name, split=split, length=length, meta=meta), open(path, "w"))
    left = DEADLINE - time.time()
    if left < 15 * 60:
        SAY(f"not starting {name}: {left/60:.0f} min left in the budget"); return False
    cmd = [sys.executable, os.path.join(HERE, "train_torch.py"), "--size", SIZE, "--steps", str(steps or STEPS), "--split-file", path,
           "--out", OUT, "--deadline", str(DEADLINE - 5 * 60), *extra]
    SAY(f"== {name}: {meta}")
    t0 = time.time()
    r = subprocess.run(cmd)
    SAY(f"   {name} exited {r.returncode} after {(time.time()-t0)/60:.1f} min")
    return r.returncode == 0


def summarize():
    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "phase2_*.json"))):
        r = json.load(open(p))
        m = (r.get("split") or {}).get("meta") or {}
        rows.append(dict(name=(r.get("split") or {}).get("name"), size=r.get("size"), params=r.get("params"),
                         experiment=m.get("experiment"), performers=m.get("performers"), hours=m.get("hours"), seed=m.get("seed"),
                         new_people=r["test"]["full"], new_people_exo=r["test"].get("exo"),
                         new_people_hidden_deg=r["test"].get("hidden_joint_rmse_deg"), seen_people_new_takes=r["val"]["full"],
                         objective=(r.get("objective") or {}).get("name", "next4"), pi_pass=r.get("pi_pass_score"),
                         train_hours=r.get("train_hours"), best_step=r.get("best_step"), wall_s=r.get("wall_s")))
    json.dump(dict(updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), size=SIZE, steps=STEPS, rows=rows),
              open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    return rows


def main():
    os.makedirs(SPLITS, exist_ok=True)
    cs = clips(); base = base_split(cs)
    n_test = len({c["subject"] for c, s in zip(cs, base) if s == "test"})
    n_pool = len({c["subject"] for c, s in zip(cs, base) if s in ("pool", "val")})
    if n_test < 3 or n_pool < 16:
        SAY(f"too few HiPHI performers for phase 2 (test {n_test}, pool {n_pool}); nothing to do"); summarize(); return 0
    H, hrs = pick_hours(cs, base)
    SAY(f"HiPHI: {n_pool} training performers ({sum(hrs):.0f} h), {n_test} held out; people sweep at {H:g} h; "
        f"{STEPS} steps of size {SIZE} per run; deadline in {(DEADLINE-time.time())/3600:.2f} h")
    plan = [("E1", P, H, 0) for P in (4, 8, 16, 32, 64)] + [("E2", 32, h, 0) for h in (H / 4, H * 4)] \
         + [("E1", P, H, 1) for P in (4, 8, 16, 32, 64)] + [("E1", P, H, 2) for P in (4, 8, 16)]
    for exp, P, hours, seed in plan:
        d = draw(cs, base, P, hours, seed)
        if d is None:
            SAY(f"skip {exp} P={P} H={hours:g}: fewer than {P} performers with {hours/P:.2f} h"); continue
        split, length, people = d
        name = f"{exp}_P{P}_H{hours:g}_s{seed}"
        ok = train(name, split, length, dict(experiment=exp, performers=P, hours=hours, seed=seed, people=people))
        summarize()
        if not ok and DEADLINE - time.time() < 15 * 60:
            break
    # E3: the whole pool, the CPU recipe vs the phase-1 extra recipe, on held-out people
    split = ["train" if s == "pool" else s for s in base]; length = [int(c["length"]) for c in cs]
    meta = dict(experiment="E3", performers=n_pool, hours=round(sum(hrs), 1), seed=0)
    left = DEADLINE - time.time() - 10 * 60
    steps3 = steps_for(left / 2.6, STEPS, 30000)        # two runs share what is left, with room for scoring
    for name, extra in (("E3_all_next4", ("--save-weights",)),       # its weights let the dct16mix run score it too
                        ("E3_all_dct16mix", ("--objective", "dct16mix", "--noisy-history"))):
        if DEADLINE - time.time() < 30 * 60:
            SAY(f"skip {name}: under 30 min left"); break
        train(name, split, length, dict(meta, recipe=name.split("_")[-1], steps=steps3), extra, steps=steps3)
        summarize()
    rows = summarize()
    SAY(f"phase 2 done: {len(rows)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
