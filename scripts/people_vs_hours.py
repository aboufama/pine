"""More people or more hours? A controlled test on CMU, the source with 102 identified people in training.

Every run trains the same model (size S) for the same compute (900 steps of 64 windows) and is scored on the 11 CMU
people held out from every run, so the only thing that changes is what the training hour is made of:

  people sweep   1 hour of training data split evenly across k people, k = 4, 8, 16, 32, 62 (2 draws of people each)
  hours sweep    32 people, 0.25, 1 and 2 hours

Each person contributes exactly the same share (the last take is trimmed), so every run in a sweep has the same
hours. For small k only people with at least 1/k hour qualify; 62 is the most people who each have 1/62 hour.
Results: runs/people/*.json and runs/people/summary.json.
"""
from __future__ import annotations
import copy, json, os, sys, time
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine.features import MotionData   # noqa: E402
from pine import train_np               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "runs", "people"); os.makedirs(OUT, exist_ok=True)
FPS = 20.0


def choose(d, k, hours, seed):
    """Pick k training people and exactly hours/k of motion from each (the last take is trimmed to fit).

    Returns (clip indices, people, {clip index: frames to keep}) or (None, None, None)."""
    rng = np.random.default_rng(seed)
    per = defaultdict(list)
    for i, c in enumerate(d.clips):
        if c["source"] == "cmu" and c["split"] == "train":
            per[c["subject"]].append(i)
    share = int(round(hours * 3600 * FPS / k))
    have = {s: int(sum(d.lens[i] for i in ids)) for s, ids in per.items()}
    eligible = sorted(s for s in per if have[s] >= share)
    if len(eligible) < k:
        return None, None, None
    people = list(rng.choice(eligible, size=k, replace=False))
    keep = {}
    for s in people:
        need = share
        for i in rng.permutation(per[s]):
            if need <= 0:
                break
            n = int(min(d.lens[i], need))
            if n < 36:            # shorter than one training window: skip it
                continue
            keep[int(i)] = n; need -= n
    return sorted(keep), people, keep


def view(d, train_ids, keep):
    v = copy.copy(d)
    v.lens = d.lens.copy()
    for i, n in keep.items():
        v.lens[i] = n
    split = np.array(["unused"] * len(d.clips), dtype=object)
    for i, c in enumerate(d.clips):
        if c["source"] == "cmu" and c["split"] in ("val", "test"):
            split[i] = c["split"]
    split[train_ids] = "train"
    v.split = split
    return v


def main():
    d = MotionData("/home/claude/data/canon", f"/home/claude/data/{os.environ.get('PINE_ROBOT', 'h2')}")
    plan = [("people", k, 1.0, s) for k in (4, 8, 16, 32, 62) for s in (0, 1)] + [("hours", 32, h, 0) for h in (0.25, 2.0)]
    for kind, k, hours, seed in plan:
        name = f"{kind}_k{k}_h{hours:g}_s{seed}"
        path = os.path.join(OUT, name + ".json")
        if os.path.exists(path):
            continue
        ids, people, keep = choose(d, k, hours, seed)
        if ids is None:
            print(f"skip {name}: not enough people with data", flush=True); continue
        v = view(d, ids, keep)
        got = float(v.lens[ids].sum() / FPS / 3600)
        print(f"== {name}: {len(people)} people, {len(ids)} takes, {got:.2f} h", flush=True)
        t0 = time.time()
        rec, _ = train_np.run(v, size="S", fraction=1.0, steps=900, seed=seed, log=lambda s: None)
        out = dict(name=name, kind=kind, people=k, hours=got, seed=seed, takes=len(ids), subjects=[str(p) for p in people],
                   new_people=rec["test"]["full"], new_people_hidden_deg=rec["test"]["hidden_joint_rmse_deg"],
                   new_people_exo=rec["test"]["exo"], seen_people_val=rec["val"]["full"], best_step=rec["best_step"],
                   test_per_window=rec["test"]["per_window"], wall_s=time.time() - t0)
        json.dump(out, open(path, "w"))
        print(f"   new people {out['new_people']:.4f}  hidden joints {out['new_people_hidden_deg']:.1f} deg  "
              f"(seen-people val {out['seen_people_val']:.4f})  {out['wall_s']:.0f}s", flush=True)
    rows = [json.load(open(os.path.join(OUT, f))) for f in sorted(os.listdir(OUT)) if f.endswith(".json") and f != "summary.json"]
    json.dump([{k: v for k, v in r.items() if k != "test_per_window"} for r in rows], open(os.path.join(OUT, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
