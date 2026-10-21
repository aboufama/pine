"""Retarget every canonical take onto a humanoid and write robot-space shards.

    python scripts/retarget.py --robot h2 --out /home/claude/data/h2 --workers 2   # Unitree H2, full size
    python scripts/retarget.py --robot g1 --out /home/claude/data/g1 --workers 2   # Unitree G1
"""
from __future__ import annotations
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine.g1 import make  # noqa: E402



def run(args):
    part, idxs, canon_dir, out_dir, robot = args
    os.environ["OMP_NUM_THREADS"] = "1"
    g = make(robot)
    d = np.load(os.path.join(canon_dir, "keypoints.npz")); kp, off = d["kp"], d["offsets"]
    res = {"pr": [], "R": [], "q": [], "err": [], "idx": []}
    t0 = time.time(); batch, bidx, nb = [], [], 0
    def flush():
        for i, ((pr, R, q), e, _) in zip(bidx, g.retarget_many(batch)):
            res["pr"].append(pr.astype(np.float32)); res["R"].append(R.astype(np.float32))
            res["q"].append(q.astype(np.float32)); res["err"].append(e.astype(np.float32)); res["idx"].append(i)
    for i in idxs:
        batch.append(kp[off[i]:off[i + 1]]); bidx.append(i); nb += off[i + 1] - off[i]
        if nb > 12000:
            flush(); batch, bidx, nb = [], [], 0
            print(f"part {part}: {len(res['idx'])}/{len(idxs)} takes, {time.time()-t0:.0f}s", flush=True)
    if batch: flush()
    lens = np.array([len(x) for x in res["q"]])
    np.savez(os.path.join(out_dir, f"part{part}.npz"), pr=np.concatenate(res["pr"]), R=np.concatenate(res["R"]),
             q=np.concatenate(res["q"]), err=np.concatenate(res["err"]), idx=np.array(res["idx"]),
             offsets=np.concatenate([[0], np.cumsum(lens)]))
    return part, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--canon", default="/home/claude/data/canon"); ap.add_argument("--out", default="/home/claude/data/g1")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--robot", default="h2")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    man = json.load(open(os.path.join(a.canon, "manifest.json")))
    idxs = [c["index"] for c in man["clips"] if "reject" not in c]
    parts = [(p, idxs[p::a.workers], a.canon, a.out, a.robot) for p in range(a.workers)]
    t0 = time.time()
    with Pool(a.workers) as pool:
        for p, dt in pool.imap_unordered(run, parts):
            print(f"part {p} done in {dt:.0f}s", flush=True)
    wall = time.time() - t0
    frames = int(sum(json.load(open(os.path.join(a.canon, "manifest.json")))["summary"]["frames"] for _ in [0]))
    json.dump(dict(robot=a.robot, wall_s=wall, workers=a.workers, ms_per_frame=wall * a.workers / frames * 1e3),
              open(os.path.join(a.out, "timing.json"), "w"))
    print(f"retarget wall {wall:.0f}s")
