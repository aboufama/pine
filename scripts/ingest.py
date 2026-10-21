"""Ingest raw public mocap into canonical keypoint shards.

    python scripts/ingest.py --raw /home/claude/data/raw --out /home/claude/data/canon

Sources handled here (the ones reachable from this workspace):
  cmu      CMU Graphics Lab, BVH conversion by B. Hahne      2,548 takes, 120 fps, free for any use
  bandai1  Bandai Namco Research Motion Dataset 1            175 takes, 30 fps, CC BY-NC 4.0
  bandai2  Bandai Namco Research Motion Dataset 2            2,902 takes, 30 fps, CC BY-NC 4.0
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time, zlib
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pine.mocap_io import read_bvh          # noqa: E402
from pine import canon                       # noqa: E402

CMU_UNIT = 0.0254 / 0.45   # Acclaim units: length 0.45 -> 1/0.45 inch


def cmu_descriptions(root: str) -> dict:
    d = {}
    p = os.path.join(root, "cmu-mocap/repo/cmu-mocap-index-text.txt")
    for line in open(p, encoding="utf-8", errors="replace"):
        m = re.match(r"^(\d+_\d+)\s+(.+)$", line.strip())
        if m:
            d[m.group(1)] = m.group(2).strip()
    return d


def jobs(raw: str) -> list[tuple]:
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "cmu-mocap/repo/data/*/*.bvh"))):
        clip = os.path.basename(f)[:-4]
        out.append(("cmu", f, clip, "cmu" + clip.split("_")[0], CMU_UNIT, "mixamo", 1))
    bd = os.path.join(raw, "Bandai-Namco-Research-Motiondataset/repo/dataset")
    for k in (1, 2):
        for f in sorted(glob.glob(os.path.join(bd, f"Bandai-Namco-Research-Motiondataset-{k}/data/*.bvh"))):
            clip = os.path.basename(f)[:-4]
            out.append((f"bandai{k}", f, clip, f"bandai{k}", 0.01, "bandai", 0))
    return out


def work(job):
    source, path, clip, subject, unit, skel, drop = job
    try:
        take = read_bvh(path, unit, source, clip, subject)
        kp = canon.to_canonical(take, skel)[drop:]
        kp = canon.resample(kp, take.fps)
        kp, rep = canon.qa_and_normalise(kp)
    except Exception as e:  # a reader failure is a rejection, never a crash
        return dict(source=source, clip=clip, subject=subject, reject=f"read error: {e}"), None
    rep.update(source=source, clip=clip, subject=subject, fps_in=take.fps)
    return rep, kp


def split_of(source: str, clip: str, subject: str) -> str:
    """Held-out people where the dataset has them (CMU), held-out takes otherwise."""
    if source == "cmu":
        return "test" if int(subject[3:]) % 10 == 7 else ("val" if zlib.crc32(clip.encode()) % 20 == 0 else "train")
    h = zlib.crc32(clip.encode()) % 20
    return "test" if h == 7 else ("val" if h == 0 else "train")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/claude/data/raw")
    ap.add_argument("--out", default="/home/claude/data/canon")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    js = jobs(a.raw)
    desc = cmu_descriptions(a.raw)
    t0 = time.time()
    clips, arrays = [], []
    with Pool(a.workers) as pool:
        for i, (rep, kp) in enumerate(pool.imap(work, js, chunksize=8)):
            rep["split"] = split_of(rep["source"], rep["clip"], rep["subject"])
            if rep["source"] == "cmu":
                rep["desc"] = desc.get(rep["clip"], "")
            else:
                parts = rep["clip"].split("_")
                rep["desc"] = f"{parts[1].replace('-', ' ')}, {parts[2]}" if len(parts) >= 4 else rep["clip"]
            if kp is not None:
                rep["index"] = len(arrays); arrays.append(kp)
            clips.append(rep)
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(js)} takes  {time.time()-t0:.0f}s", flush=True)
    lengths = np.array([len(x) for x in arrays])
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    np.savez(os.path.join(a.out, "keypoints.npz"), kp=np.concatenate(arrays).astype(np.float32), offsets=offsets)
    kept = [c for c in clips if "reject" not in c]
    summary = dict(
        fps=canon.FPS, keypoints=canon.KEYPOINTS, edges=canon.EDGES,
        takes_in=len(clips), takes_kept=len(kept), frames=int(offsets[-1]),
        hours=float(offsets[-1] / canon.FPS / 3600), seconds_wall=time.time() - t0,
        rejects={}, per_source={})
    for c in clips:
        if "reject" in c:
            key = c["reject"].split(" (")[0].split(":")[0]
            summary["rejects"][key] = summary["rejects"].get(key, 0) + 1
    for s in sorted({c["source"] for c in clips}):
        k = [c for c in kept if c["source"] == s]
        summary["per_source"][s] = dict(takes=len(k), frames=int(sum(c["frames"] for c in k)),
                                        hours=sum(c["frames"] for c in k) / canon.FPS / 3600,
                                        subjects=len({c["subject"] for c in k}))
    json.dump(dict(summary=summary, clips=clips), open(os.path.join(a.out, "manifest.json"), "w"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
