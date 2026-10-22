"""Turn everything fetch_data.py downloaded into the robot feature space of pine/features.py, as training shards.

    python gpu/build_shards.py [--workers N] [--device cuda|cpu] [--only cmu,bandai,lafan1,hiphi,lafan1_g1,amass_g1]

Output ($PINE_SHARDS, default /workspace/data/shards):
  features.npy   (frames, D) float16, normalised with the TRAIN-split mean/std (pine.features.MotionData's rule)
  clips.json     [{index, source, clip, subject, split, start, length, fps_in, desc, ...}] in feature-row order
  stats.json     robot, names, D, mean, std (train only), hours by split and source, rejects, IK check, timings

Pipeline, per source:
  BVH (CMU, Bandai Namco, LAFAN1, HiPHI) -> pine.mocap_io.read_bvh -> pine.canon.to_canonical('mixamo' map + ALIASES;
  Bandai uses its own map, as in scripts/ingest.py) -> resample to 20 fps -> qa_and_normalise (same rejects) ->
  IK on the GPU (gpu/retarget_torch.py, same maths as pine.g1.Humanoid.retarget_many, checked against it on 2,000
  frames: mean keypoint difference must be < 1 mm, else the NumPy solver is used on all cores) -> g1_features.
  Units: CMU 0.0254/0.45 and Bandai 0.01 as in scripts/ingest.py; LAFAN1 and HiPHI are centimetres, confirmed per take
  from the leg length (a take whose scale is off by 10x/100x is re-read with the scale the leg length implies).
  G1-native sources (only when PINE_ROBOT=g1): Unitree LAFAN1 csv (30 fps) and Retargeted AMASS npy (fps from the file
  name) skip IK: quaternion xyzw -> rotation matrix, resample to 20 fps, wrist joints set to 0 (the CPU pipeline solves
  only the 23 ACTIVE joints and holds the wrists at 0), then g1_features. Recorded in stats.json.
Split rule: scripts/ingest.split_of (CMU: subject % 10 == 7 is test; others crc32(clip) % 20: 7 test, 0 val).
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys, time, zlib
from multiprocessing import get_context

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402  (PINE_* defaults; repo root on sys.path)
from common import write_json  # noqa: E402
import numpy as np  # noqa: E402

from pine.mocap_io import read_bvh  # noqa: E402
from pine import canon  # noqa: E402
from pine import features as F  # noqa: E402
from pine.g1 import SPECS, make, quat_to_mat  # noqa: E402
from scripts.ingest import split_of as cpu_split_of, CMU_UNIT  # noqa: E402

SAY = lambda m: common.log(m, "build")  # noqa: E731
RAW, WORK, SHARDS = common.RAW, common.WORK, common.SHARDS
ROBOT = F.ROBOT
HELD_NOTE = ("G1-native sources: q has 29 joints; only ACTIVE (23) enter the features and the wrist joints are set "
             "to 0 first, matching the CPU pipeline, which solves 23 joints and holds the wrists at 0")


# --------------------------------------------------------------------------------------------- split / names
def split_of(source: str, clip: str, subject: str) -> str:
    """The CPU rule (scripts/ingest.split_of), for every source. Non-CMU clips need a source-unique clip id."""
    return cpu_split_of(source, clip, subject)


def cmu_descriptions() -> dict:
    p = os.path.join(RAW, "cmu", "repo", "cmu-mocap-index-text.txt")
    d = {}
    try:
        for line in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"^(\d+_\d+)\s+(.+)$", line.strip())
            if m:
                d[m.group(1)] = m.group(2).strip()
    except OSError:
        pass
    return d


# --------------------------------------------------------------------------------------------- job lists
def bvh_jobs(only: set) -> list[dict]:
    """(source, path, clip, subject, unit, skel, drop) exactly like scripts/ingest.jobs, plus LAFAN1 and HiPHI."""
    out = []
    if "cmu" in only:
        for f in sorted(glob.glob(os.path.join(RAW, "cmu", "repo", "data", "*", "*.bvh"))):
            clip = os.path.basename(f)[:-4]
            out.append(dict(source="cmu", path=f, clip=clip, subject="cmu" + clip.split("_")[0], unit=CMU_UNIT,
                            skel="mixamo", drop=1, check_unit=False))
    if "bandai" in only:
        bd = os.path.join(RAW, "bandai", "repo", "dataset")
        for k in (1, 2):
            for f in sorted(glob.glob(os.path.join(bd, f"Bandai-Namco-Research-Motiondataset-{k}", "data", "*.bvh"))):
                clip = os.path.basename(f)[:-4]
                out.append(dict(source=f"bandai{k}", path=f, clip=clip, subject=f"bandai{k}", unit=0.01, skel="bandai",
                                drop=0, check_unit=False))
    if "lafan1" in only:
        for f in sorted(glob.glob(os.path.join(RAW, "lafan1", "repo", "lafan1", "bvh", "*.bvh"))):
            clip = os.path.basename(f)[:-4]                       # [theme][take]_[subject].bvh
            subj = clip.rsplit("_", 1)[-1] if "_" in clip else "lafan1"
            out.append(dict(source="lafan1", path=f, clip=clip, subject="lafan1_" + subj, unit=0.01, skel="mixamo",
                            drop=0, check_unit=True))
    if "hiphi" in only:
        meta = hiphi_actors()
        root = os.path.join(RAW, "hiphi", "extracted")
        for f in sorted(glob.glob(os.path.join(root, "**", "*.bvh"), recursive=True)):
            mid = os.path.basename(f)[:-4]
            if "__mirror" in f:          # a left-right mirror is not a new motion (and would leak across splits)
                continue
            if mid not in meta:          # packages may name the BVH generically; the motion id is then the folder
                mid = os.path.basename(os.path.dirname(f)) if os.path.basename(os.path.dirname(f)) in meta else mid
            actor, text = meta.get(mid, ("unknown", ""))
            out.append(dict(source="hiphi", path=f, clip=mid, subject="hiphi_" + actor, unit=0.01, skel="mixamo",
                            drop=0, check_unit=True, desc=text))
    return out


def hiphi_actors() -> dict:
    """motion_id -> (actor_id, text_annotation) from metadata/hiphi_metadata.csv (columns per the dataset README)."""
    p = os.path.join(RAW, "hiphi", "metadata", "hiphi_metadata.csv")
    out = {}
    try:
        import csv
        with open(p, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                mid = (row.get("motion_id") or "").strip()
                if mid:
                    out[mid] = ((row.get("actor_id") or "unknown").strip() or "unknown",
                                (row.get("text_annotation") or "").strip()[:200])
    except OSError:
        pass
    return out


# --------------------------------------------------------------------------------------------- BVH worker
def leg_length(kp: np.ndarray) -> float:
    KP = canon.KP
    thigh = np.linalg.norm(kp[:, KP["l_hip"]] - kp[:, KP["l_knee"]], axis=1)
    shin = np.linalg.norm(kp[:, KP["l_knee"]] - kp[:, KP["l_ankle"]], axis=1)
    return float(np.median(thigh + shin))


FALLBACK = {   # used only for a keypoint pine.canon cannot find (e.g. a skeleton whose feet end in End Sites)
    "l_toe": ["LeftToeBase", "LeftToe", "LeftFoot_end", "LeftToe_end", "LeftToeBase_end"],
    "r_toe": ["RightToeBase", "RightToe", "RightFoot_end", "RightToe_end", "RightToeBase_end"],
    "chest": ["Spine3", "Spine2", "Spine1", "Chest", "Spine"],
    "neck": ["Neck", "Neck1"], "head": ["Head", "Neck2", "Neck1"],
}


def to_canonical(take, skel: str):
    """pine.canon.to_canonical; if it cannot find a keypoint, a lenient resolver (fallback names, then a
    case-insensitive suffix match). Returns (keypoints, None) or (keypoints, description of what was substituted)."""
    try:
        return canon.to_canonical(take, skel), None
    except KeyError:
        pass
    names = [n.split(":")[-1] for n in take.names]
    idx = {n: i for i, n in enumerate(names)}
    low = {}
    for i, n in enumerate(names):
        low.setdefault(n.lower(), i)
    out, subs = [], []
    for k in canon.KEYPOINTS:
        name = canon.MAPS[skel][k]
        cands = [name] + canon.ALIASES.get(name, []) + FALLBACK.get(k, [])
        j = next((idx[c] for c in cands if c in idx), None)
        if j is None:
            j = next((low[c.lower()] for c in cands if c.lower() in low), None)
        if j is None:
            j = next((i for i, n in enumerate(names) if any(n.lower().endswith(c.lower()) for c in cands)), None)
        if j is None:
            raise KeyError(f"{take.source}/{take.clip}: no joint for {k} ({name}); joints: {names[:12]}...")
        if names[j] != name:
            subs.append(f"{k}={names[j]}")
        out.append(take.pos[:, j])
    return np.stack(out, 1).astype(np.float64), ",".join(subs)


def bvh_work(job: dict):
    """One take -> (report, canonical keypoints at 20 fps or None). A failure is a rejection, never a crash."""
    rep = dict(source=job["source"], clip=job["clip"], subject=job["subject"])
    try:
        take = read_bvh(job["path"], job["unit"], job["source"], job["clip"], job["subject"])
        kp, how = to_canonical(take, job["skel"])
        kp = kp[job["drop"]:]
        if how:
            rep["map_fallback"] = how
        if job.get("check_unit") and len(kp) > 2:
            leg = leg_length(kp)
            if leg > 0 and not 0.55 < leg < 1.15:                 # e.g. a take stored in mm or m instead of cm
                for alt in (0.001, 1.0, 0.0254, 0.1):
                    if 0.55 < leg * alt / job["unit"] < 1.15:
                        kp = kp * (alt / job["unit"]); rep["unit_fixed"] = alt; break
        kp = canon.resample(kp, take.fps)
        kp, q = canon.qa_and_normalise(kp)
        rep.update(q)
        rep["fps_in"] = take.fps
    except Exception as e:  # noqa: BLE001
        rep["reject"] = f"read error: {type(e).__name__}: {e}"[:300]
        return rep, None
    return rep, kp


# --------------------------------------------------------------------------------------------- G1-native parsers
def read_g1_csv(path: str) -> np.ndarray:
    """Unitree LAFAN1_Retargeting_Dataset csv -> (N, 36): x y z qx qy qz qw + 29 joints (pine.g1.JOINTS order).
    Accepts an optional header row and an optional leading frame-index column."""
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = [p for p in re.split(r"[,\s]+", line.strip()) if p]
            if not parts:
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                if rows:
                    raise ValueError(f"{path}: non-numeric row after data")
                continue                                           # header
    if not rows:
        raise ValueError(f"{path}: no numeric rows")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: ragged rows {sorted(widths)}")
    arr = np.asarray(rows, np.float64)
    if arr.shape[1] == 37 and np.allclose(np.diff(arr[:, 0]), 1):   # leading frame index
        arr = arr[:, 1:]
    if arr.shape[1] != 36:
        raise ValueError(f"{path}: {arr.shape[1]} columns, expected 36 (7 root + 29 joints)")
    return arr


def amass_fps(path: str) -> tuple[float, str]:
    """The dataset's own visualize.py reads the rate from the name: '<motion>_poses_<fps>_jpos.npy'."""
    m = re.search(r"_poses_(\d+(?:\.\d+)?)_jpos\.npy$", os.path.basename(path))
    if m and 1.0 <= float(m.group(1)) <= 1000.0:
        return float(m.group(1)), "file name"
    return 30.0, "ASSUMED (no rate in the file name)"


def quat_xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return np.stack([quat_to_mat([w, x, y, z]) for x, y, z, w in q])


def resample_series(x: np.ndarray, fps_in: float, fps_out: float = canon.FPS) -> np.ndarray:
    """canon.resample's linear interpolation for (N, C) arrays."""
    return canon.resample(x[:, :, None], fps_in, fps_out)[:, :, 0]


def native_to_features(arr: np.ndarray, fps: float, nq: int, wrist_idx: list[int]):
    """(N, 7+nq) root xyz, quat xyzw, joints -> features at 20 fps. Quaternions are resampled by normalised lerp."""
    if arr.shape[0] < 2:
        raise ValueError("fewer than 2 frames")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite values")
    quat = arr[:, 3:7].copy()
    flip = np.concatenate([[1.0], np.sign(np.einsum("ni,ni->n", quat[1:], quat[:-1]) + 1e-12)])
    quat *= np.cumprod(flip)[:, None]                       # keep consecutive quaternions in one hemisphere
    rs = resample_series(np.concatenate([arr[:, :3], quat, arr[:, 7:7 + nq]], 1), fps)
    pr, R = rs[:, :3], quat_xyzw_to_mat(rs[:, 3:7])
    q = rs[:, 7:7 + nq].copy()
    q[:, wrist_idx] = 0.0
    return F.g1_features(pr, R, q), pr, R, q


def native_jobs(only: set) -> list[dict]:
    out = []
    if ROBOT != "g1":
        return out
    if "lafan1_g1" in only:
        for f in sorted(glob.glob(os.path.join(RAW, "lafan1_g1", "g1", "*.csv"))):
            clip = os.path.basename(f)[:-4]
            subj = clip.rsplit("_", 1)[-1] if "_" in clip else "x"
            out.append(dict(source="lafan1_g1", path=f, clip="u_" + clip, subject="lafan1_" + subj, fps=30.0,
                            fps_from="dataset card (30 FPS)", kind="csv"))
    if "amass_g1" in only:
        base = os.path.join(RAW, "amass_g1", "g1")
        have_cmu = bool(glob.glob(os.path.join(RAW, "cmu", "repo", "data", "*", "*.bvh")))
        for f in sorted(glob.glob(os.path.join(base, "**", "*.npy"), recursive=True)):
            rel = os.path.relpath(f, base)
            sub = rel.split(os.sep)[0]
            if sub == "CMU" and have_cmu:                    # same motions as the CMU BVH: do not count them twice
                continue
            fps, how = amass_fps(f)
            parts = rel.split(os.sep)
            subj = "amass_" + "/".join(parts[:2]) if len(parts) > 2 else "amass_" + sub
            out.append(dict(source="amass_g1", path=f, clip="a_" + rel.replace(os.sep, "/")[:-4], subject=subj,
                            fps=fps, fps_from=how, kind="npy"))
    return out


def native_work(job: dict):
    rep = dict(source=job["source"], clip=job["clip"], subject=job["subject"], fps_in=job["fps"], fps_from=job["fps_from"])
    try:
        arr = read_g1_csv(job["path"]) if job["kind"] == "csv" else np.load(job["path"]).astype(np.float64)
        if arr.ndim != 2 or arr.shape[1] != 36:
            raise ValueError(f"shape {arr.shape}, expected (N, 36)")
        robot_joints = SPECS[ROBOT]["joints"]
        wrists = [i for i, j in enumerate(robot_joints) if "wrist" in j]
        feats, pr, R, q = native_to_features(arr, job["fps"], len(robot_joints), wrists)
        n = len(feats)
        rep["frames"] = int(n)
        if n < 2 * canon.FPS:
            rep["reject"] = "shorter than 2 s"; return rep, None
        rep["max_speed"] = float(np.linalg.norm(np.diff(pr, axis=0), axis=1).max() * canon.FPS)
        if rep["max_speed"] > 20.0:
            rep["reject"] = f"teleport ({rep['max_speed']:.0f} m/s)"; return rep, None
        rep["pelvis_h"] = float(np.median(pr[:, 2]))
        rep["quat_norm_dev"] = float(np.abs(np.linalg.norm(arr[:, 3:7], axis=1) - 1).max())
        if rep["quat_norm_dev"] > 0.05:
            rep["reject"] = "root quaternion is not unit length (column order?)"; return rep, None
        return rep, feats
    except Exception as e:  # noqa: BLE001
        rep["reject"] = f"read error: {type(e).__name__}: {e}"[:300]
        return rep, None


# --------------------------------------------------------------------------------------------- IK
class Retargeter:
    """GPU IK when it passes the check against pine/g1.py, otherwise NumPy (pine.g1) on a process pool."""

    def __init__(self, device: str, workers: int):
        self.robot = make(ROBOT)
        self.workers = workers
        self.ik, self.check, self.backend = None, None, "numpy"
        self.device = device

    def calibrate(self, sample_takes: list[np.ndarray]) -> dict:
        import retarget_torch as RT
        if self.device == "cpu":
            self.check = dict(passed=False, reason="--device cpu"); return self.check
        try:
            import torch
            if not torch.cuda.is_available():
                self.check = dict(passed=False, reason="no CUDA device"); return self.check
            ik = RT.BatchedIK(self.robot, RT.make_ops("torch", "cuda"))
            res = RT.check_against_numpy(self.robot, sample_takes, ik, frames=2000, limit_mm=1.0)
        except Exception as e:  # noqa: BLE001
            res = dict(passed=False, reason=f"{type(e).__name__}: {e}"[:300])
        self.check = res
        if res.get("passed"):
            self.ik, self.backend = ik, "torch"
        return res

    def run(self, takes: list[np.ndarray], chunk_frames: int = 150_000):
        """takes: canonical keypoints -> list of (pr, R, q, err) per take."""
        import retarget_torch as RT
        out = [None] * len(takes)
        groups = RT.chunks_by_frames([len(t) for t in takes], chunk_frames if self.backend == "torch" else 12_000)
        if self.backend == "torch":
            for gi, g in enumerate(groups):
                res = self.ik.retarget_many([takes[i] for i in g])
                for i, ((pr, R, q), e, _) in zip(g, res):
                    out[i] = (pr, R, q, e)
                if (gi + 1) % 10 == 0 or gi + 1 == len(groups):
                    SAY(f"  IK (gpu) {gi + 1}/{len(groups)} chunks")
            return out
        ctx = get_context("fork")
        with ctx.Pool(self.workers, initializer=_np_ik_init, initargs=(ROBOT,)) as pool:
            for gi, (g, res) in enumerate(zip(groups, pool.imap(_np_ik_work, [[takes[i] for i in g] for g in groups]))):
                for i, r in zip(g, res):
                    out[i] = r
                if (gi + 1) % 20 == 0 or gi + 1 == len(groups):
                    SAY(f"  IK (numpy, {self.workers} procs) {gi + 1}/{len(groups)} chunks")
        return out


_NP_ROBOT = None


def _np_ik_init(robot):
    global _NP_ROBOT
    os.environ["OMP_NUM_THREADS"] = "1"
    _NP_ROBOT = make(robot)


def _np_ik_work(takes):
    return [(pr, R, q, e) for (pr, R, q), e, _ in _NP_ROBOT.retarget_many(takes)]


# --------------------------------------------------------------------------------------------- main
def describe(rep: dict, desc: dict) -> str:
    if rep["source"] == "cmu":
        return desc.get(rep["clip"], "")
    if rep["source"].startswith("bandai"):
        parts = rep["clip"].split("_")
        return f"{parts[1].replace('-', ' ')}, {parts[2]}" if len(parts) >= 4 else rep["clip"]
    if rep["source"] == "lafan1":
        return re.sub(r"\d+$", "", rep["clip"].rsplit("_", 1)[0])
    return rep["clip"].split("/")[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=common.cpu_limit())
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    ap.add_argument("--only", default="cmu,bandai,lafan1,hiphi,lafan1_g1,amass_g1")
    ap.add_argument("--out", default=SHARDS)
    ap.add_argument("--max-takes", type=int, default=0, help="debug: limit takes per source")
    a = ap.parse_args(argv)
    t_all = time.time()
    only = set(s for s in a.only.split(",") if s)
    if ROBOT != "g1":
        dropped = only & {"lafan1_g1", "amass_g1"}
        only -= dropped
        if dropped:
            SAY(f"robot {ROBOT}: G1-native sources {sorted(dropped)} are not used (every source goes through IK)")
    os.makedirs(a.out, exist_ok=True)
    mj = os.path.join(RAW, SPECS[ROBOT]["mjcf"])
    if not os.path.exists(mj):
        common.die(f"robot model {mj} is missing (fetch_data.py clones unitree_mujoco)")

    # 1. BVH -> canonical keypoints (CPU pool)
    jobs = bvh_jobs(only)
    if a.max_takes:
        per = {}
        jobs = [j for j in jobs if per.setdefault(j["source"], []).append(1) or len(per[j["source"]]) <= a.max_takes]
    SAY(f"robot {ROBOT}: {len(jobs)} BVH takes from {sorted({j['source'] for j in jobs})}, {a.workers} workers")
    t0 = time.time()
    reps, kps = [], []
    if jobs:
        ctx = get_context("fork")
        with ctx.Pool(a.workers, initializer=os.environ.__setitem__, initargs=("OMP_NUM_THREADS", "1")) as pool:
            for i, (rep, kp) in enumerate(pool.imap(bvh_work, jobs, chunksize=4)):
                reps.append(rep); kps.append(kp)
                if (i + 1) % 2000 == 0:
                    SAY(f"  parsed {i + 1}/{len(jobs)} takes, {time.time() - t0:.0f}s")
    # HiPHI arrives already parsed (fetch_data.canon_cache): one npz of canonical keypoints per archive
    if "hiphi" in only:
        for p in sorted(glob.glob(os.path.join(RAW, "hiphi", "canon", "*.npz"))):
            z = np.load(p, allow_pickle=False)
            # read each array ONCE: NpzFile re-reads the whole member from disk on every z["kp"] access
            rr = json.loads(str(z["reps"])); off = z["offsets"]; kp_all = z["kp"]; k = 0
            for rep in rr:
                n = rep.pop("_n", None)
                if n is None:
                    reps.append(rep); kps.append(None); continue
                reps.append(rep); kps.append(kp_all[off[k]:off[k + 1]].astype(np.float64)); k += 1
            del kp_all
        SAY(f"  HiPHI from keypoint caches: {sum(1 for r in reps if r.get('source') == 'hiphi')} takes")
    t_parse = time.time() - t0
    keep = [i for i, k in enumerate(kps) if k is not None]
    SAY(f"parsed {len(jobs)} takes in {t_parse:.0f}s: {len(keep)} kept, {len(jobs) - len(keep)} rejected")

    # 2. IK (GPU when it matches pine/g1.py)
    rt = Retargeter(a.device, a.workers)
    sols = [None] * len(kps)
    t_ik = 0.0
    if keep:
        rng = np.random.default_rng(0)
        sample = [kps[i] for i in rng.permutation(keep)[:60]]
        chk = rt.calibrate(sample)
        SAY(f"IK check vs pine/g1.py: {json.dumps({k: v for k, v in chk.items()})[:400]} -> backend {rt.backend}")
        t0 = time.time()
        res = rt.run([kps[i] for i in keep])
        for i, r in zip(keep, res):
            sols[i] = r
        t_ik = time.time() - t0
        n_fr = sum(len(kps[i]) for i in keep)
        SAY(f"IK: {n_fr:,} frames in {t_ik:.0f}s ({t_ik / max(n_fr, 1) * 1e3:.3f} ms/frame) on {rt.backend}")

    # 3. features
    desc = cmu_descriptions()
    clips, feats, errs = [], [], []
    for rep, kp, sol in zip(reps, kps, sols):
        rep["split"] = split_of(rep["source"], rep["clip"], rep["subject"])
        rep["desc"] = describe(rep, desc)
        if kp is None or sol is None:
            clips.append(rep); continue
        pr, R, q, e = sol
        f = F.g1_features(pr, R, q)
        if not np.isfinite(f).all():
            rep["reject"] = "non-finite IK solution"; clips.append(rep); continue
        rep["ik_err_cm"] = float(e.mean() * 100)
        errs.append(e.astype(np.float32))
        rep["_feat"] = len(feats); feats.append(f); clips.append(rep)
    # G1-native sources
    nj = native_jobs(only)
    if nj:
        SAY(f"{len(nj)} G1-native takes ({sorted({j['source'] for j in nj})}); {HELD_NOTE}")
        ctx = get_context("fork")
        with ctx.Pool(a.workers) as pool:
            for rep, f in pool.imap(native_work, nj, chunksize=8):
                rep["split"] = split_of(rep["source"], rep["clip"], rep["subject"])
                rep["desc"] = describe(rep, desc)
                if f is not None:
                    rep["_feat"] = len(feats); feats.append(f)
                clips.append(rep)
        assumed = sum(1 for j in nj if j["fps_from"].startswith("ASSUMED"))
        if assumed:
            SAY(f"WARNING: {assumed} AMASS files had no frame rate in their name; 30 fps was ASSUMED for them")
    kept = [c for c in clips if "_feat" in c]
    if not kept:
        common.die("no usable takes: nothing to train on (see fetch_report.json and the rejects above)")

    # 4. order, normalise with train statistics, write
    kept.sort(key=lambda c: (c["source"], c["clip"]))
    lens = np.array([len(feats[c["_feat"]]) for c in kept], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    raw = np.concatenate([feats[c["_feat"]] for c in kept]).astype(np.float32)
    split = np.array([c["split"] for c in kept])
    for sp in ("train", "val", "test"):
        if not (split == sp).any():
            common.die(f"no {sp} clips after QA; cannot build a fair split")
    tr = np.concatenate([np.arange(s, s + n) for s, n, sp in zip(starts, lens, split) if sp == "train"])
    mean = raw[tr].astype(np.float64).mean(0)
    std = np.maximum(raw[tr].astype(np.float64).std(0), 1e-3)
    X = ((raw - mean) / std).astype(np.float16)
    if not np.isfinite(X).all():
        common.die("non-finite normalised features (float16 overflow?)")
    for i, (c, s, n) in enumerate(zip(kept, starts, lens)):
        c.pop("_feat"); c.update(index=i, start=int(s), length=int(n))
    tmp = os.path.join(a.out, "features.tmp.npy")
    np.save(tmp, X)
    os.replace(tmp, os.path.join(a.out, "features.npy"))
    rejected = [dict(c) for c in clips if "_feat" not in c and "index" not in c]
    write_json(os.path.join(a.out, "clips.json"), dict(robot=ROBOT, clips=kept, rejected=rejected), indent=None)
    fps = canon.FPS
    by_split = {sp: int(lens[split == sp].sum()) for sp in ("train", "val", "test")}
    per_source = {}
    for c in kept:
        d = per_source.setdefault(c["source"], dict(takes=0, frames=0, subjects=set()))
        d["takes"] += 1; d["frames"] += c["length"]; d["subjects"].add(c["subject"])
    rej = {}
    for c in rejected:
        key = c["reject"].split(" (")[0].split(":")[0]
        rej[key] = rej.get(key, 0) + 1
    e_all = np.concatenate(errs) if errs else None
    stats = dict(
        robot=ROBOT, robot_name=SPECS[ROBOT]["name"], names=list(F.NAMES), D=F.D, active_joints=[int(j) for j in F.ACTIVE],
        held_at_zero=SPECS[ROBOT]["held"], normalized=True, dtype="float16", fps=fps,
        mean=mean.tolist(), std=std.tolist(), frames=int(lens.sum()), hours=float(lens.sum() / fps / 3600),
        frames_by_split=by_split, hours_by_split={k: v / fps / 3600 for k, v in by_split.items()},
        takes=len(kept), takes_in=len(clips), rejects=rej,
        per_source={k: dict(takes=v["takes"], frames=v["frames"], hours=v["frames"] / fps / 3600, subjects=len(v["subjects"]))
                    for k, v in sorted(per_source.items())},
        ik=dict(backend=rt.backend, check=rt.check, seconds=t_ik,
                mean_err_cm=float(e_all.mean() * 100) if e_all is not None else None,
                feet_err_cm=float(e_all[:, 15:19].mean() * 100) if e_all is not None else None,
                p95_err_cm=float(np.percentile(e_all, 95) * 100) if e_all is not None else None),
        native_note=HELD_NOTE if nj else None, parse_seconds=t_parse, wall_seconds=time.time() - t_all,
        split_rule="scripts/ingest.split_of: CMU subject % 10 == 7 -> test, else crc32(clip) % 20 == 0 -> val; "
                   "other sources crc32(clip) % 20: 7 -> test, 0 -> val",
        built=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_json(os.path.join(a.out, "stats.json"), stats)
    SAY(f"shards: {stats['frames']:,} frames ({stats['hours']:.1f} h; train {stats['hours_by_split']['train']:.1f} h), "
        f"{len(kept)} takes, sources {list(stats['per_source'])} -> {a.out} ({time.time() - t_all:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
