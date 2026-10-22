"""BONES-SEED (bones-studio/seed: 522 actors, BVH at 120 fps) -> the same robot features, merged with phase-1 shards.

    python gpu/seed_data.py            (phase 2 starts it in the background; it never stops phase 2 on failure)

1. Download the metadata and ONE motion archive (soma_proportional, else soma_uniform) from Hugging Face with HF_TOKEN,
   then stream it: unpack 3,000 BVH files, parse them into canonical keypoints, delete them, repeat. Mirrored copies are
   skipped (they add no people). The disk never holds more than the archive and one batch.
2. Map the SOMA skeleton onto the 19 canonical keypoints by the rig's hierarchy (pelvis -> hips -> knees -> ankles,
   chest -> upper arms -> elbows -> hands), not by guessing names, and detect the up axis from the first take.
3. Parse, QA and resample with build_shards.bvh_work, retarget with build_shards.Retargeter (GPU IK), build features,
   normalise them with the PHASE-1 statistics, and write $WORKSPACE/data/shards_all = phase-1 shards + SEED.
Each clip's subject is "seed_<actor_uid>" from the metadata. Writes $WORKSPACE/data/shards_all/SEED_OK when done,
or SEED_FAILED with the reason. Logs to $PINE_LOGS/seed.log.
"""
from __future__ import annotations
import csv, glob, io, json, os, re, shutil, sys, tarfile, time
from multiprocessing import get_context
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common  # noqa: E402
import numpy as np  # noqa: E402

SAY = lambda m: common.log(m, "seed")  # noqa: E731
REPO = "bones-studio/seed"
RAW = os.path.join(common.RAW, "seed")
OUT = os.path.join(os.path.dirname(common.SHARDS), "shards_all")
MAX_TAKES = int(os.environ.get("PINE_SEED_MAX_TAKES", "0"))


# ----------------------------------------------------------------------------------------------- skeleton mapping
def side_of(name: str):
    n = name.lower()
    if "left" in n or re.search(r"(^|[_\.\s:-])l([_\.\s:-]|$)", n) or re.match(r"^l[A-Z_]", name) or n.endswith("_l"):
        return "l"
    if "right" in n or re.search(r"(^|[_\.\s:-])r([_\.\s:-]|$)", n) or re.match(r"^r[A-Z_]", name) or n.endswith("_r"):
        return "r"
    return None


def auto_map(names: list[str], parents: list[int]) -> dict:
    """Canonical keypoint -> joint name, from the hierarchy. Raises ValueError with the joint list if it cannot."""
    low = [n.split(":")[-1].lower() for n in names]
    kids = {i: [j for j, p in enumerate(parents) if p == i] for i in range(len(names))}
    real = lambda i: not low[i].endswith("_end")                      # BVH End Sites become "<parent>_end"
    def find(pat, side=None, exclude=None):
        for i, n in enumerate(low):
            if real(i) and re.search(pat, n) and (side is None or side_of(names[i].split(":")[-1]) == side) \
                    and not (exclude and re.search(exclude, n)):
                return i
        return None
    m = {}
    for s in ("l", "r"):
        ankle = find(r"ankle", s)
        toe = find(r"toe|ball", s)
        foot = find(r"foot", s, exclude=r"toe|ball|end")
        if ankle is None:
            ankle = foot
        elif toe is None and foot is not None and foot != ankle:
            toe = foot                                                   # SMPL: L_Ankle and L_Foot (= toes)
        if ankle is None:
            raise ValueError(f"no {s} ankle/foot joint in {names}")
        if toe is None:
            ch = [k for k in kids[ankle]]
            toe = ch[0] if ch else ankle
        knee = parents[ankle]; hip = parents[knee]
        hand = find(r"hand|wrist", s, exclude=r"thumb|index|middle|ring|pinky|finger|end")
        if hand is None:
            raise ValueError(f"no {s} hand/wrist joint in {names}")
        elbow = parents[hand]; shoulder = parents[elbow]
        m.update({f"{s}_ankle": ankle, f"{s}_toe": toe, f"{s}_knee": knee, f"{s}_hip": hip,
                  f"{s}_wrist": hand, f"{s}_elbow": elbow, f"{s}_shoulder": shoulder})
    first = lambda *v: next((x for x in v if x is not None), None)
    pelvis = first(find(r"^(hips?|pelvis)$"), find(r"hips?|pelvis"))
    if pelvis is None:   # lowest common ancestor of the two hips
        a = set(); i = m["l_hip"]
        while i >= 0:
            a.add(i); i = parents[i]
        i = m["r_hip"]
        while i not in a and i >= 0:
            i = parents[i]
        pelvis = i
    head = first(find(r"^head$"), find(r"head", exclude=r"end|top"))
    neck = find(r"neck")
    if head is None or neck is None:
        raise ValueError(f"no head/neck joint in {names}")
    chain = []                       # pelvis -> ... -> neck
    i = neck
    while i >= 0 and i != pelvis:
        chain.append(i); i = parents[i]
    chain = chain[::-1]               # first joint above the pelvis ... neck
    torso = [j for j in chain if j != neck]
    spine = torso[0] if torso else neck
    chest = torso[-1] if torso else neck
    m.update(pelvis=pelvis, spine=spine, chest=chest, neck=neck, head=head)
    return {k: names[v] for k, v in m.items()}


# ----------------------------------------------------------------------------------------------- download
def hf():
    import huggingface_hub as hh
    return hh


def fetch() -> dict:
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        raise RuntimeError("HF_TOKEN is not set")
    hh = hf()
    files = hh.HfApi().list_repo_files(REPO, repo_type="dataset", token=tok)
    SAY(f"{REPO}: {len(files)} files; e.g. {files[:12]}")
    os.makedirs(RAW, exist_ok=True)
    meta = [f for f in files if f.startswith("metadata/") or f.lower().endswith((".csv", ".jsonl", ".parquet")) and "/" not in f]
    for f in meta[:40]:
        try:
            hh.hf_hub_download(REPO, f, repo_type="dataset", local_dir=RAW, token=tok)
        except Exception as e:  # noqa: BLE001
            SAY(f"  metadata {f}: {e}")
    arch = None
    for want in ("soma_proportional", "soma_uniform"):
        cands = [f for f in files if want in f and f.endswith((".tar.gz", ".tgz", ".tar", ".tar.zst"))]
        if cands:
            arch = sorted(cands); break
    if not arch:
        raise RuntimeError(f"no SOMA motion archive in the repository ({len([f for f in files if f.lower().endswith('.bvh')])} loose BVH)")
    local = []
    for a in arch:
        t0 = time.time()
        p = hh.hf_hub_download(REPO, a, repo_type="dataset", local_dir=RAW, token=tok)
        SAY(f"downloaded {a} ({os.path.getsize(p)/1e9:.1f} GB) in {time.time()-t0:.0f}s")
        local.append(p)
    return dict(archives=arch, local=local)


def actor_table() -> tuple[dict, dict]:
    """BVH file stem -> actor id, and stem -> is_mirror, from whatever metadata table has those columns."""
    actors, mirror = {}, {}
    tables = glob.glob(os.path.join(RAW, "metadata", "**", "*.csv"), recursive=True) + \
             glob.glob(os.path.join(RAW, "**", "*.jsonl"), recursive=True)
    for p in tables:
        try:
            rows = list(csv.DictReader(open(p, newline="", errors="replace"))) if p.endswith(".csv") else \
                   [json.loads(l) for l in open(p, errors="replace") if l.strip()]
        except Exception as e:  # noqa: BLE001
            SAY(f"  could not read {p}: {e}"); continue
        if not rows:
            continue
        cols = list(rows[0].keys())
        acol = next((c for c in cols if c.lower() == "actor_uid"), None) or next((c for c in cols if "actor" in c.lower()), None)
        fcols = [c for c in cols if any(k in c.lower() for k in ("file", "path", "bvh", "filename", "name", "motion", "clip"))]
        mcol = next((c for c in cols if "mirror" in c.lower()), None)
        SAY(f"  {os.path.relpath(p, RAW)}: {len(rows)} rows, actor column {acol}, file columns {fcols[:4]}, mirror {mcol}")
        if not acol or not fcols:
            continue
        for r in rows:
            for fc in fcols:
                v = str(r.get(fc) or "")
                if not v:
                    continue
                stem = os.path.splitext(os.path.basename(v))[0]
                actors.setdefault(stem, str(r[acol]))
                if mcol:
                    mirror.setdefault(stem, str(r.get(mcol)).lower() in ("1", "true", "yes"))
    return actors, mirror


# ----------------------------------------------------------------------------------------------- main
def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for f in ("SEED_OK", "SEED_FAILED"):
        if os.path.exists(os.path.join(OUT, f)):
            os.remove(os.path.join(OUT, f))
    try:
        return run()
    except Exception as e:  # noqa: BLE001
        import traceback
        msg = f"{type(e).__name__}: {e}"
        SAY(f"FAILED: {msg}\n{traceback.format_exc()[-3000:]}")
        open(os.path.join(OUT, "SEED_FAILED"), "w").write(msg[:2000])
        return 1


def run() -> int:
    t_all = time.time()
    shards1 = common.SHARDS
    if not os.path.exists(os.path.join(shards1, "stats.json")):
        raise RuntimeError("phase-1 shards are missing")
    hip = os.path.join(common.RAW, "hiphi", "extracted")
    if os.path.isdir(hip) and os.environ.get("PINE_KEEP_HIPHI_RAW") != "1":
        SAY("freeing disk: removing HiPHI raw BVH (the phase-1 shards hold its features)")
        shutil.rmtree(hip, ignore_errors=True)
    rep = fetch()
    actors, mirror = actor_table()
    SAY(f"metadata: {len(actors)} files with an actor id ({len(set(actors.values()))} actors), {sum(mirror.values())} marked mirrored")
    from pine import mocap_io, canon
    import build_shards as BS
    stem = lambda f: os.path.splitext(os.path.basename(f))[0]
    tmp = os.path.join(RAW, "tmp_batch")
    state = dict(mapped=False)
    reps, kps = [], []

    def process(paths):
        if not paths:
            return
        if not state["mapped"]:          # skeleton and up axis from the first file
            t = mocap_io.read_bvh(paths[0], 0.01, "seed", stem(paths[0]), "x")
            names = [n.split(":")[-1] for n in t.names]
            SAY(f"SOMA joints ({len(names)}): {names}")
            amap = auto_map(names, t.parents)
            SAY(f"keypoint map: {amap}")
            canon.MAPS["seed"] = amap; state["map"] = amap
            kp = canon.to_canonical(t, "seed")
            up = np.median(kp[:, canon.KP["head"]] - kp[:, canon.KP["l_ankle"]], axis=0)
            if abs(up[2]) < 0.5 * np.linalg.norm(up):
                SAY(f"up axis: body axis {np.round(up, 3)} after a Y-up conversion -> treating the files as Z-up")
                mocap_io.Y_UP_TO_Z_UP = np.eye(3)
            state["mapped"] = True
        jobs = [dict(source="seed", path=f, clip=stem(f), subject="seed_" + actors.get(stem(f), "unknown"), unit=0.01,
                     skel="seed", drop=0, check_unit=True) for f in paths]
        # leave CPUs for the training run that phase 2 has on the GPU at the same time
        with get_context("fork").Pool(max(1, common.cpu_limit() - 3), initializer=os.environ.__setitem__, initargs=("OMP_NUM_THREADS", "1")) as pool:
            for r, k in pool.imap(BS.bvh_work, jobs, chunksize=8):
                reps.append(r); kps.append(None if k is None else k.astype(np.float32))

    t0 = time.time(); batch = []
    for path in rep["local"]:
        mode = "r|gz" if path.endswith(("gz", "tgz")) else "r|"
        with tarfile.open(path, mode) as tf:
            for mm in tf:
                if not mm.isfile() or not mm.name.lower().endswith(".bvh") or "mirror" in mm.name.lower() \
                        or mirror.get(stem(mm.name), False):
                    continue
                target = os.path.abspath(os.path.join(tmp, mm.name))
                if not target.startswith(os.path.abspath(tmp) + os.sep):
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as out:
                    shutil.copyfileobj(tf.extractfile(mm), out)
                batch.append(target)
                if len(batch) >= 3000:
                    process(batch); shutil.rmtree(tmp, ignore_errors=True); batch = []
                    SAY(f"  {len(reps)} takes parsed ({sum(k is not None for k in kps)} kept), {time.time()-t0:.0f}s")
                if MAX_TAKES and len(reps) + len(batch) >= MAX_TAKES:
                    break
        os.remove(path)
    process(batch); shutil.rmtree(tmp, ignore_errors=True)
    amap = state.get("map")
    keep = [i for i, k in enumerate(kps) if k is not None]
    rej = {}
    for r in reps:
        if "reject" in r:
            key = r["reject"].split(" (")[0].split(":")[0]; rej[key] = rej.get(key, 0) + 1
    SAY(f"parsed {len(reps)} takes in {time.time()-t0:.0f}s: {len(keep)} kept, rejects {rej}; "
        f"{len({reps[i]['subject'] for i in keep})} actors")
    if not keep:
        raise RuntimeError(f"every SEED take was rejected: {rej}; first: {reps[0].get('reject') if reps else None}")
    rt = BS.Retargeter("cuda", max(1, common.cpu_limit() - 3))
    rng = np.random.default_rng(0)
    chk = rt.calibrate([kps[i].astype(np.float64) for i in rng.permutation(keep)[:60]])
    SAY(f"IK check: passed={chk.get('passed')} backend {rt.backend}")
    t0 = time.time(); sols = rt.run([kps[i].astype(np.float64) for i in keep])
    SAY(f"IK: {sum(len(kps[i]) for i in keep):,} frames in {time.time()-t0:.0f}s")
    from pine import features as F
    st = json.load(open(os.path.join(shards1, "stats.json")))
    mean, std = np.asarray(st["mean"]), np.asarray(st["std"])
    cl1 = json.load(open(os.path.join(shards1, "clips.json")))
    clips1 = cl1["clips"] if isinstance(cl1, dict) else cl1
    X1 = np.load(os.path.join(shards1, "features.npy"), mmap_mode="r")
    new_clips, feats, errs = [], [], []
    start = int(X1.shape[0])
    for i, sol in zip(keep, sols):
        pr, R, q, e = sol
        f = F.g1_features(pr, R, q)
        if not np.isfinite(f).all() or len(f) < 36:
            continue
        r = dict(reps[i]); r.update(index=len(clips1) + len(new_clips), start=start, length=int(len(f)), split="seed",
                                    desc=r.get("clip"), ik_err_cm=float(e.mean() * 100))
        new_clips.append(r); feats.append(((f - mean) / std).astype(np.float16)); errs.append(float(e.mean()))
        start += len(f)
    Xs = np.concatenate(feats)
    tmp = os.path.join(OUT, "features.tmp.npy")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(X1.shape[0] + Xs.shape[0], X1.shape[1]))
    step = 2_000_000
    n1 = X1.shape[0]
    for a in range(0, n1, step):
        b = min(a + step, n1)
        out[a:b] = X1[a:b]
    out[X1.shape[0]:] = Xs; out.flush(); del out
    os.replace(tmp, os.path.join(OUT, "features.npy"))
    common.write_json(os.path.join(OUT, "clips.json"), dict(robot=st["robot"], clips=clips1 + new_clips), indent=None)
    hours = float(Xs.shape[0] / 20 / 3600)
    st2 = dict(st); st2["per_source"] = dict(st.get("per_source") or {})
    st2["per_source"]["seed"] = dict(takes=len(new_clips), frames=int(Xs.shape[0]), hours=hours,
                                     subjects=len({c["subject"] for c in new_clips}))
    st2.update(frames=int(X1.shape[0] + Xs.shape[0]), hours=float((X1.shape[0] + Xs.shape[0]) / 20 / 3600),
               seed=dict(takes=len(new_clips), hours=hours, actors=len({c["subject"] for c in new_clips}), rejects=rej,
                         keypoint_map=amap, ik_mean_err_cm=float(np.mean(errs) * 100), archives=rep.get("archives"),
                         seconds=time.time() - t_all, split="split 'seed': phase2.py assigns held-out actors"))
    common.write_json(os.path.join(OUT, "stats.json"), st2)
    open(os.path.join(OUT, "SEED_OK"), "w").write(json.dumps(st2["seed"])[:4000])
    SAY(f"SEED done: {len(new_clips)} takes, {hours:.1f} h, {st2['seed']['actors']} actors -> {OUT} ({time.time()-t_all:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
