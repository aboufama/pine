"""Download the public full-body motion datasets onto the pod. Every step is optional, logged, and never fatal.

    python gpu/fetch_data.py [--only cmu,bandai,...] [--skip hiphi] [--hiphi-parts 4]

Layout: $PINE_RAW/<name>/repo (git sources) or $PINE_RAW/<name>/ (Hugging Face), plus $PINE_RAW/fetch_report.json.
  cmu             git clone --depth 1 https://github.com/una-dinosauria/cmu-mocap
  bandai          git clone --depth 1 https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset
  lafan1          git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset, git lfs pull lafan1/lafan1.zip,
                  unzip to lafan1/bvh/ (BVH, 30 fps, cm, Mixamo-style names, CC BY-NC-ND 4.0)
  unitree_mujoco  git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco (robot models; REQUIRED later)
  hiphi           HF noitomrobotics/HiPHI: gated (needs HF_TOKEN with granted access). BVH, 90 Hz, cm, Y-up, packed in
                  32 .tar.zst archives (~215 GB). One archive at a time: download, unpack, parse every take into
                  canonical keypoints (hiphi/canon/<archive>.npz, via build_shards.bvh_work), delete archive and text.
                  PINE_HIPHI_PARTS / --hiphi-parts limits how many.
  lafan1_g1       HF unitreerobotics/LAFAN1_Retargeting_Dataset, g1/*.csv  (only fetched when PINE_ROBOT=g1)
  amass_g1        HF fleaven/Retargeted_AMASS_for_robotics, g1/**/*.npy    (only fetched when PINE_ROBOT=g1)
A source that already has its .ok marker is not downloaded again.
"""
from __future__ import annotations
import argparse, csv, glob, io, json, os, shutil, subprocess, sys, tarfile, time, zipfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common  # noqa: E402
from common import write_json  # noqa: E402

SAY = lambda m: common.log(m, "fetch")  # noqa: E731
RAW = common.RAW

GIT = {
    "cmu": dict(url="https://github.com/una-dinosauria/cmu-mocap", depth=1),
    "bandai": dict(url="https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset", depth=1),
    "lafan1": dict(url="https://github.com/ubisoft/ubisoft-laforge-animation-dataset", depth=None),
    "unitree_mujoco": dict(url="https://github.com/unitreerobotics/unitree_mujoco", depth=1),
}
HF = {
    "hiphi": dict(repo="noitomrobotics/HiPHI", robots=("h2", "g1")),
    "lafan1_g1": dict(repo="unitreerobotics/LAFAN1_Retargeting_Dataset", robots=("g1",)),
    "amass_g1": dict(repo="fleaven/Retargeted_AMASS_for_robotics", robots=("g1",)),
}
ORDER = ["unitree_mujoco", "cmu", "bandai", "lafan1", "lafan1_g1", "amass_g1", "hiphi"]


# --------------------------------------------------------------------------------------------- helpers
def sh(cmd: list[str], cwd: str | None = None, timeout: float = 7200, env: dict | None = None) -> tuple[int, str]:
    """Run a command, return (exit code, last lines of output). Never raises."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env=dict(os.environ, GIT_TERMINAL_PROMPT="0", **(env or {})))
        out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        return r.returncode, "\n".join(out[-6:])
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    except OSError as e:
        return 127, str(e)


def du_bytes(path: str) -> int:
    n = 0
    for dp, _, fns in os.walk(path):
        for f in fns:
            try:
                n += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return n


def count(path: str, pattern: str) -> int:
    return len(glob.glob(os.path.join(path, pattern), recursive=True))


def du_gb(path: str) -> float:
    """Bytes actually used under path (statvfs on a RunPod network volume reports the whole filesystem, not the quota)."""
    try:
        out = subprocess.run(["du", "-s", "--block-size=1M", path], capture_output=True, text=True, timeout=300).stdout
        return float(out.split()[0]) / 1024 if out else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def canon_cache(bvh_dir: str, out_npz: str) -> int:
    """Every non-mirror HiPHI BVH under bvh_dir -> one npz of canonical keypoints + per-take reports. Returns takes kept."""
    import build_shards as BS
    from multiprocessing import get_context
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    meta = BS.hiphi_actors()
    jobs = []
    for p in sorted(glob.glob(os.path.join(bvh_dir, "**", "*.bvh"), recursive=True)):
        if "__mirror" in p:
            continue
        mid = os.path.basename(p)[:-4]
        if mid not in meta and os.path.basename(os.path.dirname(p)) in meta:
            mid = os.path.basename(os.path.dirname(p))
        actor, text = meta.get(mid, ("unknown", ""))
        jobs.append(dict(source="hiphi", path=p, clip=mid, subject="hiphi_" + actor, unit=0.01, skel="mixamo",
                         drop=0, check_unit=True, desc=text))
    reps, kps = [], []
    with get_context("fork").Pool(os.cpu_count() or 4, initializer=os.environ.__setitem__, initargs=("OMP_NUM_THREADS", "1")) as pool:
        for rep, kp in pool.imap(BS.bvh_work, jobs, chunksize=4):
            if kp is not None:
                rep["_n"] = len(kp); kps.append(kp.astype(np.float32))
            reps.append(rep)
    off = np.concatenate([[0], np.cumsum([len(k) for k in kps])]).astype(np.int64)
    np.savez(out_npz + ".tmp.npz", kp=np.concatenate(kps) if kps else np.zeros((0, 19, 3), np.float32), offsets=off,
             reps=np.array(json.dumps(reps)))
    os.replace(out_npz + ".tmp.npz", out_npz)
    return len(kps)


def free_gb(path: str) -> float:
    os.makedirs(path, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def marker(name: str) -> str:
    return os.path.join(RAW, name, ".ok")


def done_before(name: str) -> dict | None:
    try:
        return json.load(open(marker(name)))
    except (OSError, ValueError):
        return None


def mark(name: str, info: dict) -> None:
    write_json(marker(name), info)


def git_clone(name: str) -> dict:
    spec = GIT[name]
    dest = os.path.join(RAW, name, "repo")
    if os.path.isdir(os.path.join(dest, ".git")):
        rc, out = sh(["git", "-C", dest, "rev-parse", "HEAD"], timeout=60)
        if rc == 0:
            return dict(status="present", path=dest, commit=out.strip().splitlines()[-1])
        shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # Abort a stalled transfer (under 200 KB/s for 90 s) instead of waiting out the hour-long timeout, then retry.
    cmd = ["git", "-c", "http.lowSpeedLimit=200000", "-c", "http.lowSpeedTime=90", "clone", "--quiet"] \
        + (["--depth", str(spec["depth"])] if spec["depth"] else []) + [spec["url"], dest]
    env = {"GIT_LFS_SKIP_SMUDGE": "1"} if name == "lafan1" else None
    for attempt in range(3):
        rc, out = sh(cmd, timeout=1800, env=env)
        if rc == 0:
            break
        SAY(f"  {name}: git clone failed (attempt {attempt + 1}/3): {out}")
        shutil.rmtree(dest, ignore_errors=True)
        time.sleep(10 * (attempt + 1))
    else:
        return dict(status="failed", error=out, url=spec["url"])
    rc, out = sh(["git", "-C", dest, "rev-parse", "HEAD"], timeout=60)
    return dict(status="ok", path=dest, commit=out.strip().splitlines()[-1] if rc == 0 else None, url=spec["url"])


# --------------------------------------------------------------------------------------------- git sources
def fetch_cmu() -> dict:
    r = git_clone("cmu")
    if r["status"] in ("ok", "present"):
        r["bvh_files"] = count(r["path"], "data/*/*.bvh")
        r["license"] = "CMU Graphics Lab Motion Capture Database: free to use, the data may not be resold"
    return r


def fetch_bandai() -> dict:
    r = git_clone("bandai")
    if r["status"] in ("ok", "present"):
        r["bvh_files"] = count(r["path"], "dataset/*/data/*.bvh")
        r["license"] = "CC BY-NC 4.0"
    return r


def fetch_unitree() -> dict:
    r = git_clone("unitree_mujoco")
    if r["status"] in ("ok", "present"):
        from pine.g1 import SPECS
        r["models"] = {k: os.path.exists(os.path.join(RAW, v["mjcf"])) for k, v in SPECS.items()}
        r["license"] = "BSD-3-Clause"
        if not r["models"].get(common.ROBOT):
            r["status"] = "failed"; r["error"] = f"model for {common.ROBOT} not found at {SPECS[common.ROBOT]['mjcf']}"
    return r


def fetch_lafan1() -> dict:
    r = git_clone("lafan1")
    if r["status"] not in ("ok", "present"):
        return r
    repo = r["path"]
    z = os.path.join(repo, "lafan1", "lafan1.zip")
    out_dir = os.path.join(repo, "lafan1", "bvh")
    if count(out_dir, "*.bvh") >= 70:
        r.update(bvh_files=count(out_dir, "*.bvh"), bvh_dir=out_dir)
        return r
    rc, out = sh(["git", "lfs", "version"], timeout=60)
    if rc != 0:
        SAY("  lafan1: git-lfs missing, trying apt-get install git-lfs")
        sh(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq git-lfs"], timeout=900)
    sh(["git", "-C", repo, "lfs", "install", "--local"], timeout=120)
    rc, out = sh(["git", "-C", repo, "lfs", "pull", "--include", "lafan1/lafan1.zip"], timeout=3600)
    if rc != 0 or not zipfile.is_zipfile(z):
        r.update(status="failed", error=f"git lfs pull of lafan1/lafan1.zip failed: {out}")
        return r
    try:
        n = unzip_bvh(z, out_dir)
    except (zipfile.BadZipFile, OSError, ValueError) as e:
        r.update(status="failed", error=f"unzip failed: {e}")
        return r
    r.update(bvh_files=n, bvh_dir=out_dir, zip_bytes=os.path.getsize(z),
             license="CC BY-NC-ND 4.0", format="BVH, 30 fps, cm, Mixamo-style joint names")
    return r


def unzip_bvh(zpath: str, out_dir: str) -> int:
    """Extract only *.bvh members, flattened, with path traversal refused."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as zf:
        for m in zf.infolist():
            if m.is_dir() or not m.filename.lower().endswith(".bvh"):
                continue
            base = os.path.basename(m.filename.replace("\\", "/"))
            if not base or base.startswith(".") or base != os.path.basename(base):
                continue
            with zf.open(m) as src, open(os.path.join(out_dir, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            n += 1
    return n


# --------------------------------------------------------------------------------------------- Hugging Face
def hf_api():
    try:
        import huggingface_hub as hh
        return hh
    except ImportError:
        return None


def hf_token():
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def hf_snapshot(repo: str, dest: str, patterns: list[str]) -> str:
    hh = hf_api()
    return hh.snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, allow_patterns=patterns,
                                token=hf_token(), max_workers=8)


def hf_access(repo: str) -> tuple[bool, str]:
    """(ok, reason). Tries to read the repo's file list; gated repos need a token with granted access."""
    hh = hf_api()
    if hh is None:
        return False, "huggingface_hub is not installed"
    try:
        hh.HfApi().list_repo_files(repo, repo_type="dataset", token=hf_token())
        return True, "ok"
    except Exception as e:  # noqa: BLE001  (GatedRepoError, RepositoryNotFoundError, HTTPError, ...)
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:300]}"


def fetch_lafan1_g1() -> dict:
    ok, why = hf_access(HF["lafan1_g1"]["repo"])
    if not ok:
        return dict(status="skipped", reason=why)
    dest = os.path.join(RAW, "lafan1_g1")
    hf_snapshot(HF["lafan1_g1"]["repo"], dest, ["g1/*.csv", "README.md"])
    files = sorted(glob.glob(os.path.join(dest, "g1", "*.csv")))
    r = dict(status="ok" if files else "failed", path=dest, csv_files=len(files), fps=30,
             license="CC BY-NC-ND 4.0", format="root_joint(XYZQXQYQZQW) + 29 G1 joints per row")
    if files:
        r["check"] = check_lafan1_g1_csv(files[0])
    return r


def check_lafan1_g1_csv(path: str) -> dict:
    """Shape and sanity of one Unitree retargeted LAFAN1 csv: 36 columns, unit quaternion in columns 3:7 (xyzw)."""
    import numpy as np
    from build_shards import read_g1_csv           # one parser for fetch-time checks and the build
    arr = read_g1_csv(path)
    qn = np.linalg.norm(arr[:, 3:7], axis=1)
    return dict(rows=int(arr.shape[0]), cols=int(arr.shape[1]), quat_norm_mean=float(qn.mean()),
                root_z_median=float(np.median(arr[:, 2])))


def fetch_amass_g1() -> dict:
    ok, why = hf_access(HF["amass_g1"]["repo"])
    if not ok:
        return dict(status="skipped", reason=why)
    dest = os.path.join(RAW, "amass_g1")
    hf_snapshot(HF["amass_g1"]["repo"], dest, ["g1/**/*.npy", "g1/*.md", "g1/*.txt", "g1/*.py", "README.md"])
    files = sorted(glob.glob(os.path.join(dest, "g1", "**", "*.npy"), recursive=True))
    return dict(status="ok" if files else "failed", path=dest, npy_files=len(files),
                format="rows [-1, 36]: root pos, quaternion xyzw, 29 joints; fps from the file name (_poses_<fps>_jpos)",
                license="CC BY 4.0 tag; each AMASS sub-dataset keeps its own licence")


def fetch_hiphi(max_parts: int) -> dict:
    repo = HF["hiphi"]["repo"]
    if not hf_token():
        return dict(status="skipped", reason="HF_TOKEN is not set (HiPHI is gated: request access on its page first)")
    ok, why = hf_access(repo)
    if not ok:
        return dict(status="skipped", reason=f"no access to {repo} with this HF_TOKEN ({why})")
    hh = hf_api()
    dest = os.path.join(RAW, "hiphi")
    os.makedirs(dest, exist_ok=True)
    hf_snapshot(repo, dest, ["metadata/*", "data/*.csv", "README.md", "LICENSE.md", "docs/*"])
    files = [f for f in hh.HfApi().list_repo_files(repo, repo_type="dataset", token=hf_token())
             if f.startswith("data/") and f.endswith(".tar.zst")]
    files = sorted(files)[: max_parts] if max_parts > 0 else sorted(files)
    zstd_ok = shutil.which("zstd") is not None or _has_zstandard()
    if not zstd_ok:
        sh(["bash", "-lc", "apt-get update -qq && apt-get install -y -qq zstd"], timeout=900)
        zstd_ok = shutil.which("zstd") is not None
    if not zstd_ok:
        return dict(status="failed", reason="no zstd decompressor (install zstd)")
    done, failed = [], []
    ex_root = os.path.join(dest, "extracted")
    for f in files:
        flag = os.path.join(dest, ".done_" + os.path.basename(f))
        if os.path.exists(flag):
            done.append(f); continue
        if du_gb(os.path.dirname(dest)) > float(os.environ.get("PINE_VOLUME_GB", "300")) * 0.85:
            SAY(f"  hiphi: stopping, {du_gb(os.path.dirname(dest)):.0f} GB used of the volume"); break
        t0 = time.time()
        try:
            for attempt in range(3):
                try:
                    local = hh.hf_hub_download(repo, f, repo_type="dataset", local_dir=dest, token=hf_token())
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        raise
                    SAY(f"  hiphi: {os.path.basename(f)} download failed (attempt {attempt + 1}/3): {e}"[:300])
                    time.sleep(20 * (attempt + 1))
            # Unpacked BVH is several times the archive: parse each archive straight into canonical keypoints
            # (pine.canon, 20 fps) and delete the text before the next archive, so the disk never holds more than one.
            tmp = os.path.join(dest, "tmp_extract")
            shutil.rmtree(tmp, ignore_errors=True)
            n = extract_tar_zst(local, tmp, keep=(".bvh", ".json", ".csv"))
            os.remove(local)
            kept = canon_cache(tmp, os.path.join(dest, "canon", os.path.basename(f).split(".")[0] + ".npz"))
            shutil.rmtree(tmp, ignore_errors=True)
            open(flag, "w").write(json.dumps(dict(files=n, takes_kept=kept, seconds=time.time() - t0)))
            done.append(f)
            SAY(f"  hiphi: {os.path.basename(f)} -> {n} files, {kept} takes kept as keypoints, {time.time() - t0:.0f}s, "
                f"{du_gb(dest):.1f} GB on disk")
        except Exception as e:  # noqa: BLE001
            failed.append(dict(file=f, error=f"{type(e).__name__}: {e}"[:300]))
            SAY(f"  hiphi: {f} failed: {e}")
    bvh = len(glob.glob(os.path.join(dest, "canon", "*.npz")))
    return dict(status="ok" if done else "failed", path=dest, canon_archives=bvh, archives=len(done),
                archives_total=len([1 for _ in files]), failed=failed, bvh_files=bvh,
                license="ModalityNet Open Research License (non-commercial research)",
                format="BVH, 90 Hz, cm, Y-up; each motion also has a __mirror copy")


def _has_zstandard() -> bool:
    try:
        import zstandard  # noqa: F401
        return True
    except ImportError:
        return False


def extract_tar_zst(path: str, dest: str, keep=(".bvh",)) -> int:
    """Stream-extract a .tar.zst, keeping regular files with the given suffixes. Refuses unsafe paths."""
    os.makedirs(dest, exist_ok=True)
    dest_abs = os.path.abspath(dest)
    if shutil.which("zstd"):
        proc = subprocess.Popen(["zstd", "-dc", "--long=31", path], stdout=subprocess.PIPE)
        stream = proc.stdout
    else:
        import zstandard
        proc = None
        stream = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(open(path, "rb"))
    n = 0
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tf:
            for m in tf:
                if not m.isfile() or not m.name.lower().endswith(keep):
                    continue
                if "__mirror" in m.name and os.environ.get("PINE_KEEP_MIRRORS") != "1":
                    continue          # HiPHI ships a left-right mirror of every take; mirroring adds no information
                target = os.path.abspath(os.path.join(dest, m.name))
                if not target.startswith(dest_abs + os.sep):
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                src = tf.extractfile(m)
                with open(target + ".part", "wb") as out:
                    shutil.copyfileobj(src, out)
                os.replace(target + ".part", target)
                n += 1
    finally:
        if proc is not None:
            proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError(f"zstd exited with {proc.returncode} for {path}")
    return n


# --------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated subset of " + ",".join(ORDER))
    ap.add_argument("--skip", default=os.environ.get("PINE_SKIP", ""), help="comma-separated sources to skip")
    ap.add_argument("--hiphi-parts", type=int, default=common.env_int("PINE_HIPHI_PARTS", 0),
                    help="number of HiPHI archives to fetch (0 = all 32)")
    a = ap.parse_args(argv)
    os.makedirs(RAW, exist_ok=True)
    only = [s for s in a.only.split(",") if s]
    skip = set(s for s in a.skip.split(",") if s)
    robot = common.ROBOT
    report_path = os.path.join(RAW, "fetch_report.json")
    try:
        report = json.load(open(report_path))
    except (OSError, ValueError):
        report = {}
    report.update(robot=robot, started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), free_gb_before=round(free_gb(RAW), 1))
    report.setdefault("sources", {})
    fns = dict(cmu=fetch_cmu, bandai=fetch_bandai, lafan1=fetch_lafan1, unitree_mujoco=fetch_unitree,
               lafan1_g1=fetch_lafan1_g1, amass_g1=fetch_amass_g1, hiphi=lambda: fetch_hiphi(a.hiphi_parts))
    for name in ORDER:
        um = report["sources"].get("unitree_mujoco", {})
        if name != "unitree_mujoco" and not only and um and um.get("status") not in ("ok", "present"):
            break
        if only and name not in only:
            continue
        if name in skip:
            report["sources"][name] = dict(status="skipped", reason="PINE_SKIP / --skip"); continue
        if name in HF and robot not in HF[name]["robots"]:
            report["sources"][name] = dict(status="skipped", reason=f"G1-native data; used only when PINE_ROBOT=g1 (robot is {robot})")
            SAY(f"{name}: skipped (G1-native, robot is {robot})"); continue
        prev = done_before(name)
        if prev and prev.get("status") == "ok" and name != "hiphi":
            report["sources"][name] = dict(prev, status="ok", cached=True)
            SAY(f"{name}: already fetched"); continue
        SAY(f"{name}: fetching ...")
        t0 = time.time()
        try:
            res = fns[name]()
        except Exception as e:  # noqa: BLE001  (never fatal)
            res = dict(status="failed", error=f"{type(e).__name__}: {e}"[:500])
        res["seconds"] = round(time.time() - t0, 1)
        p = res.get("path")
        if p and os.path.isdir(p) and res.get("status") == "ok" and name != "hiphi":
            res["bytes"] = du_bytes(p)
        report["sources"][name] = res
        if res.get("status") == "ok":
            mark(name, res)
        SAY(f"{name}: {res.get('status')} in {res['seconds']:.0f}s "
            + json.dumps({k: v for k, v in res.items() if k not in ('path', 'status', 'seconds')})[:400])
        write_json(report_path, report)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["free_gb_after"] = round(free_gb(RAW), 1)
    write_json(report_path, report)
    um = report["sources"].get("unitree_mujoco", {})
    if not only and um.get("status") not in ("ok", "present"):
        SAY("FATAL: the robot model (unitree_mujoco) could not be fetched; this host's network is too slow or blocked")
        return 3
    SAY(f"fetch report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
