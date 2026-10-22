"""Shared paths, environment and logging for the GPU scale-up. No torch import here.

Import this module before any `pine.*` module: pine.g1 reads PINE_RAW and pine.features reads PINE_ROBOT
at import time, so the defaults below must be in the environment first.

Layout on the pod (PINE_WORKSPACE, default /workspace):
    data/raw/<name>/...        fetched datasets (fetch_data.py), fetch_report.json
    data/work/...              intermediate canonical / robot-space chunks (build_shards.py, resumable)
    data/shards/               features.npy, clips.json, stats.json (build_shards.py)
    runs/gpu/                  run_<size>_<fraction>.json, prediction.json, weights_<size>.npz (sweep_gpu.py)
    logs/                      everything that runs writes a log here
"""
from __future__ import annotations
import json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # the pine-engine folder
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

WORKSPACE = os.environ.setdefault("PINE_WORKSPACE", "/workspace")
DATA = os.path.join(WORKSPACE, "data")
RAW = os.environ.setdefault("PINE_RAW", os.path.join(DATA, "raw"))
WORK = os.path.join(DATA, "work")
SHARDS = os.environ.setdefault("PINE_SHARDS", os.path.join(DATA, "shards"))
RUNS = os.environ.setdefault("PINE_RUNS", os.path.join(WORKSPACE, "runs", "gpu"))
LOGS = os.environ.setdefault("PINE_LOGS", os.path.join(WORKSPACE, "logs"))
ROBOT = os.environ.setdefault("PINE_ROBOT", "h2")

_LOG_FILES: dict[str, object] = {}


def log(msg: str, name: str | None = None) -> None:
    """Print with a UTC timestamp and append to $PINE_LOGS/<name>.log (name defaults to the script name)."""
    name = name or os.path.splitext(os.path.basename(sys.argv[0] or "pine"))[0] or "pine"
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}Z] {msg}"
    print(line, flush=True)
    try:
        if name not in _LOG_FILES:
            os.makedirs(LOGS, exist_ok=True)
            _LOG_FILES[name] = open(os.path.join(LOGS, name + ".log"), "a", buffering=1)
        _LOG_FILES[name].write(line + "\n")
    except OSError:
        pass                                     # logging must never be the reason a run dies


def write_json(path: str, obj, indent: int | None = 1) -> None:
    """Atomic JSON write (a crash never leaves a half-written file behind)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def progress(stage: str, **fields) -> None:
    """Small machine-readable progress file ($PINE_LOGS/progress_<stage>.json) for the GitHub heartbeat."""
    try:
        write_json(os.path.join(LOGS, f"progress_{stage}.json"), dict(fields, stage=stage, at=time.time()))
    except OSError:
        pass


def env_float(name: str, default: float) -> float:
    v = os.environ.get(name, "")
    try:
        return float(v) if v.strip() else float(default)
    except ValueError:
        raise SystemExit(f"environment variable {name}={v!r} is not a number")


def env_int(name: str, default: int) -> int:
    return int(env_float(name, default))


def die(msg: str, code: int = 1):
    log("FATAL: " + msg)
    raise SystemExit(code)


def cpu_limit() -> int:
    """CPUs this container may use: the cgroup quota if there is one (os.cpu_count() reports the whole host)."""
    n = os.cpu_count() or 4
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if q != "max":
            n = min(n, max(1, int(int(q) / int(per))))
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                n = min(n, max(1, q // per))
        except (OSError, ValueError):
            pass
    try:
        n = min(n, len(os.sched_getaffinity(0)))
    except AttributeError:
        pass
    return n
