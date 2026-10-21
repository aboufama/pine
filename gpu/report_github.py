"""Push the pod's progress and results to branch engine-results of github.com/aboufama/pine.

    python gpu/report_github.py [--final] [--note "text"] [--dry-run]

The branch holds one commit (force-pushed each time): status.json, every finished run record, the registered
prediction, fits, dataset and shard summaries, and the tails of the logs with any token-looking string removed.
Needs GH_TOKEN (a fine-grained token with Contents: write on that repository); it is passed to git through a
credential helper, never on a command line or in a URL, and never printed.
"""
from __future__ import annotations
import argparse, glob, json, os, re, shutil, subprocess, sys, time

WS = os.environ.get("PINE_WORKSPACE", "/workspace")
RUNS = os.environ.get("PINE_RUNS", f"{WS}/runs/gpu"); LOGS = os.environ.get("PINE_LOGS", f"{WS}/logs")
RAW = os.environ.get("PINE_RAW", f"{WS}/data/raw"); SHARDS = os.environ.get("PINE_SHARDS", f"{WS}/data/shards")
REPO = os.environ.get("PINE_RESULTS_REPO", "https://github.com/aboufama/pine")
BRANCH = os.environ.get("PINE_RESULTS_BRANCH", "engine-results")
SECRETS = [os.environ.get(k, "") for k in ("GH_TOKEN", "HF_TOKEN", "RUNPOD_API_KEY")]
PATTERN = re.compile(r"(hf_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|rpa_[A-Za-z0-9]{16,})")


def scrub(text: str) -> str:
    for s in SECRETS:
        if s and len(s) > 8:
            text = text.replace(s, "[secret]")
    return PATTERN.sub("[secret]", text)


def tail(path: str, n: int = 300) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return scrub("".join(f.readlines()[-n:]))
    except OSError:
        return ""


def read(path):
    try:
        return json.load(open(path))
    except Exception:
        return None


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return [dict(zip(["name", "util", "mem_used", "mem_total"], [x.strip() for x in l.split(",")])) for l in out]
    except Exception:
        return []


def free_gb(path):
    try:
        st = os.statvfs(path); return round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        return None


def build(tree: str, final: bool, note: str | None) -> dict:
    os.makedirs(os.path.join(tree, "runs"), exist_ok=True); os.makedirs(os.path.join(tree, "logs"), exist_ok=True)
    t0 = float(open(f"{WS}/T0").read()) if os.path.exists(f"{WS}/T0") else time.time()
    gpus = gpu_info(); n = max(1, len(gpus))
    usd = float(os.environ.get("PINE_USD_PER_HOUR", "0") or 0)
    if not usd and gpus:   # price per GPU-hour from the plan, matched on the GPU's name
        plan = read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plan.json")) or {}
        for g in plan.get("gpus", []):
            key = g.get("name", "").split(" ")[0:2]
            if all(k.lower() in gpus[0]["name"].lower() for k in key) and ("PCIe" in g.get("name", "")) == ("PCIe" in gpus[0]["name"]):
                usd = float(g.get("usd_per_hr", 0)); break
    elapsed = (time.time() - t0) / 3600
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        name = os.path.basename(p)
        shutil.copy(p, os.path.join(tree, "runs", name))
        r = read(p)
        if name.startswith("run_") and isinstance(r, dict) and "val" in r:
            runs.append(dict(size=r.get("size"), fraction=r.get("fraction"), params=r.get("params"), train_hours=r.get("train_hours"),
                             val=r["val"].get("full"), test=r.get("test", {}).get("full"), wall_s=r.get("wall_s")))
    hiphi_done = len(glob.glob(os.path.join(RAW, "hiphi", ".done_*")))
    status = dict(
        updated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), final=final, note=note,
        stage=open(f"{WS}/STATUS").read().strip() if os.path.exists(f"{WS}/STATUS") else "starting",
        done=open(f"{WS}/DONE").read().strip() if os.path.exists(f"{WS}/DONE") else None,
        code=open(f"{WS}/CODE_SHA").read().strip() if os.path.exists(f"{WS}/CODE_SHA") else None,
        pod=os.environ.get("RUNPOD_POD_ID"), gpus=gpus, hours_since_boot=round(elapsed, 3), gpu_hours=round(elapsed * n, 3),
        est_cost_usd=round(elapsed * n * usd, 2) if usd else None, cap_gpu_hours=float(os.environ.get("PINE_MAX_GPU_HOURS", "8")),
        disk_free_gb=free_gb(WS), hiphi_archives_extracted=hiphi_done, runs=runs,
        shards=read(os.path.join(SHARDS, "stats.json")), fetch=read(os.path.join(RAW, "fetch_report.json")))
    for src, dst in ((os.path.join(SHARDS, "stats.json"), "shards_stats.json"), (os.path.join(RAW, "fetch_report.json"), "fetch_report.json")):
        if os.path.exists(src):
            shutil.copy(src, os.path.join(tree, dst))
    for p in glob.glob(os.path.join(LOGS, "*.log")):
        open(os.path.join(tree, "logs", os.path.basename(p)), "w").write(tail(p))
    res = f"{WS}/results.tgz"
    if final and os.path.exists(res) and os.path.getsize(res) < 90e6:
        shutil.copy(res, os.path.join(tree, "results.tgz"))
    json.dump(status, open(os.path.join(tree, "status.json"), "w"), indent=1, default=str)
    open(os.path.join(tree, "README.md"), "w").write(
        "# Pine Engine GPU run\n\nWritten by the RunPod pod every 10 minutes. `status.json` is the live state; `runs/` holds the "
        "run records, the prediction registered before the largest run, and the fits; `logs/` holds log tails.\n")
    return status


def push(tree: str) -> bool:
    tok = os.environ.get("GH_TOKEN", "")
    if not tok:
        print("GH_TOKEN not set; not pushing"); return False
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", PINE_GH=tok)
    helper = "!f() { echo username=pine-engine-pod; echo password=$PINE_GH; }; f"
    g = lambda *a: subprocess.run(["git", "-C", tree, "-c", "user.name=Pine Engine pod", "-c", "user.email=pod@pine.invalid",
                                   "-c", f"credential.helper={helper}", *a], env=env, capture_output=True, text=True, timeout=600)
    shutil.rmtree(os.path.join(tree, ".git"), ignore_errors=True)
    for args in (["init", "-q"], ["checkout", "-q", "--orphan", BRANCH], ["add", "-A"], ["commit", "-q", "-m", "pod report"]):
        r = g(*args)
        if r.returncode != 0:
            print("git", args[0], "failed:", scrub(r.stderr)[:300]); return False
    r = g("push", "-q", "-f", REPO + ".git", f"HEAD:refs/heads/{BRANCH}")
    if r.returncode != 0:
        print("push failed:", scrub(r.stderr)[:400]); return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true"); ap.add_argument("--note"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tree = os.path.join(WS, ".report"); shutil.rmtree(tree, ignore_errors=True); os.makedirs(tree)
    st = build(tree, a.final, a.note)
    if a.dry_run:
        print(json.dumps({k: st[k] for k in ("stage", "gpu_hours", "runs", "hiphi_archives_extracted")}, default=str)); return 0
    ok = push(tree)
    print(f"report {'pushed' if ok else 'NOT pushed'}: stage={st['stage'][:60]!r} runs={len(st['runs'])}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
