#!/usr/bin/env bash
# CPU rehearsal of every pod stage after the download, on the exact engine-code, with small data:
#   fetch (unitree vendored, Bandai, LAFAN1, CMU via the git/tarball path) -> HiPHI-format keypoint caches (fake performers,
#   built with fetch_data.canon_cache exactly as on the pod) -> torch IK check -> build_shards (torch IK on CPU)
#   -> sweep_gpu (phase 1) -> phase-2 switch condition -> phase2.py (people sweeps, recipes, flagship)
# Only CPU adaptations are applied: trainer default device cpu, IK device cpu, tiny model sizes.
set -uo pipefail
ROOT="$PWD"
git clone -q -b engine-code --depth 1 https://github.com/aboufama/pine code || exit 1
cd code; echo "engine-code $(git log --oneline -1)"
WS="$ROOT/ws"
export PINE_WORKSPACE="$WS" PINE_RAW="$WS/data/raw" PINE_SHARDS="$WS/data/shards" PINE_RUNS="$WS/runs/gpu" \
       PINE_LOGS="$WS/logs" PINE_ROBOT=h2 PYTHONUNBUFFERED=1 PINE_IK_DEVICE=cpu
mkdir -p "$PINE_RAW" "$PINE_SHARDS" "$PINE_RUNS" "$PINE_LOGS"
python - <<'PYEOF'
import re
def sub(p, a, b, n=1):
    s = open(p).read(); assert s.count(a) >= 1, (p, a); s = s.replace(a, b) if n == 0 else s.replace(a, b, n); open(p, "w").write(s)
sub("gpu/train_torch.py", 'ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")',
    'ap.add_argument("--device", choices=("cuda", "cpu"), default="cpu")')
sub("gpu/build_shards.py", 'if not torch.cuda.is_available():', 'if os.environ.get("PINE_IK_DEVICE") != "cpu" and not torch.cuda.is_available():')
sub("gpu/build_shards.py", 'RT.make_ops("torch", "cuda")', 'RT.make_ops("torch", os.environ.get("PINE_IK_DEVICE", "cuda"))')
s = open("pine/train_np.py").read()
s = re.sub(r'"M": \(96, 3, 4\), "L": \(128, 4, 4\), "XL": \(192, 5, 6\)', '"M": (16, 1, 2), "L": (24, 1, 2), "XL": (32, 1, 2)', s)
s = re.sub(r'"2XL": \(256, 6, 8\), "3XL": \(384, 8, 8\), "4XL": \(512, 10, 8\), "5XL": \(768, 12, 12\)',
           '"2XL": (32, 2, 2), "3XL": (48, 2, 2), "4XL": (64, 2, 4), "5XL": (96, 2, 4)', s)
assert '"5XL": (96, 2, 4)' in s and '"XL": (32, 1, 2)' in s; open("pine/train_np.py", "w").write(s)
print("CPU adaptations applied")
PYEOF
[ $? = 0 ] || exit 1
FAILED=""
stage() { local n="$1"; shift; echo; echo "=== STAGE $n"; local t=$(date +%s)
  if "$@"; then echo "=== STAGE $n OK ($(( $(date +%s) - t ))s)"; else echo "=== STAGE $n FAILED (exit $?, $(( $(date +%s) - t ))s)"; FAILED="$FAILED $n"; fi; }

stage fetch python gpu/fetch_data.py --only unitree_mujoco,bandai,lafan1,cmu

fake_hiphi() { python - <<'PYEOF'
# HiPHI layout (per the dataset card): HiPHI/data/{frame}/{lu}/{motion_id}/motion_actor.bvh, mirrors as {motion_id}__mirror,
# metadata/hiphi_metadata.csv with motion_id, actor_id, text_annotation. LAFAN1 BVH (Mixamo-style) stands in; every take
# appears twice under different motion ids so there are 50 performers. Two archives, cached with fetch_data.canon_cache.
import csv, glob, os, shutil, sys
sys.path.insert(0, "gpu")
import fetch_data as FD
RAW = os.environ["PINE_RAW"]
src = sorted(glob.glob(os.path.join(RAW, "lafan1", "repo", "lafan1", "bvh", "*.bvh")))
assert src, "no LAFAN1 BVH"
rows, parts = [], {1: [], 2: []}
for k in range(2):
    for i, f in enumerate(src):
        j = k * len(src) + i
        mid = f"Body_movement-fake{j // 10:02d}_{j % 10 + 1:04d}"
        part = 1 + (j % 2)
        for m in (mid, mid + "__mirror"):
            d = os.path.join(RAW, "hiphi", f"x{part}", "HiPHI", "data", "Body", f"lu{j // 10}", m)
            os.makedirs(d, exist_ok=True); shutil.copy(f, os.path.join(d, "motion_actor.bvh"))
        rows.append(dict(motion_id=mid, actor_id=f"A{j % 50:03d}", text_annotation=os.path.basename(f)))
os.makedirs(os.path.join(RAW, "hiphi", "metadata"), exist_ok=True)
with open(os.path.join(RAW, "hiphi", "metadata", "hiphi_metadata.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, ["motion_id", "frame", "lu", "frame_lu", "duration_sec", "frame_count", "actor_id",
                            "text_annotation", "is_hoi", "object_categories", "mirrored"])
    w.writeheader()
    for r in rows:
        w.writerow(dict(r, frame="Body", mirrored=False))
for part in (1, 2):
    n = FD.canon_cache(os.path.join(RAW, "hiphi", f"x{part}"),
                       os.path.join(RAW, "hiphi", "canon", f"HiPHI_data_part_000{part}_of_0032.npz"))
    shutil.rmtree(os.path.join(RAW, "hiphi", f"x{part}"))
    print(f"fake HiPHI archive {part}: {n} takes kept")
import json, numpy as np
reps = []
for p in sorted(glob.glob(os.path.join(RAW, "hiphi", "canon", "*.npz"))):
    reps += json.loads(str(np.load(p)["reps"]))
subj = {r["subject"] for r in reps}
print(f"{len(reps)} takes, {len(subj)} performers, unknown: {sum(1 for r in reps if r['subject'].endswith('unknown'))}, "
      f"mirrors: {sum(1 for r in reps if 'mirror' in r['clip'])}, rejects: {[r.get('reject') for r in reps if 'reject' in r][:3]}")
assert len(subj) >= 40 and not any(r["subject"].endswith("unknown") for r in reps)
PYEOF
}
stage fake_hiphi fake_hiphi

ik_check() { python - <<'PYEOF'
import glob, json, os, sys, time
import numpy as np
sys.path.insert(0, "gpu")
import common  # noqa
import retarget_torch as RT
from pine.g1 import make
p = sorted(glob.glob(os.path.join(os.environ["PINE_RAW"], "hiphi", "canon", "*.npz")))[0]
z = np.load(p); off = z["offsets"]
takes = [z["kp"][off[i]:off[i + 1]].astype(np.float64) for i in range(min(8, len(off) - 1))]
robot = make("h2")
ik = RT.BatchedIK(robot, RT.make_ops("torch", "cpu"))
t = time.time(); res = RT.check_against_numpy(robot, takes, ik, frames=2000, limit_mm=1.0)
print("check_against_numpy:", json.dumps(res)[:600], f"({time.time() - t:.0f}s)")
t = time.time(); out = ik.retarget_many(takes[:4]); n = sum(len(x) for x in takes[:4])
print(f"torch-cpu IK: {n} frames in {time.time() - t:.1f}s ({(time.time() - t) / n * 1e3:.3f} ms/frame)")
assert res.get("passed"), "torch IK does not match pine/g1.py"
PYEOF
}
stage ik_check ik_check

stage build_shards python gpu/build_shards.py --max-takes 60
python -c "import json; s=json.load(open('$PINE_SHARDS/stats.json')); print('shards:', round(s['hours'],2), 'h', s['per_source'], 'ik', s['ik']['backend'], s['ik']['mean_err_cm'])" || true

export PINE_STEPS=40 PINE_BATCH=64 PINE_VAL_WINDOWS=256 PINE_T0=$(date +%s) PINE_MAX_GPU_HOURS=8
stage sweep_dry python gpu/sweep_gpu.py --dry-run
stage sweep python gpu/sweep_gpu.py
ls "$PINE_RUNS"

phase_switch() {   # the run_all.sh condition for phase 2 (results.tgz/T0 stand-ins), then its budget arithmetic
  touch "$WS/results.tgz"; date +%s > "$WS/T0"
  [ "$(cat gpu/PHASE)" = "2" ] && [ -f "$PINE_RUNS/fits.json" ] && [ -f "$WS/results.tgz" ] && [ -f "$WS/T0" ]
}
stage phase_switch phase_switch

export PINE_PHASE2_STEPS=30 PINE_SKIP_SEED=1 PINE_PHASE2_DEADLINE=$(python -c "import time; print(time.time() + 55 * 60)")
stage phase2 python gpu/phase2.py
python - <<'PYEOF' || true
import json, os
p = os.path.join(os.environ["PINE_RUNS"], "phase2", "summary.json")
s = json.load(open(p)); print("phase2 rows:", len(s["rows"]))
for r in s["rows"]:
    print(" ", r["name"], r["size"], r["params"], "new people", round(r["new_people"], 4), "steps", r["steps"], "best", r["best_step"])
PYEOF
ls -la "$PINE_RUNS/phase2" | head -40
echo; echo "REHEARSAL FAILED STAGES:${FAILED:- none}"
[ -z "$FAILED" ]
