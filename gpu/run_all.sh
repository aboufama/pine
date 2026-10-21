#!/usr/bin/env bash
# Pod entrypoint for the Pine Engine GPU scale-up. Started by runpod/launch.py under nohup from /workspace/pine-engine.
#
#   bash gpu/run_all.sh                      the whole pipeline
#   bash gpu/run_all.sh --pack-only <tgz>    only pack what exists so far (launch.py fetch --partial)
#
# Steps: pip install what is missing -> gpu/parity.py (abort on failure: nothing trains) -> gpu/fetch_data.py ->
# gpu/build_shards.py -> gpu/sweep_gpu.py -> /workspace/results.tgz -> one-line summary.
# Everything is logged to /workspace/logs/. Markers: /workspace/RUNNING while it runs, /workspace/STATUS (current
# step), /workspace/DONE when finished (with the summary). Cost guard: PINE_MAX_GPU_HOURS is enforced by the sweep;
# a watchdog here also packs and stops the pod if the cap is passed by more than 30 min, and the pod stops itself
# PINE_AUTOSTOP_MIN minutes after finishing (sooner once `launch.py fetch` has copied the results).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="${PINE_WORKSPACE:-/workspace}"
export PINE_WORKSPACE="$WS"
export PINE_RAW="${PINE_RAW:-$WS/data/raw}"
export PINE_SHARDS="${PINE_SHARDS:-$WS/data/shards}"
export PINE_RUNS="${PINE_RUNS:-$WS/runs/gpu}"
export PINE_LOGS="${PINE_LOGS:-$WS/logs}"
export PINE_ROBOT="${PINE_ROBOT:-h2}"
export PINE_MAX_GPU_HOURS="${PINE_MAX_GPU_HOURS:-8}"
export PINE_AUTOSTOP_MIN="${PINE_AUTOSTOP_MIN:-120}"
export PINE_T0="${PINE_T0:-$(date +%s)}"
export PYTHONUNBUFFERED=1
mkdir -p "$PINE_LOGS" "$PINE_RUNS" "$PINE_RAW" "$PINE_SHARDS"
RESULTS="$WS/results.tgz"
PY="$(command -v python || command -v python3)"

# Phase 2 (gpu/phase2.py) runs when gpu/PHASE says 2 and phase 1 has finished on this volume (fits.json and
# results.tgz exist). It gets what phase 1 left of the GPU-hour cap (the cap minus the time from first boot to the end
# of phase 1), and at least PINE_PHASE2_MIN_HOURS (default 4) so the SEED sweep and the flagship fit.
PHASE=1
if [ "$(cat "$REPO/gpu/PHASE" 2>/dev/null)" = "2" ] && [ -f "$PINE_RUNS/fits.json" ] && [ -f "$RESULTS" ] && [ -f "$WS/T0" ]; then
  PHASE=2
  left_h="$("$PY" -c "import os; t0=float(open('$WS/T0').read()); end1=os.path.getmtime('$RESULTS'); print(max(float(os.environ.get('PINE_PHASE2_MIN_HOURS', 4)), float(os.environ['PINE_MAX_GPU_HOURS']) - (end1 - t0) / 3600))")"
  export PINE_T0="$(date +%s)" PINE_MAX_GPU_HOURS="$left_h"
  export PINE_PHASE2_DEADLINE="$("$PY" -c "import time; print(time.time() + max(0.0, $left_h * 3600 - 15 * 60))")"
fi

pack() {   # pack <out.tgz>: run records, weights, fits, logs, shard stats, fetch report, plan
  local out="$1" tmp
  tmp="$(mktemp -d)"
  cp -a "$PINE_RUNS"/. "$tmp"/ 2>/dev/null || true
  mkdir -p "$tmp/logs" "$tmp/data"
  cp -a "$PINE_LOGS"/. "$tmp/logs"/ 2>/dev/null || true
  for f in "$PINE_SHARDS/stats.json" "$PINE_RAW/fetch_report.json" "$REPO/gpu/plan.json"; do
    [ -f "$f" ] && cp "$f" "$tmp/data/" 2>/dev/null
  done
  [ -f "$PINE_SHARDS/clips.json" ] && gzip -c "$PINE_SHARDS/clips.json" > "$tmp/data/clips.json.gz" 2>/dev/null
  tar -czf "$out.tmp" -C "$tmp" . && mv "$out.tmp" "$out"
  rm -rf "$tmp"
  echo "packed $(du -h "$out" | cut -f1) -> $out"
}

if [ "${1:-}" = "--pack-only" ]; then
  pack "${2:-$WS/results_partial.tgz}"
  exit 0
fi

LOG="$PINE_LOGS/run_all.log"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
status() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" > "$WS/STATUS"; say "== $*"; }

stop_pod() {   # stop (not terminate) this pod: GPU billing ends, /workspace survives
  say "stopping pod ${RUNPOD_POD_ID:-?} ($1)"
  if [ -z "${RUNPOD_POD_ID:-}" ]; then say "RUNPOD_POD_ID not set; cannot stop myself"; return 1; fi
  if command -v runpodctl >/dev/null 2>&1 && runpodctl stop pod "$RUNPOD_POD_ID"; then return 0; fi
  if [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
      "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" >/dev/null && return 0
  fi
  say "could not stop the pod; stop it with: python3 runpod/launch.py down --yes"
  return 1
}

if [ -e "$WS/RUNNING" ] && kill -0 "$(cat "$WS/RUNNING" 2>/dev/null)" 2>/dev/null; then
  say "run_all.sh is already running (pid $(cat "$WS/RUNNING")); exiting"; exit 0
fi
echo $$ > "$WS/RUNNING"
rm -f "$WS/DONE" "$WS/FETCHED"
trap 'rm -f "$WS/RUNNING"' EXIT

# ---------------------------------------------------------------------- watchdog: hard cap + auto-stop after finishing
(
  cap_s=$("$PY" -c "import os; g=max(1,int(os.environ.get('PINE_GPUS') or os.environ.get('RUNPOD_GPU_COUNT') or 1)); print(int((float(os.environ['PINE_MAX_GPU_HOURS'])/g+0.5)*3600))" 2>/dev/null || echo 30600)
  while sleep 60; do
    now=$(date +%s)
    if [ -e "$WS/DONE" ]; then
      done_t=$(stat -c %Y "$WS/DONE")
      if [ -e "$WS/FETCHED" ]; then sleep 120; stop_pod "results fetched"; exit 0; fi
      if [ "${PINE_AUTOSTOP_MIN}" -gt 0 ] && [ $((now - done_t)) -ge $((PINE_AUTOSTOP_MIN * 60)) ]; then
        stop_pod "finished ${PINE_AUTOSTOP_MIN} min ago and nobody fetched the results"; exit 0
      fi
    elif [ $((now - PINE_T0)) -ge "$cap_s" ]; then
      say "WATCHDOG: past PINE_MAX_GPU_HOURS=${PINE_MAX_GPU_HOURS} by 30 min; packing and stopping"
      pkill -f gpu/train_torch.py || true; pkill -f gpu/sweep_gpu.py || true; pkill -f gpu/build_shards.py || true
      sleep 20
      pack "$RESULTS"; echo "watchdog stop at $(date -u)" > "$WS/DONE"
      [ "${PINE_AUTOSTOP_MIN}" -gt 0 ] && stop_pod "budget cap"
      exit 0
    fi
  done
) &
WATCHDOG=$!

finish() {   # finish <exit code> <message>
  local rc="$1" msg="$2"
  status "packing results"
  pack "$RESULTS"
  local summary
  summary="$("$PY" - "$PINE_RUNS" "$PINE_SHARDS/stats.json" "$PINE_T0" "$msg" <<'EOF' 2>/dev/null
import glob, json, os, sys, time
runs, stats_p, t0, msg = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(runs, "run_*.json")))]
st = json.load(open(stats_p)) if os.path.exists(stats_p) else {}
best = min(recs, key=lambda r: r["val"]["full"]) if recs else None
pred = os.path.join(runs, "prediction.json")
p = json.load(open(pred)) if os.path.exists(pred) else None
obs = next((r for r in recs if p and r["size"] == p["size"] and r["fraction"] == 1.0), None)
parts = [f"PINE {msg}", f"robot {st.get('robot', os.environ.get('PINE_ROBOT'))}", f"{st.get('hours', 0):.1f} h data",
         f"{len(recs)} runs"]
if best:
    parts.append(f"best {best['size']} val {best['val']['full']:.4f} test {best['test']['full']:.4f} hidden {best['val']['hidden_joint_rmse_deg']:.1f} deg")
if p:
    parts.append(f"predicted {p['size']} {p['loss']:.4f} [{p['lo']:.4f},{p['hi']:.4f}]" + (f" observed {obs['val']['full']:.4f}" if obs else " (not run)"))
parts.append(f"{(time.time() - t0) / 3600:.2f} h wall")
print(" | ".join(parts))
EOF
)"
  [ -n "$summary" ] || summary="PINE $msg (no summary available)"
  echo "$summary" > "$WS/DONE"
  status "done: $msg"
  echo "$summary"
  exit "$rc"
}

say "Pine Engine GPU run (phase $PHASE): robot $PINE_ROBOT, cap $PINE_MAX_GPU_HOURS GPU-h, repo $REPO, python $PY"
cd "$REPO"

# ---------------------------------------------------------------------- 1. environment
status "environment"
nvidia-smi || say "WARNING: nvidia-smi failed"
cuda_diag() {   # what CUDA sees: device nodes, driver, libraries, and the raw driver-API result
  say "--- CUDA diagnostics"
  ls -la /dev/nvidia* 2>&1 | head -20
  cat /proc/driver/nvidia/version 2>&1 | head -2
  env | grep -E "^(CUDA|NVIDIA|LD_LIBRARY_PATH|PATH)=" || true
  ldconfig -p 2>/dev/null | grep -E "libcuda\.so|libnvidia-ml" | head -6
  "$PY" - <<'EOF' 2>&1 | head -5
import ctypes
try:
    lib = ctypes.CDLL("libcuda.so.1"); print("libcuda", lib._name, "cuInit ->", lib.cuInit(0))
    v = ctypes.c_int(); lib.cuDriverGetVersion(ctypes.byref(v)); print("driver CUDA version", v.value)
except OSError as e:
    print("libcuda not loadable:", e)
EOF
  say "--- end diagnostics"
}
cuda_diag
cuda_ok=0
for attempt in $(seq 1 12); do   # some hosts report "CUDA unknown error" for the first minutes after boot
  if [ "$attempt" = 2 ] && command -v nvidia-modprobe >/dev/null; then say "trying nvidia-modprobe -u -c=0"; nvidia-modprobe -u -c=0 || true; fi
  if [ "$attempt" = 3 ] && echo "${LD_LIBRARY_PATH:-}" | grep -q compat; then
    say "dropping CUDA forward-compat libraries from LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v compat | paste -sd: -)"
  fi
  if [ "$attempt" = 4 ]; then
    for d in /usr/local/cuda/compat /usr/local/cuda-*/compat; do [ -d "$d" ] && say "hiding $d" && mv "$d" "$d.off" 2>/dev/null; done
    ldconfig 2>/dev/null || true
  fi
  if "$PY" - <<'EOF'
import sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "devices", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA is not available"
torch.ones(8, device="cuda").sum().item()
print("gpu", torch.cuda.get_device_name(0), "bf16", torch.cuda.is_bf16_supported())
EOF
  then cuda_ok=1; break; fi
  say "CUDA not usable yet (attempt $attempt of 12); retrying in 20 s"; sleep 20
done
if [ "$cuda_ok" != 1 ]; then
  cuda_diag; say "FATAL: torch with CUDA is not usable on this host"; echo "$(date -u) CUDA unusable" > "$WS/BAD_HOST"
  finish 2 "aborted: no torch/CUDA (bad host)"
fi
need=""
"$PY" -c "import huggingface_hub" 2>/dev/null || need="$need huggingface_hub"
"$PY" -c "import scipy" 2>/dev/null || need="$need scipy"
"$PY" -c "import numpy" 2>/dev/null || need="$need numpy"
"$PY" -c "import hf_transfer" 2>/dev/null || need="$need hf_transfer"
"$PY" -c "import zstandard" 2>/dev/null || command -v zstd >/dev/null || need="$need zstandard"
if [ -n "$need" ]; then
  say "pip install$need"
  "$PY" -m pip install -q $need || say "WARNING: pip install failed for:$need"
fi
"$PY" -c "import hf_transfer" 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0
command -v git-lfs >/dev/null || git lfs version >/dev/null 2>&1 || \
  (apt-get update -qq && apt-get install -y -qq git-lfs) >/dev/null 2>&1 || say "WARNING: git-lfs unavailable (LAFAN1 will be skipped)"
"$PY" -c "import scipy, numpy" || { say "FATAL: numpy/scipy missing"; finish 2 "aborted: python packages"; }

# ---------------------------------------------------------------------- 2. parity gate
status "parity (torch model vs NumPy reference)"
if ! "$PY" gpu/parity.py 2>&1 | tee "$PINE_LOGS/parity.log"; then
  say "FATAL: parity failed; nothing will be trained (see $PINE_LOGS/parity.log)"
  finish 1 "aborted: parity failed"
fi
grep -q "PARITY OK" "$PINE_LOGS/parity.log" || finish 1 "aborted: parity did not report OK"

# ---------------------------------------------------------------------- phase 2 (instead of steps 3 and 4)
if [ "$PHASE" = 2 ]; then
  if "$PY" -c "import sys; sys.exit(0 if float('$PINE_MAX_GPU_HOURS') >= 0.5 else 1)"; then
    status "phase 2: people vs hours on held-out HiPHI performers (${PINE_MAX_GPU_HOURS} GPU-h left)"
    "$PY" gpu/phase2.py || say "WARNING: phase2.py exited with an error"
  else
    say "phase 2 skipped: only ${PINE_MAX_GPU_HOURS} GPU-h left of the cap"
  fi
  finish 0 "DONE (phase 2)"
fi

# ---------------------------------------------------------------------- 3. data
status "fetching datasets"
"$PY" gpu/fetch_data.py || say "WARNING: fetch_data.py exited with $? (continuing with whatever was fetched)"

status "building shards"
if ! "$PY" gpu/build_shards.py; then
  say "FATAL: build_shards.py failed"
  finish 1 "aborted: no shards"
fi

# ---------------------------------------------------------------------- 4. sweep
status "sweep"
"$PY" gpu/sweep_gpu.py --dry-run || true
if ! "$PY" gpu/sweep_gpu.py; then
  say "WARNING: sweep_gpu.py exited with an error"
  finish 1 "sweep ended with an error"
fi
kill "$WATCHDOG" 2>/dev/null; wait "$WATCHDOG" 2>/dev/null
# keep a watchdog for the auto-stop after finishing
(
  while sleep 60; do
    [ -e "$WS/DONE" ] || continue
    if [ -e "$WS/FETCHED" ]; then sleep 120; stop_pod "results fetched"; exit 0; fi
    if [ "${PINE_AUTOSTOP_MIN}" -gt 0 ] && [ $(( $(date +%s) - $(stat -c %Y "$WS/DONE") )) -ge $((PINE_AUTOSTOP_MIN * 60)) ]; then
      stop_pod "finished ${PINE_AUTOSTOP_MIN} min ago"; exit 0
    fi
  done
) </dev/null >>"$PINE_LOGS/autostop.log" 2>&1 &
disown || true
finish 0 "DONE"
