#!/usr/bin/env bash
# Target of the RunPod start command. Hands-off loop:
#   take the latest code from branch engine-code -> run gpu/run_all.sh -> report to branch engine-results
#   -> on success: final report, stop the pod
#   -> on failure: keep reporting and wait up to 60 min for a fix on engine-code, then rerun (or stop)
# The GPU-hour cap (PINE_MAX_GPU_HOURS, default 8) counts from the pod's first boot, across reruns.
set -uo pipefail
WS=/workspace; CODE="$WS/pine-engine"; BR=engine-code; URL=https://github.com/aboufama/pine
mkdir -p "$WS/logs"
exec >>"$WS/logs/bootstrap.log" 2>&1
say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }
[ -f "$WS/T0" ] || date +%s > "$WS/T0"
export PINE_T0="$(cat "$WS/T0")" PINE_MAX_GPU_HOURS="${PINE_MAX_GPU_HOURS:-8}" PINE_AUTOSTOP_MIN=0 PINE_ROBOT="${PINE_ROBOT:-h2}"
PY="$(command -v python || command -v python3)"

stop_pod() {
  say "stopping pod ${RUNPOD_POD_ID:-?}: $1"
  "$PY" "$CODE/gpu/report_github.py" --note "pod stopping: $1" || true
  if [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null 2>&1 && runpodctl stop pod "$RUNPOD_POD_ID"; then return 0; fi
  if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -fsS -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" >/dev/null && return 0
  fi
  say "could not stop the pod by itself"
}
remote_sha() { git ls-remote "$URL" "refs/heads/$BR" 2>/dev/null | cut -f1; }
update_code() {
  if [ -d "$CODE/.git" ]; then git -C "$CODE" fetch -q --depth 1 origin "$BR" && git -C "$CODE" reset -q --hard FETCH_HEAD
  else rm -rf "$CODE"; git clone -q -b "$BR" --depth 1 "$URL" "$CODE"; fi
}
over_cap() { "$PY" -c "import os,time; raise SystemExit(0 if (time.time()-float(os.environ['PINE_T0']))/3600 >= float(os.environ['PINE_MAX_GPU_HOURS']) else 1)"; }

say "bootstrap: pod ${RUNPOD_POD_ID:-?}, GPU cap ${PINE_MAX_GPU_HOURS} h from $(date -u -d @"$PINE_T0" 2>/dev/null || cat "$WS/T0")"
( while sleep 600; do "$PY" "$CODE/gpu/report_github.py" || true; done ) &
while true; do
  sha="$(remote_sha)"; update_code || say "code update failed; using what is on disk"
  echo "${sha:-unknown}" > "$WS/CODE_SHA"
  if [ -f "$WS/DONE" ] && [ "$(cat "$WS/DONE_SHA" 2>/dev/null)" = "$sha" ]; then
    say "this code already ran ($(head -c 80 "$WS/DONE"))"
  else
    rm -f "$WS/DONE"; "$PY" "$CODE/gpu/report_github.py" --note "starting run_all with code ${sha:0:7}" || true
    bash "$CODE/gpu/run_all.sh"; echo "$sha" > "$WS/DONE_SHA"
  fi
  "$PY" "$CODE/gpu/report_github.py" --final || true
  if grep -q "^PINE DONE" "$WS/DONE" 2>/dev/null; then stop_pod "finished"; sleep infinity; fi
  if grep -q "watchdog" "$WS/DONE" 2>/dev/null || over_cap; then stop_pod "GPU-hour cap reached"; sleep infinity; fi
  say "run did not finish cleanly; waiting up to 60 min for a fix on $BR"
  for _ in $(seq 1 12); do sleep 300; [ "$(remote_sha)" != "$sha" ] && continue 2; over_cap && break; done
  stop_pod "no fix arrived"; sleep infinity
done
