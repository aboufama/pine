#!/usr/bin/env bash
# Run research variants one after another (never two at once) at low priority; the watchdog pauses the
# job whenever anything else needs the CPU and resumes it when the machine is quiet again.
set -u
cd /home/claude/pine-engine
export PINE_ROBOT=h2 PYTHONDONTWRITEBYTECODE=1
for V in "$@"; do
  nice -n 19 python3 -B research/history_corruption.py --variant "$V" > "runs/research/$V.log" 2>&1 &
  PID=$!
  python3 -B research/watchdog.py "$PID" >> runs/research/watchdog.log 2>&1 &
  WD=$!
  wait "$PID"; RC=$?
  echo "exit $RC" >> "runs/research/$V.log"
  wait "$WD"
done
