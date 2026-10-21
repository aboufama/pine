#!/usr/bin/env bash
# Publish demo/ to the live Pine site as /engine/ (GitHub Pages serves aboufama/pine from main).
set -euo pipefail
SRC=/home/claude/pine-engine/site
SITE=/home/claude/pine
mkdir -p "$SITE/engine"
cp "$SRC/index.html" "$SITE/engine/index.html"
cp /home/claude/pine-engine/demo/data.json "$SITE/engine/data.json"
cd "$SITE"
git fetch -q origin main && git rebase -q origin/main
git add engine/index.html engine/data.json
if git diff --cached --quiet; then echo "nothing to publish"; exit 0; fi
git commit -q -m "Engine demo: public mocap retargeted to a Unitree H2 and pretrained, with scaling results

Adds /engine/ only; the homepage is unchanged.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01D9Js3UWSYnEaaPu3hkLW22"
git push -q origin HEAD:main
git log --oneline -1
