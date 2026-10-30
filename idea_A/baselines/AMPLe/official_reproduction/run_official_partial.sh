#!/bin/bash
# Smallest faithful official AMPLe run that can load public data.
# Acquisition stays volume. burnin stays 50000. ONLY n_rounds is shortened.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ISOLATED_PY:-/Users/grantwright/Research/idea-A-B/.isolated_baselines_venv/bin/python}"
OUT="$(cd "$(dirname "$0")" && pwd)"
export WANDB_MODE=offline
export WANDB_DIR="$OUT/wandb"
export WANDB_PROJECT=ACL25
export HF_HOME="$OUT/.cache/huggingface"
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR" "$HF_HOME" "$OUT/logs"
cd "$ROOT"
echo "python=$PY"
echo "cwd=$PWD"
echo "disk before:"
df -h /System/Volumes/Data | tail -1
# README example uses --dims 0 1, which rewrites rm_names and then requests
# minhyeonoh/feedback_assistant_harmlesshelpful_271 (HTTP 401 / not public).
# Omitting --dims loads the public 3-aspect dataset
# minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271 (~4.5MB).
# Official acquisition=volume and burnin=50000 are unchanged.
"$PY" -u main.py \
  --seed=1 \
  --task=assistant \
  --rm_names harmless helpful humor \
  --n_rounds=3 \
  --true_preference 0.2 0.7 0.1 \
  --true_beta=10 \
  --beta=2.0 \
  --jitter=0.01 \
  --alpha=10 \
  --init=previous \
  --acquisition=volume \
  --burnin=50000 \
  --gamma=0.0 \
  --mode=likelihood \
  --margin=none \
  --inputs=0 \
  2>&1 | tee "$OUT/logs/main_nrounds3_volume.log"
echo "disk after:"
df -h /System/Volumes/Data | tail -1
