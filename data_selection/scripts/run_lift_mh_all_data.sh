#!/usr/bin/env bash
# First real experiment: RoboMimic Lift MH, image CNN Diffusion Policy, All Data, seed 0.
# Refuses to start unless CUDA is available and the Lift MH image dataset exists.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "This Mac cannot run image Diffusion Policy training." >&2
  exit 2
fi
DATASET="${ROBOMIMIC_LIFT_MH_DATASET:?Set ROBOMIMIC_LIFT_MH_DATASET to the Lift MH image hdf5}"
PY="${DATA_SELECTION_PYTHON:-python3}"
"$PY" tools/preflight.py --require-cuda --require-dataset "$DATASET" --expect-variant mh --expect-task lift
"$PY" scripts/run_unified_baselines.py \
  --method all_data \
  --benchmark robomimic_lift_mh \
  --task lift \
  --seed 0 \
  --config configs/experiments/robomimic_lift_mh_all_data_seed0.yaml \
  --output-dir results/unified/robomimic_lift_mh/all_data/seed_0 \
  --device cuda \
  --retention-ratio 1.0
