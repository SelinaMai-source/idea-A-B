#!/usr/bin/env bash
# Linux GPU preflight. Does not install on Darwin and does not start training.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "setup_gpu.sh refuses to install or train on Darwin." >&2
  exit 2
fi
PY="${DATA_SELECTION_PYTHON:-python3}"
bash scripts/setup_environment.sh --env unified_dp
"$PY" tools/inspect_environment.py --require-cuda
"$PY" tools/preflight.py --require-cuda
echo "Preflight did not download data and did not start training."
