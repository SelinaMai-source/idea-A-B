#!/usr/bin/env bash
# Records the GPU check. Does not install packages and does not run on Darwin.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "setup_gpu.sh refuses to install or train on Darwin." >&2
  exit 2
fi
PY="${DATA_SELECTION_PYTHON:-python3}"
"$PY" tools/inspect_environment.py --require-cuda
"$PY" tools/preflight.py --require-cuda
echo "Preflight did not download data and did not start training."
