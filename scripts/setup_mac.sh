#!/usr/bin/env bash
# Create the Python 3.12 virtualenv for this Mac and install the NumPy stack.
# Torch is attempted afterwards. A torch failure does not remove the NumPy stack.
# This script does not install vLLM, bitsandbytes, flash-attn, CUDA, NCCL,
# transformers, or datasets, and it does not download models.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PY="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: expected Python 3.12 at ${PY}" >&2
  exit 1
fi

if [[ ! -d "${ROOT}/.venv" ]]; then
  "${PY}" -m venv "${ROOT}/.venv"
fi

# shellcheck disable=SC1091
source "${ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn pandas pyyaml matplotlib pytest

mkdir -p "${ROOT}/shared/hardware"
TORCH_LOG="${ROOT}/shared/hardware/torch_install.log"
set +e
python -m pip install torch >"${TORCH_LOG}" 2>&1
TORCH_STATUS=$?
set -e
if [[ "${TORCH_STATUS}" -ne 0 ]]; then
  echo "WARNING: torch install failed with exit ${TORCH_STATUS}. See ${TORCH_LOG}." >&2
  echo "Continuing with the NumPy stack." >&2
else
  echo "torch install succeeded. Log: ${TORCH_LOG}"
fi

python "${ROOT}/scripts/verify_installation.py"
python -m shared.hardware
echo "Environment manifest: ${ROOT}/reports/environment_manifest.json"
