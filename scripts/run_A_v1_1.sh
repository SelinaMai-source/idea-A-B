#!/usr/bin/env bash
# Run Idea A v1.1 Tier 1, then Tier 2, then one-factor ablations.
# NumPy only. Do not load Qwen. Do not start a second memory-heavy job.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  echo "ERROR: expected ${PY}" >&2
  exit 1
fi
"${PY}" "${ROOT}/idea_A/experiments/v1.1/run.py" --tier 1 "$@"
"${PY}" "${ROOT}/idea_A/experiments/v1.1/run.py" --tier 2 "$@"
"${PY}" "${ROOT}/idea_A/experiments/v1.1/run.py" --tier ablations "$@"
