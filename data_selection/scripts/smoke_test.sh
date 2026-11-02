#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${DATA_SELECTION_PYTHON:-../.venv/bin/python}"
"$PY" -m pytest
