#!/usr/bin/env bash
# Usage: download_robomimic.sh lowdim|image
# The image zip is refused on Darwin.
set -euo pipefail
cd "$(dirname "$0")/.."
KIND="${1:-}"
PY="${DATA_SELECTION_PYTHON:-../.venv/bin/python}"
DEST_ROOT="${DATA_ROOT:-$PWD/artifacts/downloads}"
mkdir -p "$DEST_ROOT"
if [[ "$KIND" == "image" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "refusing robomimic_image on Darwin" >&2
    exit 2
  fi
  "$PY" tools/download_assets.py --asset robomimic_image --allow-large --dest "$DEST_ROOT/robomimic_image.zip"
elif [[ "$KIND" == "lowdim" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    echo "refusing robomimic_lowdim on Darwin; content-length is about 1.8GiB and was not requested" >&2
    exit 2
  fi
  "$PY" tools/download_assets.py --asset robomimic_lowdim --dest "$DEST_ROOT/robomimic_lowdim.zip"
else
  echo "usage: $0 lowdim|image" >&2
  exit 2
fi
