#!/usr/bin/env bash
# Does not download HDF5. Prints the upstream command and exits until the checkout exists.
set -euo pipefail
cd "$(dirname "$0")/.."
LIBERO_DIR="benchmarks/_sources/libero"
if [[ ! -d "$LIBERO_DIR/.git" ]]; then
  echo "LIBERO checkout is not available. Fetch it before running the upstream downloader." >&2
  exit 2
fi
if [[ ! -f "$LIBERO_DIR/benchmark_scripts/download_libero_datasets.py" ]]; then
  echo "benchmark_scripts/download_libero_datasets.py was not found in the checkout. Do not guess another path." >&2
  exit 2
fi
echo "Upstream downloader is present. This wrapper does not start it."
echo "On the server, from $LIBERO_DIR:"
echo "  python benchmark_scripts/download_libero_datasets.py --datasets libero_10"
echo "  HF_ENDPOINT=https://hf-mirror.com python benchmark_scripts/download_libero_datasets.py --use-huggingface --datasets libero_10"
exit 0
