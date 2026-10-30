#!/bin/bash
# Single-process subset of official synthetic-iter.
# Official full command `mpiexec -np 10 python main.py` is BLOCKED (no mpi4py/OpenMPI).
# This injects a rank=0 stub so the original SGD-GD loop in main.py still runs
# as one replica. It is NOT a 10-rank paper reproduction.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/../synthetic-iter"
PY="${ISOLATED_PY:-/Users/grantwright/Research/idea-A-B/.isolated_baselines_venv/bin/python}"
export PYTHONPATH="$HERE/mpi4py_stub:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
mkdir -p "$SRC/output" "$HERE/logs" "$HERE/output"
cd "$SRC"
echo "python=$PY"
echo "PYTHONPATH=$PYTHONPATH"
echo "cwd=$PWD"
echo "disk before:"
df -h /System/Volumes/Data | tail -1
"$PY" -u main.py 2>&1 | tee "$HERE/logs/synthetic_iter_single_replica.log"
echo "disk after:"
df -h /System/Volumes/Data | tail -1
if [ -d "$SRC/output" ]; then
  cp -R "$SRC/output/." "$HERE/output/" || true
fi
if [ -f "$SRC/log.txt" ]; then
  cp "$SRC/log.txt" "$HERE/log.txt"
fi
