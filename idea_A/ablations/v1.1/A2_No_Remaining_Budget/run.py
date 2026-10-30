#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from idea_A.experiments.v11 import run_ablations

def main() -> int:
    run_ablations(only='A2_No_Remaining_Budget')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
