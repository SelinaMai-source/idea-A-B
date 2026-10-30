#!/usr/bin/env python3
"""CLI for Idea A v1.1 experiments."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from idea_A.experiments.v11 import main

if __name__ == "__main__":
    raise SystemExit(main())
