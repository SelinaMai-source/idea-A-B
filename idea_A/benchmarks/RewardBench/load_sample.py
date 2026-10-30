#!/usr/bin/env python3
"""Load the committed RewardBench schema sample. Does not score a leaderboard."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_sample():
    return json.loads((HERE / "schema_sample.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    rows = load_sample()
    print(f"n={len(rows)} fields={sorted(rows[0].keys()) if rows else []}")
    print("This sample is a schema check, not a RewardBench leaderboard score.")
