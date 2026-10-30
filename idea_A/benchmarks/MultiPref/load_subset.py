#!/usr/bin/env python3
"""Load the committed MultiPref subset. Does not invent a single ground-truth label."""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_subset(split=None):
    rows = json.loads((HERE / "subset.json").read_text(encoding="utf-8"))
    if split is None:
        return rows
    return [row for row in rows if row.get("split") == split]


def load_ids():
    return json.loads((HERE / "subset_ids.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    data = load_subset()
    print(f"n={len(data)} splits={{k: sum(1 for r in data if r['split']==k) for k in ('train','val','test')}}")
    print("annotation types preserved: normal_worker_annotations, expert_worker_annotations, model_generated_gpt4")
