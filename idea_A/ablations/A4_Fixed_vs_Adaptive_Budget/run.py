#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml

from idea_A.experiments.ablation_support import run_ablation
from idea_A.experiments.protocol import MethodSpec

HERE = Path(__file__).resolve().parent


def main() -> int:
    config = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    specs = [
        MethodSpec("a4_neyman_oneshot", "neyman", "ipw", notes="A4: one-shot Neyman of the full budget."),
        MethodSpec("a4_neyman_batched", "neyman_batched", "ipw", notes="A4: batched remaining-budget Neyman."),
    ]
    run_ablation("A4_Fixed_vs_Adaptive_Budget", HERE, specs, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
