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
        MethodSpec("a3_observed_only", "none", "observed_only", uses_budget=False, notes="A3: no audit."),
        MethodSpec("a3_naive_ips", "none", "naive_ips", uses_budget=False, notes="A3: naive IPS on natural S."),
    ]
    run_ablation("A3_No_Anchor_Intervention", HERE, specs, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
