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
        MethodSpec(
            "a2_neyman_unweighted",
            "neyman",
            "observed_only",
            notes="A2: Neyman allocation, unweighted audited mean (no A/q).",
        ),
        MethodSpec("neyman_ipw_reference", "neyman", "ipw", notes="Reference: same allocation with IPW."),
    ]
    run_ablation("A2_No_Propensity_Correction", HERE, specs, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
