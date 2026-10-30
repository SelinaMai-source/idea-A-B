#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml

from idea_A.experiments.ablation_support import run_ablation
from idea_A.experiments.protocol import MethodSpec, misspecify_q

HERE = Path(__file__).resolve().parent


def main() -> int:
    config = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
    mode = str(config.get("q_misspecification", "mix_uniform"))
    specs = [
        MethodSpec(
            "a5_neyman_ipw_wrong_q",
            "neyman",
            "ipw",
            estimator_q_transform=lambda q, m=mode: misspecify_q(q, m),
            notes="A5: true Neyman draws, IPW with misspecified q.",
        ),
        MethodSpec("neyman_ipw_reference", "neyman", "ipw", notes="Reference: IPW with the true q."),
    ]
    run_ablation("A5_Misspecified_Propensity", HERE, specs, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
