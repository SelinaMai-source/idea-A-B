#!/usr/bin/env python3
"""Replace adaptive exploration with uniform allocation inside the controller."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from idea_B.experiments.run_ablation import BUDGET, SEEDS, run_ablation_jobs

RESULTS = Path(__file__).resolve().parent / "results"


def main() -> int:
    jobs = []
    for seed in SEEDS:
        jobs.append(
            {
                "env": "B2",
                "method": "ObservabilityController",
                "seed": seed,
                "budget": BUDGET,
                "prefix": "ablationB2_uniform",
                "arm": "uniform_inside_controller",
                "method_overrides": {"mix_uniform": 1.0},
            }
        )
        jobs.append(
            {
                "env": "B2",
                "method": "ObservabilityController",
                "seed": seed,
                "budget": BUDGET,
                "prefix": "ablationB2_adaptive",
                "arm": "adaptive_controller",
                "method_overrides": {"mix_uniform": 0.1},
            }
        )
    run_ablation_jobs(jobs, RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
