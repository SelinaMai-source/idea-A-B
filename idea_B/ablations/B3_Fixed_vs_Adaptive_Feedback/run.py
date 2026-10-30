#!/usr/bin/env python3
"""Fixed equal-skill allocation versus adaptive observability allocation."""

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
                "method": "EqualSkillAllocation",
                "seed": seed,
                "budget": BUDGET,
                "prefix": "ablationB3_fixed",
                "arm": "fixed_equal_skill",
            }
        )
        jobs.append(
            {
                "env": "B2",
                "method": "ObservabilityController",
                "seed": seed,
                "budget": BUDGET,
                "prefix": "ablationB3_adaptive",
                "arm": "adaptive_controller",
            }
        )
    run_ablation_jobs(jobs, RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
