#!/usr/bin/env python3
"""Disable observability control on B2; keep the same seeds and budget."""

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
                "method": "NoFeedbackController",
                "seed": seed,
                "budget": BUDGET,
                "prefix": "ablationB1_no_control",
                "arm": "no_control",
            }
        )
        jobs.append(
            {
                "env": "B2",
                "method": "ObservabilityController",
                "seed": seed,
                "budget": BUDGET,
                "prefix": "ablationB1_reference",
                "arm": "reference_controller",
            }
        )
    run_ablation_jobs(jobs, RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
