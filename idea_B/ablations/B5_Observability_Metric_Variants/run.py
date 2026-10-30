#!/usr/bin/env python3
"""Compare lambda_min, trace-inverse, and logdet allocation scores."""

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
        for metric, arm, prefix in (
            ("lambda_min", "lambda_min", "ablationB5_lmin"),
            ("trace_inverse", "trace_inverse", "ablationB5_atrace"),
            ("logdet", "logdet", "ablationB5_logdet"),
        ):
            jobs.append(
                {
                    "env": "B2",
                    "method": "ObservabilityController",
                    "seed": seed,
                    "budget": BUDGET,
                    "prefix": prefix,
                    "arm": arm,
                    "method_overrides": {"information_metric": metric},
                }
            )
    run_ablation_jobs(jobs, RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
