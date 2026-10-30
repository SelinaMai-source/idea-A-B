"""Shared helper for Idea B one-component ablations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from idea_B.experiments.run_synthetic import run_one

SEEDS = [0, 1, 2]
BUDGET = 0.05


def run_ablation_jobs(
    jobs: list[dict[str, Any]],
    results_root: Path,
    *,
    horizon: int = 32,
    population: int = 40,
    write_plots: bool = True,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    results_root.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        rec = run_one(
            job["env"],
            job["method"],
            int(job["seed"]),
            float(job.get("budget", BUDGET)),
            horizon=horizon,
            population=population,
            results_root=results_root,
            write_plots=write_plots,
            prefix=str(job.get("prefix", "ablation")),
            env_overrides=job.get("env_overrides"),
            method_overrides=job.get("method_overrides"),
        )
        rec["ablation_arm"] = job.get("arm")
        records.append(rec)
    (results_root / "ablation_index.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return records
