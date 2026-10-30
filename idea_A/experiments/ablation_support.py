"""Shared helpers for A1–A5. Each ablation still has its own config/run/README."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.experiments.protocol import MethodSpec, run_world_methods
from idea_A.experiments.run_synthetic import _flatten_rows, _write_run_plot
from shared.evaluation.results import write_run
from shared.hardware.probe import collect_environment

REPO_ROOT = Path(__file__).resolve().parents[2]


def ablation_cells(config: dict[str, Any]) -> list[tuple[str, float, int]]:
    """All mechanisms at b=0.05 plus all budgets on MNAR and zero-overlap."""
    from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import MECHANISMS

    mechanisms = list(config.get("mechanisms", list(MECHANISMS)))
    budgets = [float(b) for b in config.get("budgets", [0.01, 0.05, 0.10, 0.20])]
    seeds = [int(s) for s in config.get("seeds", [0, 1, 2])]
    focus = {"mnar", "zero_overlap"}
    selected: list[tuple[str, float, int]] = []
    for mechanism in mechanisms:
        for budget in budgets:
            keep = abs(budget - 0.05) < 1e-12 or mechanism in focus
            if not keep:
                continue
            for seed in seeds:
                selected.append((mechanism, budget, seed))
    return selected


def run_ablation(
    ablation_id: str,
    ablation_dir: Path,
    specs: list[MethodSpec],
    config: dict[str, Any],
) -> dict[str, Any]:
    results_dir = ablation_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment(REPO_ROOT)
    environment = dict(environment)
    environment["experiments_run"] = True
    environment["ablation"] = ablation_id
    cells = ablation_cells(config)
    worlds: dict[tuple[str, int], Any] = {}
    payloads = []
    skipped_note = (
        "Ablation matrix is all mechanisms at b=0.05 plus all budgets on "
        "mnar and zero_overlap, seeds 0/1/2. Other budget×mechanism cells "
        "are skipped to keep runtime feasible; one component is changed."
    )
    for mechanism, budget, seed in cells:
        key = (mechanism, int(seed))
        if key not in worlds:
            worlds[key] = simulate(mechanism, int(seed), n=int(config.get("n", 2000)))
        world, rows = run_world_methods(
            mechanism, int(seed), float(budget), config, specs=specs, world=worlds[key]
        )
        exp_id = f"{ablation_id}_{mechanism}_b{float(budget):.2f}_s{int(seed)}"
        dest = results_dir / exp_id
        run_config = {
            "ablation": ablation_id,
            "experiment_id": exp_id,
            "mechanism": mechanism,
            "seed": int(seed),
            "budget": float(budget),
            "methods": [spec.method_id for spec in specs],
            "one_component_change": True,
            "config": config,
        }
        metrics = {
            "experiment_id": exp_id,
            "ablation": ablation_id,
            "mechanism": mechanism,
            "seed": int(seed),
            "budget": float(budget),
            "truths": {
                "population_mean_reward": world.truths["population_mean_reward"],
                "preference_rate": world.truths["preference_rate"],
                "policy_value_target_distribution": world.truths["policy_value_target_distribution"],
            },
            "methods": rows,
            "rows": _flatten_rows(rows),
        }
        log_text = (
            f"utc={datetime.now(timezone.utc).isoformat()}\n"
            f"ablation={ablation_id}\n"
            f"mechanism={mechanism} seed={seed} budget={budget}\n"
            f"methods={[spec.method_id for spec in specs]}\n"
        )
        write_run(dest, run_config, metrics, environment, log_text)
        _write_run_plot(dest, rows, n_seeds=1)
        payloads.append({"experiment_id": exp_id, "result_dir": str(dest), "metrics": metrics})
        print(f"[{ablation_id}] wrote {exp_id}", flush=True)

    index = {
        "ablation": ablation_id,
        "n_runs": len(payloads),
        "skipped_note": skipped_note,
        "result_dirs": [p["result_dir"] for p in payloads],
        "experiment_ids": [p["experiment_id"] for p in payloads],
    }
    (results_dir / "INDEX.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ablation_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    plt.close("all")
    return index
