"""Write one Idea B run directory, including trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from idea_B.experiments.plotting import save_run_plots
from shared.evaluation.results import write_run


def experiment_id(env_name: str, method: str, budget: float, seed: int, prefix: str = "synthetic") -> str:
    budget_tag = f"{budget:.2f}".replace(".", "p")
    return f"{prefix}_{env_name}_{method}_budget{budget_tag}_seed{seed}"


def save_closed_loop_run(
    result_dir: Path,
    env: Any,
    result: dict[str, Any],
    log_text: str,
    *,
    write_plots: bool = True,
) -> Path:
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    trajectory = result["trajectory"]
    config = dict(result["config"])
    config["environment"] = env.name
    n_skills = int(env.d)

    environment = {
        "benchmark": "Synthetic_Closed_Loop",
        "idea": "Feedback-Induced Observability Collapse",
        **env.metadata(),
        "collapse": result["collapse"],
        "method_meta": result["method_meta"],
        "summary": result["summary"],
        "theta_final": result["theta_final"],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_installed": False,
    }
    try:
        import torch  # noqa: F401
    except Exception:
        environment["torch_installed"] = False
    else:
        environment["torch_installed"] = True

    write_run(result_dir, config, trajectory, environment, log_text)

    trajectories_path = result_dir / "trajectories.json"
    trajectories_path.write_text(
        json.dumps(
            {
                "trajectory": trajectory,
                "summary": result["summary"],
                "collapse": result["collapse"],
                "theta_final": result["theta_final"],
                "W_T": result["W_T"],
                "W_T_cond": result["W_T_cond"],
            },
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        result_dir / "trajectories.npz",
        **{key: np.asarray(value) for key, value in trajectory.items()},
    )
    if write_plots:
        save_run_plots(result_dir / "plots", trajectory, n_skills)
    return result_dir
