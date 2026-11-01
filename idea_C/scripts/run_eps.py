#!/usr/bin/env python3
"""e-PS (arXiv:2609.26651) runner.

Default: print the plan and refuse paper-scale. Pass ``--run`` to execute
smoke/pilot. Paper-scale (500 trials) requires ``--allow-paper-scale``.
30-seed is always labeled ``pilot``, never ``paper-scale``.
Oracle LR and Adaptive plug-in are separate flags.

This module does not start Monte Carlo unless ``--run`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

IDEA_C = Path(__file__).resolve().parents[1]
CONFIG_DIR = IDEA_C / "configs"
if str(IDEA_C.parent) not in sys.path:
    sys.path.insert(0, str(IDEA_C.parent))

from idea_C.baselines.eps.algorithms import run_eps
from idea_C.baselines.eps.dgp import draw_theta, make_eps_instance
from idea_C.shared.metrics import (
    fdr,
    samples_at_target_metric_cells,
    samples_at_target_tpr,
    summarize_samples_at_target,
    tpp,
)
from idea_C.shared.seeds import seed_everything
from idea_C.shared.serialize import write_run
from idea_C.shared.stats import mean_std_ci

SCALES = ("smoke", "pilot", "paper")


def load_config(name: str) -> dict:
    path = CONFIG_DIR / name
    if not path.exists():
        path = CONFIG_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_scale(scale: str, allow_paper: bool) -> str:
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    if scale == "paper" and not allow_paper:
        raise SystemExit(
            "Refusing paper-scale (500 trials). Re-run with "
            "--allow-paper-scale if you really want the paper budget. "
            "30-seed is pilot, never paper-scale."
        )
    return scale


def run_trials(
    cfg: dict,
    *,
    sampler: str,
    eprocess: str,
    scale: str,
    seed0: int,
    family: str,
) -> dict:
    k = int(cfg["K"])
    alpha = float(cfg.get("alpha", 0.05))
    n_trials = int(cfg["n_trials"][scale])
    horizon = int(cfg["horizon"][scale])
    truth_rng = seed_everything(seed0)
    theta = draw_theta(k, truth_rng, pi1=float(cfg.get("dgp", {}).get("pi1", 0.1)))
    fdps: list[float] = []
    tpps: list[float] = []
    samples: list[float] = []
    t0 = time.perf_counter()
    for i in range(n_trials):
        rng = seed_everything(seed0 + 1 + i)
        env = make_eps_instance(k, rng, family=family, theta=theta)
        result = run_eps(env, rng, sampler=sampler, eprocess=eprocess, alpha=alpha, budget=horizon)
        is_null = env.is_null
        fdps.append(fdr(result.final_rejected, is_null))
        tpp_path = [tpp(row, is_null) for row in result.rejected_path]
        tpps.append(tpp_path[-1] if tpp_path else 0.0)
        samples.append(samples_at_target_tpr(tpp_path, delta=alpha))
    runtime = time.perf_counter() - t0
    summary = summarize_samples_at_target(
        samples, target_tpr=1.0 - alpha, max_horizon=horizon, method=f"{sampler}/{eprocess}"
    )
    cells = samples_at_target_metric_cells(summary)
    return {
        "FDR↓": mean_std_ci(fdps)["mean"],
        "TPR↑": mean_std_ci(tpps)["mean"],
        "TPP↑": mean_std_ci(tpps)["mean"],
        "Samples@Target TPR↓": cells["Samples@Target TPR↓"],
        "Runtime↓": runtime,
        "n_trials": n_trials,
        "horizon": horizon,
        "scale": scale,
        "samples_at_target": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="e-PS runner (preprint, not CCF-A)")
    parser.add_argument("--config", default="eps_gaussian.json")
    parser.add_argument("--sampler", default="eps", choices=["uniform", "greedy", "eps"])
    parser.add_argument("--eprocess", default="oracle", choices=["oracle", "adaptive"])
    parser.add_argument("--scale", default="smoke", choices=list(SCALES))
    parser.add_argument("--allow-paper-scale", action="store_true")
    parser.add_argument("--run", action="store_true", help="actually execute trials")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        scale = resolve_scale(args.scale, args.allow_paper_scale)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    table_id = str(cfg["table_id"])
    family = "student_t" if "student" in table_id else "gaussian"
    label = "paper-scale" if scale == "paper" else ("pilot" if scale == "pilot" else "smoke")
    print(
        f"e-PS plan table_id={table_id} sampler={args.sampler} eprocess={args.eprocess} "
        f"scale={scale} label={label} trials={cfg['n_trials'][scale]} "
        f"horizon={cfg['horizon'][scale]} K={cfg['K']} "
        f"venue=preprint_not_CCF-A code_status=independent_reimplementation"
    )
    if not args.run:
        print("No --run: refusing to start Monte Carlo. Algorithms are imported and ready.")
        return 0

    metrics = run_trials(
        cfg,
        sampler=args.sampler,
        eprocess=args.eprocess,
        scale=scale,
        seed0=args.seed,
        family=family,
    )
    run_id = f"{table_id}_{args.sampler}_{args.eprocess}_{scale}_seed{args.seed}"
    write_run(
        run_id,
        table_id=table_id,
        version=f"v0_eps_{args.sampler}_{args.eprocess}_{scale}",
        content=f"{args.sampler} + {args.eprocess} {label} {metrics['n_trials']}-seed",
        data=table_id,
        config={
            **cfg,
            "sampler": args.sampler,
            "eprocess": args.eprocess,
            "scale": scale,
            "seed0": args.seed,
        },
        seed=args.seed,
        metrics={k: metrics[k] for k in ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓")},
        status="ok",
    )
    print(f"wrote results/v0/raw/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
