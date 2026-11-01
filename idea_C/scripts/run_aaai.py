#!/usr/bin/env python3
"""AAAI 2026 robust bandit runner.

Default: print the plan and refuse paper-scale. Pass ``--run`` to execute
smoke/pilot. Paper-scale (500 trials) requires ``--allow-paper-scale``.
30-seed is always labeled ``pilot``, never ``paper-scale``.

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

from idea_C.baselines.robust_bandit.algorithms import run_bandit_mt
from idea_C.baselines.robust_bandit.dgp import make_gaussian_instance, make_heavytail_instance
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


def _n_nonnull(spec, k: int) -> int:
    from idea_C.baselines.robust_bandit.dgp import n_nonnull_from_spec

    return n_nonnull_from_spec(spec, k)


def run_trials(cfg: dict, *, method: str, scale: str, seed0: int, table_id: str) -> dict:
    k = int(cfg["K"])
    delta = float(cfg.get("delta", 0.05))
    n_trials = int(cfg["n_trials"][scale])
    horizon = int(cfg["horizon"][scale])
    n1 = cfg.get("n_nonnull", cfg.get("n_nonnull_default", 2))
    fdps: list[float] = []
    tpps: list[float] = []
    samples: list[float] = []
    protocol_statuses: list[str] = []
    post_burnin: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_trials):
        rng = seed_everything(seed0 + i)
        if table_id == "aaai_heavy_tailed":
            env = make_heavytail_instance(k=k, n_nonnull=int(n1), rng=rng)
        else:
            env = make_gaussian_instance(k, n1, rng)
        result = run_bandit_mt(env, method=method, delta=delta, budget=horizon)
        is_null = env.is_null
        fdps.append(fdr(result.final_rejected, is_null))
        tpp_path = [tpp(row, is_null) for row in result.rejected_path]
        tpps.append(tpp_path[-1] if tpp_path else 0.0)
        samples.append(samples_at_target_tpr(tpp_path, delta=delta))
        protocol_statuses.append(str(result.protocol_status))
        post_burnin.append(
            {
                "seed": seed0 + i,
                "n_post_burnin_pulls": int(result.n_post_burnin_pulls),
                "entered_adaptive_phase": bool(result.entered_adaptive_phase),
                "burnin_per_arm": int(result.burnin_per_arm),
                "burnin_total": int(result.burnin_total),
                "post_burnin_counts": [int(x) for x in result.post_burnin_counts.tolist()],
                "post_burnin_frequencies": [float(x) for x in result.post_burnin_frequencies.tolist()],
                "n_pulls": [int(x) for x in result.n_pulls.tolist()],
                "protocol_status": str(result.protocol_status),
            }
        )
        print(
            f"  {method} seed={seed0 + i} protocol={result.protocol_status} "
            f"post_burnin={result.n_post_burnin_pulls} "
            f"KB={result.burnin_total} pulls={result.n_pulls.tolist()} "
            f"freq={result.post_burnin_frequencies.tolist()}"
        )
    runtime = time.perf_counter() - t0
    summary = summarize_samples_at_target(
        samples, target_tpr=1.0 - delta, max_horizon=horizon, method=method
    )
    cells = samples_at_target_metric_cells(summary)
    protocol = (
        "INVALID_PROTOCOL"
        if any(status == "INVALID_PROTOCOL" for status in protocol_statuses)
        else "ok"
    )
    return {
        "FDR↓": mean_std_ci(fdps)["mean"],
        "TPR↑": mean_std_ci(tpps)["mean"],
        "TPP↑": mean_std_ci(tpps)["mean"],
        "Samples@Target TPR↓": cells["Samples@Target TPR↓"],
        "Runtime↓": runtime,
        "n_trials": n_trials,
        "horizon": horizon,
        "scale": scale,
        "protocol_status": protocol,
        "samples_at_target": summary,
        "post_burnin": post_burnin,
        "entered_adaptive_phase": any(item["entered_adaptive_phase"] for item in post_burnin),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AAAI 2026 robust bandit runner")
    parser.add_argument("--config", default="aaai_robust_gaussian.json")
    parser.add_argument("--method", default="eprocess_based",
                        choices=["uniform", "mean_based", "eprocess_based", "algo3_heavytail", "nonrobust_stress"])
    parser.add_argument("--scale", default="smoke", choices=list(SCALES))
    parser.add_argument("--allow-paper-scale", action="store_true")
    parser.add_argument("--run", action="store_true", help="actually execute trials")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-nonnull", default=None, help="override |H1|")
    args = parser.parse_args(argv)

    try:
        scale = resolve_scale(args.scale, args.allow_paper_scale)
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    table_id = str(cfg["table_id"])
    if args.n_nonnull is not None:
        cfg["n_nonnull_default"] = args.n_nonnull
        cfg["n_nonnull"] = args.n_nonnull
    label = "paper-scale" if scale == "paper" else ("pilot" if scale == "pilot" else "smoke")
    print(
        f"AAAI plan table_id={table_id} method={args.method} scale={scale} "
        f"label={label} trials={cfg['n_trials'][scale]} "
        f"horizon={cfg['horizon'][scale]} K={cfg['K']} "
        f"code_status=independent_reimplementation"
    )
    if not args.run:
        print("No --run: refusing to start Monte Carlo. Algorithms are imported and ready.")
        return 0

    metrics = run_trials(cfg, method=args.method, scale=scale, seed0=args.seed, table_id=table_id)
    run_id = f"{table_id}_{args.method}_{scale}_seed{args.seed}"
    write_run(
        run_id,
        table_id=table_id,
        version=f"v0_{args.method}_{scale}",
        content=f"{args.method} {label} {metrics['n_trials']}-seed",
        data=table_id,
        config={**cfg, "method": args.method, "scale": scale, "seed0": args.seed},
        seed=args.seed,
        metrics={k: metrics[k] for k in ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓")},
        status="ok",
    )
    print(f"wrote results/v0/raw/{run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
