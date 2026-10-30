"""Build reports/COMPARABILITY_MATRIX.json from existing run artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "COMPARABILITY_MATRIX.json"


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cell(summary: dict[str, Any], method: str, mechanism: str, budget: str = "b=0.05") -> dict[str, Any]:
    payload = summary["per_method"][method][mechanism][budget]
    pop = payload["population_mean_reward"]
    return {
        "n_seeds": payload["n_seeds"],
        "expected_audit_fraction_mean": payload["expected_audit_fraction"]["mean"],
        "realized_audit_fraction_mean": payload["realized_audit_fraction"]["mean"],
        "population_mean_bias_mean": pop["bias"]["mean"],
        "population_mean_bias_std": pop["bias"]["std"],
        "population_mean_rmse_across_seeds": pop["rmse_across_seeds"],
        "source": "idea_A/experiments/common_protocol/SUMMARY.json",
    }


def pack_b(method_block: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_seeds": method_block["n_seeds"],
        "late_true_performance_mean": method_block["late_true_performance"]["mean"],
        "late_true_performance_std": method_block["late_true_performance"]["std"],
        "late_true_performance_values": method_block["late_true_performance"]["values"],
        "late_lambda_min_mean": method_block["late_lambda_min"]["mean"],
        "late_lambda_min_std": method_block["late_lambda_min"]["std"],
        "late_lambda_min_values": method_block["late_lambda_min"]["values"],
        "late_weak_skill_error_mean": method_block["late_weak_skill_error"]["mean"],
        "mean_feedback_cost_mean": method_block["mean_feedback_cost"]["mean"],
        "endogenous_collapse_flags": method_block["endogenous_collapse_flags"],
        "source": "idea_B/experiments/common_protocol/SUMMARY.json",
    }


def main() -> int:
    a_sum = load(ROOT / "idea_A/experiments/common_protocol/SUMMARY.json")
    b_sum = load(ROOT / "idea_B/experiments/common_protocol/SUMMARY.json")
    a_audit = load(ROOT / "idea_A/baselines/REPRODUCTION_AUDIT.json")
    b_audit = load(ROOT / "idea_B/baselines/REPRODUCTION_AUDIT.json")
    ample = load(ROOT / "idea_A/baselines/AMPLe/official_reproduction/metrics.json")
    pp = load(ROOT / "idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/metrics.json")
    ncvx = load(ROOT / "idea_B/baselines/Performative_SGD_NeurIPS2024/official_reproduction/metrics.json")

    ample_audit = a_audit["baselines"][0]
    hyper_audit = a_audit["baselines"][1]
    drago_audit = a_audit["baselines"][2]
    pp_audit = b_audit["baselines"][0]
    sgd_audit = b_audit["baselines"][1]

    a_cells = {
        "uniform_ipw": {
            "mnar": cell(a_sum, "uniform_ipw", "mnar"),
            "zero_overlap": cell(a_sum, "uniform_ipw", "zero_overlap"),
        },
        "neyman_ipw": {
            "mnar": cell(a_sum, "neyman_ipw", "mnar"),
            "zero_overlap": cell(a_sum, "neyman_ipw", "zero_overlap"),
        },
        "volume_adapted_ipw": {
            "mnar": cell(a_sum, "volume_adapted_ipw", "mnar"),
            "zero_overlap": cell(a_sum, "volume_adapted_ipw", "zero_overlap"),
        },
    }

    matrix: dict[str, Any] = {
        "title": "Comparability matrix for Idea A / Idea B smoke v1",
        "rule": (
            "Do not mix incomparable methods into one leaderboard. "
            "Status is Direct / partial / NOT DIRECTLY COMPARABLE. "
            "Numbers below are copied from run JSON; none are invented."
        ),
        "n_seeds_note": "n_seeds=3 is preliminary. Paired intervals are not confirmatory.",
        "leaderboards": {
            "idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap": {
                "methods": ["uniform_ipw", "neyman_ipw", "volume_adapted_ipw"],
                "protocol": "Idea A synthetic selective-feedback, known-q audit IPW, b=0.05, seeds 0,1,2",
                "results": "idea_A/experiments/common_protocol/SUMMARY.json",
            },
            "idea_B_B2_budget0.05_common_protocol": {
                "methods": [
                    "ObservabilityController",
                    "UniformExploration",
                    "GreedyDeployment",
                ],
                "protocol": "Idea B environment B2, 5% audit budget, horizon 32, seeds 0,1,2",
                "results": "idea_B/experiments/common_protocol/SUMMARY.json",
                "note": "GreedyDeployment is a generic control, not official SGD-GD.",
            },
            "official_reproductions_not_on_idea_leaderboards": {
                "methods": [
                    "AMPLe_official",
                    "PP-bandit_official",
                    "ncvxpp_synthetic_iter_partial",
                ],
                "note": "Keep these tables separate. Different targets, information, and budgets.",
            },
        },
        "rows": [
            {
                "method_id": "AMPLe_official",
                "paper_or_source": ample_audit["paper"],
                "venue": ample_audit["venue"],
                "vs": "Idea A",
                "status": "NOT DIRECTLY COMPARABLE",
                "protocol_label": "partial",
                "local_status": ample_audit["local_status"],
                "why": (
                    "AMPLe is query-efficient multi-dimensional preference learning: "
                    "volume/random acquisition over RM-scored pairs to identify a "
                    "simplex preference vector. Idea A is identification of "
                    "finite-population functionals under endogenous censoring with "
                    "an independent audit channel and known q. Different target, "
                    "budget, and information. Official AMPLe numbers must not enter "
                    "the Idea A Neyman/Uniform leaderboard."
                ),
                "paths": [ample_audit["local_results_path"]],
                "official_numbers_from_partial_run": {
                    "upstream_commit": ample_audit["upstream_commit"],
                    "n_rounds": 3,
                    "burnin": 50000,
                    "acquisition": "volume",
                    "l2_to_true_t0": ample["l2_to_true"]["0"],
                    "l2_to_true_t3": ample["l2_to_true"]["3"],
                    "true_preference": ample["true_preference"],
                    "estimation_t3": ample["estimation"]["3"],
                    "blocker": ample_audit["blocker"],
                },
                "leaderboard": "official_reproductions_not_on_idea_leaderboards",
            },
            {
                "method_id": "volume_adapted_ipw",
                "paper_or_source": (
                    "Mathematical analogue of AMPLe volume / max-variance, "
                    "implemented on Idea A's simulator. Not AMPLe official."
                ),
                "vs": "Idea A",
                "status": "partial",
                "protocol_label": "adapted_to_common_protocol",
                "local_status": "complete_on_requested_slice",
                "why": (
                    "Same synthetic world, budget, estimands, and known-q IPW as "
                    "Uniform and Neyman, so it may sit on the Idea A common-protocol "
                    "board as an allocation heuristic. It is only partial as a "
                    "paper comparator: AMPLe's official volume scores RM-labeled "
                    "preference pairs under an MH posterior on the simplex and was "
                    "not reused (would need the HF dataset and torch)."
                ),
                "paths": [
                    "idea_A/experiments/common_protocol/SUMMARY.json",
                    "idea_A/experiments/common_protocol/volume_adapter.py",
                ],
                "numbers_from_this_run": a_cells["volume_adapted_ipw"],
                "paired_vs_uniform_rmse": a_sum["paired_volume_vs_uniform"],
                "paired_vs_neyman_rmse": a_sum["paired_volume_vs_neyman"],
                "leaderboard": "idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap",
            },
            {
                "method_id": "HyPER",
                "paper_or_source": hyper_audit["paper"],
                "venue": hyper_audit["venue"],
                "vs": "Idea A",
                "status": "NOT DIRECTLY COMPARABLE",
                "protocol_label": "blocked",
                "local_status": hyper_audit["local_status"],
                "why": (
                    "HyPER routes human vs AI annotation to maximize downstream "
                    "reward-model / RewardBench performance. Idea A audits labels "
                    "to identify functionals under endogenous censoring. Blocked "
                    "on this machine (TPU, vLLM, 8B–13B RM training). No simplified "
                    "HyPER was invented."
                ),
                "paths": [hyper_audit["local_results_path"]],
                "blocker": hyper_audit["blocker"],
                "leaderboard": None,
            },
            {
                "method_id": "Drago_ICML2026",
                "paper_or_source": drago_audit["paper"],
                "venue": drago_audit["venue"],
                "vs": "Idea A",
                "status": "NOT DIRECTLY COMPARABLE",
                "protocol_label": "blocked",
                "local_status": drago_audit["local_status"],
                "why": (
                    "Theory comparator only: globally compatible reward "
                    "identification from preference queries vs Idea A finite-budget "
                    "audit estimation. No official code. Not an empirical competitor."
                ),
                "paths": [drago_audit["local_results_path"]],
                "blocker": drago_audit["blocker"],
                "leaderboard": None,
            },
            {
                "method_id": "uniform_ipw",
                "paper_or_source": "Generic statistical baseline (uniform audit + known-q IPW)",
                "vs": "Idea A",
                "status": "Direct",
                "protocol_label": "generic_control",
                "local_status": "complete_on_requested_slice",
                "why": (
                    "Same world, budget, estimands, and audit IPW as Neyman and the "
                    "adapted volume heuristic. Uniform q_t = b does not use the "
                    "natural channel, so for a fixed seed the audit-IPW numbers "
                    "match across mechanisms (Y is generated before S)."
                ),
                "paths": ["idea_A/experiments/common_protocol/SUMMARY.json"],
                "numbers_from_this_run": a_cells["uniform_ipw"],
                "leaderboard": "idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap",
            },
            {
                "method_id": "neyman_ipw",
                "paper_or_source": (
                    "Classical stratified Neyman allocation + exploration floor + "
                    "known-q IPW. Textbook two-phase sampling, not a novel algorithm."
                ),
                "vs": "Idea A",
                "status": "Direct",
                "protocol_label": "generic_control",
                "local_status": "complete_on_requested_slice",
                "why": (
                    "This is Idea A v1.0. Directly comparable to Uniform and the "
                    "adapted volume heuristic on the synthetic selective-feedback "
                    "protocol."
                ),
                "paths": ["idea_A/experiments/common_protocol/SUMMARY.json"],
                "numbers_from_this_run": a_cells["neyman_ipw"],
                "paired_vs_uniform": a_sum["paired_neyman_vs_uniform"],
                "leaderboard": "idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap",
            },
            {
                "method_id": "IPS_SNIPS_DR_Uncertainty_observed_only_naive_IPS",
                "paper_or_source": "Generic statistical estimators / allocations in Idea A smoke",
                "vs": "Idea A",
                "status": "Direct",
                "protocol_label": "generic_control",
                "local_status": "complete_in_smoke",
                "why": (
                    "Same synthetic protocol as Neyman/Uniform. Already run in "
                    "idea_A/results (84 runs). Not re-executed here. Not published "
                    "paper reproductions."
                ),
                "paths": ["idea_A/results/SMOKE_SUMMARY.json"],
                "leaderboard": "idea_A_synthetic_smoke_full_matrix (separate from AMPLe/HyPER)",
            },
            {
                "method_id": "PP-bandit_official",
                "paper_or_source": pp_audit["paper"],
                "venue": pp_audit["venue"],
                "vs": "Idea B",
                "status": "NOT DIRECTLY COMPARABLE",
                "protocol_label": "official_reproduction",
                "local_status": pp_audit["local_status"],
                "why": (
                    "Official numerical experiment is a 1-D reparameterized "
                    "zeroth-order performative-risk demo (Appendix H). It has "
                    "decision-dependent distributions but no Idea B skill-wise "
                    "feedback censoring, no observability Gramian, and no audit "
                    "budget. Keep as an algorithmic primitive table, not on the "
                    "B2 observability leaderboard."
                ),
                "paths": [pp_audit["local_results_path"]],
                "official_numbers": {
                    "numpy_seed": pp["numpy_seed"],
                    "elapsed_sec": pp["elapsed_sec"],
                    "single_run_final_PR_phi": pp["single_run"]["final_PR_phi"],
                    "single_run_final_PR_theta": pp["single_run"]["final_PR_theta"],
                    "multi_run_PR_phi_final_mean": pp["multi_run"]["PR_phi_final_mean"],
                    "multi_run_PR_phi_final_std": pp["multi_run"]["PR_phi_final_std"],
                    "multi_run_PR_theta_final_mean": pp["multi_run"]["PR_theta_final_mean"],
                    "multi_run_PR_theta_final_std": pp["multi_run"]["PR_theta_final_std"],
                },
                "leaderboard": "official_reproductions_not_on_idea_leaderboards",
            },
            {
                "method_id": "pp_bandit_zeroth_order_reparam_adapted",
                "paper_or_source": "Attempted port of PP-bandit two-point ZO + y^2 reparameterization",
                "vs": "Idea B",
                "status": "NOT DIRECTLY COMPARABLE",
                "protocol_label": "adapted-skipped",
                "local_status": "BLOCKED",
                "why": b_sum["adapted_skipped"][0]["reason"],
                "paths": ["idea_B/experiments/common_protocol/SUMMARY.json"],
                "ran": False,
                "leaderboard": None,
            },
            {
                "method_id": "ncvxpp_Performative_SGD",
                "paper_or_source": sgd_audit["paper"],
                "venue": sgd_audit["venue"],
                "vs": "Idea B",
                "status": "NOT DIRECTLY COMPARABLE",
                "protocol_label": "partial",
                "local_status": sgd_audit["local_status"],
                "why": (
                    "Performative SGD / SGD-GD is decision-dependent risk for a "
                    "linear classifier. It does not implement Idea B's skill-wise "
                    "feedback censoring or observability Gramian. Official full "
                    "mpiexec -np 10 run is blocked (no mpi4py). Single-replica "
                    "synthetic-iter is partial, not a 10-replica paper reproduction. "
                    "Do not relabel Idea B Greedy/Lazy as official SGD-GD."
                ),
                "paths": [sgd_audit["local_results_path"]],
                "official_partial_numbers": {
                    "wall_sec": ncvx.get("wall_sec"),
                    "local_status": ncvx.get("local_status"),
                    "npy_summaries": ncvx.get("npy_summaries"),
                },
                "blocker": sgd_audit["blocker"],
                "leaderboard": "official_reproductions_not_on_idea_leaderboards",
            },
            {
                "method_id": "ObservabilityController",
                "paper_or_source": "Idea B v1.0 method (observed-design Gramian allocation)",
                "vs": "Idea B",
                "status": "Direct",
                "protocol_label": "idea_B_v1.0",
                "local_status": "complete_on_requested_slice",
                "why": "Native method on environment B2 with the same 5% budget and seeds as Uniform and Greedy.",
                "paths": b_sum["methods"]["ObservabilityController"]["paths"],
                "numbers_from_this_run": pack_b(b_sum["methods"]["ObservabilityController"]),
                "paired_vs_uniform": b_sum["paired_controller_vs_uniform"],
                "leaderboard": "idea_B_B2_budget0.05_common_protocol",
            },
            {
                "method_id": "UniformExploration",
                "paper_or_source": "Generic control (uniform audit probability = budget)",
                "vs": "Idea B",
                "status": "Direct",
                "protocol_label": "generic_control",
                "local_status": "complete_on_requested_slice",
                "why": "Same B2 closed loop, budget, and metrics as the observability controller.",
                "paths": b_sum["methods"]["UniformExploration"]["paths"],
                "numbers_from_this_run": pack_b(b_sum["methods"]["UniformExploration"]),
                "leaderboard": "idea_B_B2_budget0.05_common_protocol",
            },
            {
                "method_id": "GreedyDeployment",
                "paper_or_source": "Generic control (audit the highest predicted scores)",
                "vs": "Idea B",
                "status": "Direct",
                "protocol_label": "generic_control",
                "local_status": "complete_on_requested_slice",
                "why": (
                    "Same B2 protocol as Uniform and the observability controller. "
                    "NOT an official Performative SGD / SGD-GD reproduction. "
                    "LazyDeployment is the same generic family and was not re-run "
                    "in this slice."
                ),
                "paths": b_sum["methods"]["GreedyDeployment"]["paths"],
                "numbers_from_this_run": pack_b(b_sum["methods"]["GreedyDeployment"]),
                "leaderboard": "idea_B_B2_budget0.05_common_protocol",
            },
            {
                "method_id": "LazyDeployment_StandardOnlineSGD",
                "paper_or_source": "Generic controls in Idea B smoke",
                "vs": "Idea B",
                "status": "Direct",
                "protocol_label": "generic_control",
                "local_status": "complete_in_smoke",
                "why": (
                    "Already present in idea_B/results as generic controls. Direct "
                    "on the Idea B protocol. Must not be relabeled as official "
                    "SGD-GD or PP-bandit. Not part of the 9-run common-protocol slice."
                ),
                "paths": ["idea_B/results/SMOKE_SUMMARY.json"],
                "leaderboard": "idea_B_synthetic_smoke (separate from official PP/SGD tables)",
            },
        ],
        "skipped": [
            {
                "item": "pp_bandit_zeroth_order_reparam_adapted",
                "status": "BLOCKED / adapted-skipped",
                "reason": b_sum["adapted_skipped"][0]["reason"],
            },
            {
                "item": "HyPER simplified implementation",
                "status": "not done (pre-declared)",
                "reason": "No simplified HyPER was invented.",
            },
            {
                "item": "three final Chinese reports",
                "status": "out of scope for this agent",
                "reason": "Parent instruction: do not generate the three final Chinese reports.",
            },
        ],
        "idea_A_common_protocol_n_runs": a_sum["matrix"]["n_runs_written"],
        "idea_B_common_protocol_n_runs": b_sum["n_runs"],
        "sources": {
            "idea_A_summary": "idea_A/experiments/common_protocol/SUMMARY.json",
            "idea_B_summary": "idea_B/experiments/common_protocol/SUMMARY.json",
            "idea_A_audit": "idea_A/baselines/REPRODUCTION_AUDIT.json",
            "idea_B_audit": "idea_B/baselines/REPRODUCTION_AUDIT.json",
        },
    }
    OUT.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
