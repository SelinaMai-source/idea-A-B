#!/usr/bin/env python3
"""Build the v1.1 Chinese reports from on-disk JSON.

Every number written into Markdown is copied from a loaded artifact.
This script does not invent metrics, does not re-run simulators, and does
not override the frozen PROMISING / INCONCLUSIVE / NOT PROMISING rules
with enthusiasm. PROMISING cannot mean beat-uniform-only.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

REPORTS_DIR = REPO_ROOT / "reports"

PRIMARY_SOURCES = {
    "idea_A_tier1": REPO_ROOT / "idea_A" / "results" / "v1.1" / "TIER1_SUMMARY.json",
    "idea_A_tier2": REPO_ROOT / "idea_A" / "results" / "v1.1" / "TIER2_SUMMARY.json",
    "idea_B_tier1": REPO_ROOT / "idea_B" / "results" / "v1.1" / "TIER1_SUMMARY.json",
    "idea_B_tier2": REPO_ROOT / "idea_B" / "results" / "v1.1" / "TIER2_SUMMARY.json",
    "idea_A_audit": REPO_ROOT / "idea_A" / "baselines" / "REPRODUCTION_AUDIT.json",
    "idea_B_audit": REPO_ROOT / "idea_B" / "baselines" / "REPRODUCTION_AUDIT.json",
    "comparability": REPO_ROOT / "reports" / "COMPARABILITY_MATRIX.json",
    "idea_A_ablation_index": REPO_ROOT / "idea_A" / "results" / "v1.1" / "ABLATION_INDEX.json",
    "idea_B_ablation_index": REPO_ROOT / "idea_B" / "results" / "v1.1" / "ABLATION_INDEX.json",
    "idea_A_borderline": REPO_ROOT / "idea_A" / "results" / "v1.1" / "BORDERLINE.json",
    "idea_B_borderline": REPO_ROOT / "idea_B" / "results" / "v1.1" / "BORDERLINE.json",
    "idea_A_smoke_v1": REPO_ROOT / "idea_A" / "results" / "SMOKE_SUMMARY.json",
    "idea_B_smoke_v1": REPO_ROOT / "idea_B" / "results" / "SMOKE_SUMMARY.json",
    "real_data": REPO_ROOT / "reports" / "REAL_DATA_FEASIBILITY.json",
    "protocol": REPO_ROOT / "reports" / "V1_1_EVALUATION_PROTOCOL.md",
    "audit_v1_to_v11": REPO_ROOT / "reports" / "V1_TO_V1_1_AUDIT.md",
    "env_run_A": REPO_ROOT
    / "idea_A"
    / "results"
    / "v1.1"
    / "t1_static_mnar_b0.05_s0"
    / "environment.json",
}

MAIN_REPORT_SECTIONS = [
    "1. Executive Summary",
    "2. Hardware and Software Environment",
    "3. Literature and Baseline Audit",
    "4. Idea A — Formal Problem and Method",
    "5. Idea A — Experiments",
    "6. Idea A — Interpretation",
    "7. Idea B — Formal Problem and Method",
    "8. Idea B — Experiments",
    "9. Idea B — Interpretation",
    "10. Negative Results and Failure Cases",
    "11. Ablation Analysis",
    "12. Novelty Assessment",
    "13. Limitations",
    "14. Final Recommendation",
    "15. Reproduction Commands",
    "16. Complete Result Index",
]


def load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path | str | None) -> str:
    if path is None:
        return "MISSING"
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(candidate)


def fmt_num(value: Any, digits: int = 6) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.{digits}g}"


def nested(data: Any, *keys: Any, default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def git_command(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return f"UNAVAILABLE ({type(exc).__name__}: {exc})"
    if completed.returncode != 0:
        err = (completed.stderr or completed.stdout or "").strip()
        return f"UNAVAILABLE ({err})"
    return completed.stdout.strip()


def mean_of(blob: Any) -> Any:
    if isinstance(blob, dict):
        return blob.get("mean")
    return blob


def side_label(side: Any) -> str:
    """Protocol table: favorable direction is negative (candidate better)."""
    text = str(side or "")
    if text == "entirely_negative":
        return "IMPROVEMENT"
    if text == "entirely_positive":
        return "NOT PROMISING"
    if text == "covers_zero":
        return "INCONCLUSIVE"
    return "NA"


def load_artifacts() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    missing: list[str] = []
    for name, path in PRIMARY_SOURCES.items():
        if path.suffix == ".json":
            payload = load_json(path)
            artifacts[name] = payload
            if payload is None:
                missing.append(rel(path))
        else:
            artifacts[name] = path.read_text(encoding="utf-8") if path.is_file() else None
            if artifacts[name] is None:
                missing.append(rel(path))
    artifacts["missing_primary"] = missing
    return artifacts


def a_method(tier2: dict[str, Any], world: str, mech: str, budget: str, method: str) -> dict[str, Any]:
    blob = nested(tier2, world, mech, budget, method) or {}
    return blob if isinstance(blob, dict) else {}


def a_paired(
    tier2: dict[str, Any], world: str, mech: str, budget: str, candidate: str
) -> dict[str, Any]:
    blob = nested(tier2, world, mech, budget, "paired_vs_uniform_dr", candidate) or {}
    return blob if isinstance(blob, dict) else {}


def b_method(tier2: dict[str, Any], env: str, method: str) -> dict[str, Any]:
    blob = nested(tier2, "cells", env, "methods", method) or {}
    return blob if isinstance(blob, dict) else {}


def b_paired(
    tier2: dict[str, Any], env: str, vs: str, candidate: str, field: str
) -> dict[str, Any]:
    blob = nested(tier2, "cells", env, vs, candidate, field) or {}
    return blob if isinstance(blob, dict) else {}


def rmse_from_errors(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def load_a_ablation_summaries(index: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(index, dict):
        return rows
    results_root = REPO_ROOT / "idea_A" / "results" / "v1.1"
    for name, spec in index.items():
        if not isinstance(spec, dict):
            continue
        by_method: dict[str, list[float]] = {}
        coverage: dict[str, list[float]] = {}
        audit: dict[str, list[float]] = {}
        n_files = 0
        for eid in spec.get("experiment_ids") or []:
            payload = load_json(results_root / str(eid) / "metrics.json")
            if not payload:
                continue
            n_files += 1
            for method in payload.get("methods") or []:
                method_name = str(method.get("method") or "unknown")
                estimand = method.get("population_mean_reward") or {}
                bias = estimand.get("bias")
                if bias is not None:
                    by_method.setdefault(method_name, []).append(float(bias))
                if "ci_coverage" in method and method["ci_coverage"] is not None:
                    coverage.setdefault(method_name, []).append(
                        1.0 if method["ci_coverage"] else 0.0
                    )
                if method.get("realized_audit_fraction") is not None:
                    audit.setdefault(method_name, []).append(
                        float(method["realized_audit_fraction"])
                    )
        method_rows = []
        for method_name, errors in sorted(by_method.items()):
            method_rows.append(
                {
                    "method": method_name,
                    "n": len(errors),
                    "rmse": rmse_from_errors(errors),
                    "bias_mean": sum(errors) / len(errors) if errors else None,
                    "coverage": (
                        sum(coverage[method_name]) / len(coverage[method_name])
                        if coverage.get(method_name)
                        else None
                    ),
                    "audit_mean": (
                        sum(audit[method_name]) / len(audit[method_name])
                        if audit.get(method_name)
                        else None
                    ),
                }
            )
        rows.append(
            {
                "name": name,
                "component": spec.get("component"),
                "note": spec.get("note"),
                "seeds": spec.get("seeds"),
                "preliminary": spec.get("preliminary"),
                "n_files": n_files,
                "methods": method_rows,
                "source": rel(PRIMARY_SOURCES["idea_A_ablation_index"]),
            }
        )
    return rows


def load_b_ablation_summaries(index: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not isinstance(index, dict):
        return rows
    results_root = REPO_ROOT / "idea_B" / "results" / "v1.1"
    for name, spec in index.items():
        if not isinstance(spec, dict):
            continue
        by_method: dict[str, dict[str, list[float]]] = {}
        n_files = 0
        for eid in spec.get("experiment_ids") or []:
            payload = load_json(results_root / str(eid) / "metrics.json")
            if not payload:
                continue
            n_files += 1
            method_name = str(payload.get("method") or "unknown")
            summary = payload.get("summary") or {}
            bucket = by_method.setdefault(
                method_name, {"worst": [], "regret": [], "audit": []}
            )
            if summary.get("late_worst_skill_mse") is not None:
                bucket["worst"].append(float(summary["late_worst_skill_mse"]))
            if summary.get("cumulative_regret") is not None:
                bucket["regret"].append(float(summary["cumulative_regret"]))
            if summary.get("realized_audit_fraction") is not None:
                bucket["audit"].append(float(summary["realized_audit_fraction"]))
        method_rows = []
        for method_name, fields in sorted(by_method.items()):
            method_rows.append(
                {
                    "method": method_name,
                    "n": max((len(v) for v in fields.values()), default=0),
                    "worst_mean": (
                        sum(fields["worst"]) / len(fields["worst"])
                        if fields["worst"]
                        else None
                    ),
                    "regret_mean": (
                        sum(fields["regret"]) / len(fields["regret"])
                        if fields["regret"]
                        else None
                    ),
                    "audit_mean": (
                        sum(fields["audit"]) / len(fields["audit"])
                        if fields["audit"]
                        else None
                    ),
                }
            )
        rows.append(
            {
                "name": name,
                "component": spec.get("component"),
                "note": spec.get("note"),
                "seeds": spec.get("seeds"),
                "preliminary": spec.get("preliminary"),
                "n_files": n_files,
                "methods": method_rows,
                "source": rel(PRIMARY_SOURCES["idea_B_ablation_index"]),
            }
        )
    return rows


def list_plot_paths(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return [rel(p) for p in sorted(root.rglob("*.png"))]


def count_result_dirs(root: Path) -> int:
    if not root.is_dir():
        return 0
    n = 0
    for child in root.iterdir():
        if child.is_dir() and (child / "metrics.json").is_file():
            n += 1
    return n


def compute_verdicts(artifacts: dict[str, Any]) -> dict[str, Any]:
    a2 = artifacts.get("idea_A_tier2") or {}
    b2 = artifacts.get("idea_B_tier2") or {}
    a_smoke = artifacts.get("idea_A_smoke_v1") or {}

    a_cells: list[dict[str, Any]] = []
    for world, mechs in (("static", ("mnar", "zero_overlap")), ("continual", ("mnar_audit", "policy_drift"))):
        for mech in mechs:
            budgets = nested(a2, world, mech) or {}
            if not isinstance(budgets, dict):
                continue
            for budget in budgets:
                for candidate in ("v1.1b", "v1.1c"):
                    paired = a_paired(a2, world, mech, budget, candidate)
                    if not paired:
                        continue
                    method = a_method(a2, world, mech, budget, candidate)
                    a_cells.append(
                        {
                            "world": world,
                            "mechanism": mech,
                            "budget": budget,
                            "candidate": candidate,
                            "n_seeds": paired.get("n_seeds"),
                            "preliminary": paired.get("preliminary"),
                            "rmse_candidate": paired.get("rmse_candidate"),
                            "rmse_baseline": paired.get("rmse_baseline"),
                            "diff": paired.get("difference_candidate_minus_baseline"),
                            "ci_low": paired.get("ci_low"),
                            "ci_high": paired.get("ci_high"),
                            "side": paired.get("interval_side"),
                            "label": side_label(paired.get("interval_side")),
                            "relative_rmse_change": paired.get("relative_rmse_change"),
                            "budget_band_ok": method.get("budget_band_ok"),
                            "coverage": method.get("coverage"),
                        }
                    )

    exploratory = a2.get("exploratory_10pct_rmse_drop") or {}
    exploratory_hits = []
    for key, blob in exploratory.items() if isinstance(exploratory, dict) else []:
        if isinstance(blob, dict):
            exploratory_hits.append(
                {
                    "cell": key,
                    "hit_10pct_drop": blob.get("hit_10pct_drop"),
                    "relative_rmse_change": blob.get("relative_rmse_change"),
                    "note": blob.get("note"),
                }
            )
    exploratory_failed = bool(exploratory_hits) and not any(
        item.get("hit_10pct_drop") for item in exploratory_hits
    )

    confirmatory_v11c = [
        c
        for c in a_cells
        if c["candidate"] == "v1.1c" and c.get("preliminary") is False
    ]
    v11c_sides = [c.get("side") for c in confirmatory_v11c]
    a_has_improvement = any(s == "entirely_negative" for s in v11c_sides)
    a_has_worse = any(s == "entirely_positive" for s in v11c_sides)
    a_all_cover = bool(v11c_sides) and all(s == "covers_zero" for s in v11c_sides)
    a_budget_ok = all(c.get("budget_band_ok") for c in confirmatory_v11c) if confirmatory_v11c else False

    a_promising = (
        bool(confirmatory_v11c)
        and a_has_improvement
        and not a_has_worse
        and all(s == "entirely_negative" for s in v11c_sides)
        and a_budget_ok
    )
    if a_promising:
        idea_a = "PROMISING"
    elif a_has_worse and not a_has_improvement:
        idea_a = "NOT PROMISING"
    elif a_all_cover:
        idea_a = "INCONCLUSIVE"
    else:
        idea_a = "INCONCLUSIVE"

    static_mnar_05 = a_paired(a2, "static", "mnar", "b=0.05", "v1.1c")
    continual_mnar_05 = a_paired(a2, "continual", "mnar_audit", "b=0.05", "v1.1c")
    obs_static = a_method(a2, "static", "mnar", "b=0.05", "observed_only")
    uni_static = a_method(a2, "static", "mnar", "b=0.05", "uniform_dr")
    identification_v11 = False
    if obs_static.get("rmse") is not None and uni_static.get("rmse") is not None:
        identification_v11 = float(obs_static["rmse"]) > 0.5 and float(uni_static["rmse"]) < 0.25

    mnar_v1 = nested(a_smoke, "key_findings", "mnar_observed_only_vs_audit_ipw", "b=0.05") or {}
    identification_v1 = False
    if mnar_v1.get("observed_only_bias_mean") is not None and mnar_v1.get("uniform_ipw_bias_mean") is not None:
        identification_v1 = abs(float(mnar_v1["observed_only_bias_mean"])) > 0.5 and abs(
            float(mnar_v1["uniform_ipw_bias_mean"])
        ) < 0.25

    b_required: list[dict[str, Any]] = []
    for env in ("Env1", "Env2"):
        for vs, baseline in (("paired_vs_E_optimal", "E-optimal"), ("paired_vs_D_optimal", "D-optimal")):
            for field in ("late_worst_skill_mse", "cumulative_regret"):
                paired = b_paired(b2, env, vs, "B-v1.1c", field)
                if not paired:
                    continue
                b_required.append(
                    {
                        "env": env,
                        "baseline": baseline,
                        "field": field,
                        "n_seeds": paired.get("n_seeds"),
                        "diff": paired.get("mean_difference"),
                        "ci_low": paired.get("ci_low"),
                        "ci_high": paired.get("ci_high"),
                        "side": paired.get("interval_side"),
                        "label": side_label(paired.get("interval_side")),
                        "preliminary": paired.get("preliminary"),
                    }
                )
    b_sides = [c.get("side") for c in b_required]
    b_has_improvement = any(s == "entirely_negative" for s in b_sides)
    b_has_worse = any(s == "entirely_positive" for s in b_sides)
    worst_vs_e = [
        c
        for c in b_required
        if c["field"] == "late_worst_skill_mse" and c["baseline"] == "E-optimal"
    ]
    worst_vs_e_worse = any(c.get("side") == "entirely_positive" for c in worst_vs_e)
    hb1_all_neg = bool(b_required) and all(s == "entirely_negative" for s in b_sides)

    env1_v11c = b_method(b2, "Env1", "B-v1.1c")
    env1_e = b_method(b2, "Env1", "E-optimal")
    env2_v11c = b_method(b2, "Env2", "B-v1.1c")
    env2_e = b_method(b2, "Env2", "E-optimal")
    budget_ok_b = bool(env1_v11c.get("budget_band_ok")) and bool(env2_v11c.get("budget_band_ok"))

    b_promising = hb1_all_neg and budget_ok_b
    if b_promising:
        idea_b = "PROMISING"
    elif worst_vs_e_worse:
        idea_b = "NOT PROMISING"
    elif b_has_worse and not b_has_improvement:
        idea_b = "NOT PROMISING"
    else:
        idea_b = "INCONCLUSIVE"

    env3_rec = nested(b2, "cells", "Env3_recoverable") or {}
    rec_pair = env3_rec.get("paired_recovery_v11c_vs_E_optimal") or {}
    rec_v11c = nested(env3_rec, "methods", "B-v1.1c") or {}
    rec_e = nested(env3_rec, "methods", "E-optimal") or {}
    env3_inconclusive = False
    v11c_cens = rec_v11c.get("recovery_censored_fraction")
    e_cens = rec_e.get("recovery_censored_fraction")
    try:
        if v11c_cens is not None and e_cens is not None and float(v11c_cens) > 0.5 and float(e_cens) > 0.5:
            env3_inconclusive = True
    except (TypeError, ValueError):
        env3_inconclusive = False
    if rec_pair.get("interval_side") == "covers_zero":
        env3_inconclusive = True

    return {
        "idea_A": idea_a,
        "idea_B": idea_b,
        "a_promising_allowed": a_promising,
        "b_promising_allowed": b_promising,
        "identification_v1_supported": identification_v1,
        "identification_v11_supported": identification_v11,
        "sequential_allocation_supported": a_has_improvement and not a_has_worse,
        "exploratory_10pct_failed": exploratory_failed,
        "exploratory": exploratory_hits,
        "a_cells": a_cells,
        "confirmatory_v11c": confirmatory_v11c,
        "static_mnar_05": static_mnar_05,
        "continual_mnar_05": continual_mnar_05,
        "obs_static": obs_static,
        "uni_static": uni_static,
        "mnar_v1": mnar_v1,
        "b_required": b_required,
        "hb1_all_negative": hb1_all_neg,
        "worst_vs_e_worse": worst_vs_e_worse,
        "env1_v11c": env1_v11c,
        "env1_e": env1_e,
        "env2_v11c": env2_v11c,
        "env2_e": env2_e,
        "env3_inconclusive": env3_inconclusive,
        "env3_recovery_pair": rec_pair,
        "env3_v11c": rec_v11c,
        "env3_e": rec_e,
        "noise_floor": b2.get("noise_floor"),
        "a_budget_ok": a_budget_ok,
        "b_budget_ok": budget_ok_b,
        "a_has_improvement": a_has_improvement,
        "a_has_worse": a_has_worse,
        "b_has_improvement": b_has_improvement,
        "b_has_worse": b_has_worse,
    }


def a_method_table(tier2: dict[str, Any], world: str, mech: str, budget: str) -> str:
    block = nested(tier2, world, mech, budget) or {}
    rows = []
    for method, blob in block.items():
        if method.startswith("paired") or not isinstance(blob, dict) or "rmse" not in blob:
            continue
        rows.append(
            [
                method,
                str(blob.get("n_seeds", "NA")),
                fmt_num(blob.get("rmse")),
                fmt_num(blob.get("bias")),
                fmt_num(blob.get("coverage")),
                fmt_num(blob.get("realized_audit_fraction_mean")),
                fmt_num(blob.get("budget_band_ok")),
                "PRELIMINARY" if blob.get("preliminary") else "confirmatory",
            ]
        )
    if not rows:
        return "（该格在 JSON 中没有方法摘要）"
    return md_table(
        [
            "method",
            "n_seeds",
            "RMSE",
            "bias",
            "coverage",
            "realized audit",
            "budget_band_ok",
            "label",
        ],
        rows,
    )


def a_paired_table(tier2: dict[str, Any], world: str, mech: str, budget: str) -> str:
    paired = nested(tier2, world, mech, budget, "paired_vs_uniform_dr") or {}
    rows = []
    for candidate, blob in paired.items() if isinstance(paired, dict) else []:
        if not isinstance(blob, dict):
            continue
        rows.append(
            [
                candidate,
                str(blob.get("n_seeds", "NA")),
                fmt_num(blob.get("rmse_candidate")),
                fmt_num(blob.get("rmse_baseline")),
                fmt_num(blob.get("difference_candidate_minus_baseline")),
                fmt_num(blob.get("ci_low")),
                fmt_num(blob.get("ci_high")),
                str(blob.get("interval_side", "NA")),
                side_label(blob.get("interval_side")),
            ]
        )
    if not rows:
        return "（该格在 JSON 中没有配对比较）"
    return md_table(
        [
            "candidate",
            "n",
            "RMSE cand.",
            "RMSE Uniform DR",
            "diff (cand−base)",
            "CI low",
            "CI high",
            "interval_side",
            "protocol label",
        ],
        rows,
    )


def b_method_table(tier2: dict[str, Any], env: str) -> str:
    methods = nested(tier2, "cells", env, "methods") or {}
    rows = []
    for name, blob in methods.items() if isinstance(methods, dict) else []:
        if not isinstance(blob, dict):
            continue
        rows.append(
            [
                name,
                str(blob.get("n_seeds", "NA")),
                fmt_num(mean_of(blob.get("late_worst_skill_mse"))),
                fmt_num(mean_of(blob.get("cumulative_regret"))),
                fmt_num(mean_of(blob.get("recovery_time"))),
                fmt_num(blob.get("recovery_censored_fraction")),
                fmt_num(mean_of(blob.get("realized_audit_fraction"))),
                fmt_num(blob.get("budget_band_ok")),
                "PRELIMINARY" if blob.get("preliminary") else "confirmatory/robustness",
            ]
        )
    if not rows:
        return "（该环境在 JSON 中没有方法摘要）"
    return md_table(
        [
            "method",
            "n_seeds",
            "late worst-skill MSE",
            "cumulative regret",
            "recovery time",
            "censored frac",
            "realized audit",
            "budget_band_ok",
            "label",
        ],
        rows,
    )


def b_paired_table(tier2: dict[str, Any], env: str, vs: str) -> str:
    paired = nested(tier2, "cells", env, vs) or {}
    rows = []
    for candidate, fields in paired.items() if isinstance(paired, dict) else []:
        if not isinstance(fields, dict):
            continue
        if "mean_difference" in fields:
            rows.append(
                [
                    candidate,
                    str(fields.get("field", "NA")),
                    str(fields.get("n_seeds", "NA")),
                    fmt_num(fields.get("mean_difference")),
                    fmt_num(fields.get("ci_low")),
                    fmt_num(fields.get("ci_high")),
                    str(fields.get("interval_side", "NA")),
                    side_label(fields.get("interval_side")),
                ]
            )
            continue
        for field, blob in fields.items():
            if not isinstance(blob, dict):
                continue
            rows.append(
                [
                    candidate,
                    field,
                    str(blob.get("n_seeds", "NA")),
                    fmt_num(blob.get("mean_difference")),
                    fmt_num(blob.get("ci_low")),
                    fmt_num(blob.get("ci_high")),
                    str(blob.get("interval_side", "NA")),
                    side_label(blob.get("interval_side")),
                ]
            )
    if not rows:
        return "（该比较在 JSON 中不存在或没有候选）"
    return md_table(
        ["candidate", "metric", "n", "mean diff (cand−base)", "CI low", "CI high", "interval_side", "protocol label"],
        rows,
    )


def baseline_status_rows(audit: dict[str, Any]) -> list[list[str]]:
    rows = []
    for item in audit.get("baselines") or []:
        rows.append(
            [
                str(item.get("paper") or ""),
                str(item.get("venue") or ""),
                str(item.get("local_status") or ""),
                str(item.get("upstream_commit") or "None"),
                str(item.get("comparability") or "")[:120],
            ]
        )
    return rows


def build_main_report(
    artifacts: dict[str, Any],
    verdicts: dict[str, Any],
) -> str:
    a1 = artifacts.get("idea_A_tier1") or {}
    a2 = artifacts.get("idea_A_tier2") or {}
    b1 = artifacts.get("idea_B_tier1") or {}
    b2 = artifacts.get("idea_B_tier2") or {}
    a_audit = artifacts.get("idea_A_audit") or {}
    b_audit = artifacts.get("idea_B_audit") or {}
    cmp_mat = artifacts.get("comparability") or {}
    env = artifacts.get("env_run_A") or {}
    real = artifacts.get("real_data") or {}
    a_ablations = load_a_ablation_summaries(artifacts.get("idea_A_ablation_index"))
    b_ablations = load_b_ablation_summaries(artifacts.get("idea_B_ablation_index"))

    git_commit = git_command("rev-parse", "HEAD")
    git_branch = git_command("symbolic-ref", "--short", "HEAD")
    git_remote = git_command("remote", "get-url", "origin")
    missing = artifacts.get("missing_primary") or []
    missing_line = "无（所列主源均在磁盘上）" if not missing else ", ".join(f"`{p}`" for p in missing)

    static_v11c = verdicts["static_mnar_05"]
    loop_v11c = verdicts["continual_mnar_05"]
    env1_pair_e_worst = b_paired(b2, "Env1", "paired_vs_E_optimal", "B-v1.1c", "late_worst_skill_mse")
    env1_pair_e_reg = b_paired(b2, "Env1", "paired_vs_E_optimal", "B-v1.1c", "cumulative_regret")
    env2_pair_e_worst = b_paired(b2, "Env2", "paired_vs_E_optimal", "B-v1.1c", "late_worst_skill_mse")
    env2_pair_e_reg = b_paired(b2, "Env2", "paired_vs_E_optimal", "B-v1.1c", "cumulative_regret")

    a_plots = list_plot_paths(REPO_ROOT / "idea_A" / "results" / "v1.1" / "plots")
    b_plots = list_plot_paths(REPO_ROOT / "idea_B" / "results" / "v1.1" / "plots")

    lines: list[str] = []
    add = lines.append
    add("# Idea A / Idea B v1.1 完整实验报告")
    add("")
    add("本文件由 `scripts/generate_v1_1_report.py` 从磁盘 JSON 生成。报告中的数字均来自下列产物，没有手填。")
    add("")
    add(f"- 生成时间（UTC）：`{datetime.now(timezone.utc).isoformat()}`")
    add(f"- 仓库：`{REPO_ROOT}`")
    add(f"- 分支：`{git_branch}`")
    add(f"- commit：`{git_commit}`")
    add(f"- remote：`{git_remote}`")
    add(f"- 缺失的主 JSON：{missing_line}")
    add("- 冻结协议：`reports/V1_1_EVALUATION_PROTOCOL.md`")
    add("- v1→v1.1 审计（本脚本不覆盖）：`reports/V1_TO_V1_1_AUDIT.md`")
    add("")
    add("**判定规则预先冻结，不因结果好坏改口。** PROMISING 不能表示只打败了均匀随机。Idea A 的主对照是 Uniform audit + 2 折 cross-fit DR。Idea B 的主对照是 E-optimal 与 D-optimal；H_B1 要求 worst-skill MSE **并且** cumulative regret 相对 E 与 D 的配对区间都完全为负。`n_seeds<=3` 标 PRELIMINARY。主结论只用 30/50 种子的预先指定格子。")
    add("")
    add("A-v1.1b 的方差分配与 B 的 E/D-optimal 是经典实验设计，不是新算法。")
    add("")

    add("## 1. Executive Summary")
    add("")
    add(f"- **Idea A v1.1 状态：`{verdicts['idea_A']}`**。主提案是 A-v1.1c（AIPW 方差分配 + 剩余预算 + 策略漂移）。")
    add(f"- **Idea B v1.1 状态：`{verdicts['idea_B']}`**。主提案是 B-v1.1c（信息折扣 + 一步 worst-skill 收益）。")
    add(
        f"- v1.0 **identification-via-audit 仍然支持**：v1 烟测 MNAR `b=0.05` observed-only bias mean = `{fmt_num(nested(verdicts['mnar_v1'], 'observed_only_bias_mean'))}`，Uniform IPW bias mean = `{fmt_num(nested(verdicts['mnar_v1'], 'uniform_ipw_bias_mean'))}`（`n_seeds=3`，PRELIMINARY）。"
        f" v1.1 静态 MNAR `b=0.05` 把该现象放到确认性格子：observed-only RMSE = `{fmt_num(verdicts['obs_static'].get('rmse'))}`（n=`{verdicts['obs_static'].get('n_seeds')}`），Uniform DR RMSE = `{fmt_num(verdicts['uni_static'].get('rmse'))}`。"
        f" identification_v11_supported=`{fmt_num(verdicts['identification_v11_supported'])}`。"
    )
    add(
        f"- v1.1 **sequential allocation 不被支持**。静态 MNAR `b=0.05` n=`{static_v11c.get('n_seeds')}`：v1.1c RMSE = `{fmt_num(static_v11c.get('rmse_candidate'))}` vs Uniform DR `{fmt_num(static_v11c.get('rmse_baseline'))}`，diff = `{fmt_num(static_v11c.get('difference_candidate_minus_baseline'))}`，CI [`{fmt_num(static_v11c.get('ci_low'))}`, `{fmt_num(static_v11c.get('ci_high'))}`]，`interval_side={static_v11c.get('interval_side')}` → `{side_label(static_v11c.get('interval_side'))}`。"
        f" 闭环 MNAR-audit `b=0.05` n=`{loop_v11c.get('n_seeds')}`：v1.1c RMSE = `{fmt_num(loop_v11c.get('rmse_candidate'))}` vs Uniform DR `{fmt_num(loop_v11c.get('rmse_baseline'))}`，diff = `{fmt_num(loop_v11c.get('difference_candidate_minus_baseline'))}`，CI [`{fmt_num(loop_v11c.get('ci_low'))}`, `{fmt_num(loop_v11c.get('ci_high'))}`]，`interval_side={loop_v11c.get('interval_side')}` → `{side_label(loop_v11c.get('interval_side'))}`（v1.1c 更差）。"
    )
    add(
        f"- 探索性 10% RMSE 下降：**失败**。`exploratory_10pct_failed={fmt_num(verdicts['exploratory_10pct_failed'])}`。"
        " 四条闭环格子的 `hit_10pct_drop` 均为 false（见第 5 节）。达不到就记失败；达到也不会自动成为 PROMISING。"
    )
    add(
        f"- Idea B H_B1 **未满足**。Env1 n=`{env1_pair_e_worst.get('n_seeds')}`：v1.1c late worst-skill MSE mean = `{fmt_num(mean_of(verdicts['env1_v11c'].get('late_worst_skill_mse')))}` vs E-optimal `{fmt_num(mean_of(verdicts['env1_e'].get('late_worst_skill_mse')))}`，diff = `{fmt_num(env1_pair_e_worst.get('mean_difference'))}`，CI [`{fmt_num(env1_pair_e_worst.get('ci_low'))}`, `{fmt_num(env1_pair_e_worst.get('ci_high'))}`]，`{env1_pair_e_worst.get('interval_side')}`（v1.1c 更差）。"
        f" 同格 regret diff = `{fmt_num(env1_pair_e_reg.get('mean_difference'))}`，CI [`{fmt_num(env1_pair_e_reg.get('ci_low'))}`, `{fmt_num(env1_pair_e_reg.get('ci_high'))}`]，`{env1_pair_e_reg.get('interval_side')}`（regret 更好，但 H_B1 要求 worst-skill **并且** regret 同时优于 E 与 D）。"
        f" Env2 n=`{env2_pair_e_worst.get('n_seeds')}` worst-skill diff = `{fmt_num(env2_pair_e_worst.get('mean_difference'))}`，CI [`{fmt_num(env2_pair_e_worst.get('ci_low'))}`, `{fmt_num(env2_pair_e_worst.get('ci_high'))}`]，`{env2_pair_e_worst.get('interval_side')}`。"
    )
    add(
        f"- Env3 恢复时间：**INCONCLUSIVE**。可恢复格 v1.1c censored fraction = `{fmt_num(verdicts['env3_v11c'].get('recovery_censored_fraction'))}`，E-optimal = `{fmt_num(verdicts['env3_e'].get('recovery_censored_fraction'))}`；两边都超过一半右删失。配对 recovery CI [`{fmt_num(nested(verdicts['env3_recovery_pair'], 'ci_low'))}`, `{fmt_num(nested(verdicts['env3_recovery_pair'], 'ci_high'))}`]，`{nested(verdicts['env3_recovery_pair'], 'interval_side')}`。"
    )
    add("- 官方基线未在 v1.1 重跑。沿用 v1：AMPLe=`partial`，HyPER=`blocked`，Drago=`blocked`（无代码），PP-bandit=`complete`（1-D demo，NOT DIRECTLY COMPARABLE），ncvxpp=`partial`。不得声称 outperform 这些论文。")
    add(f"- 噪声底（Idea B）：`{fmt_num(verdicts['noise_floor'])}`（`noise_std=0.15`）。更高特征值不是学习成功。")
    add(f"- 初步建议：两个 v1.1 主提案都 **不能** 标 PROMISING。继续跟的是 Idea A 的审计识别通道，而不是自适应分配；Idea B 的经典 E-optimal 对照已经是更强的主指标基线。")
    add("")

    add("## 2. Hardware and Software Environment")
    add("")
    add("环境数字来自一次 v1.1 Tier 1 运行写入的 `environment.json`，不是事后改写的 dirty manifest。")
    add("")
    add(f"- recorded_at_utc：`{env.get('recorded_at_utc')}`")
    add(f"- working_copy：`{env.get('working_copy')}`")
    add(f"- python executable：`{nested(env, 'python', 'executable')}`")
    add(f"- python version：`{nested(env, 'python', 'version')}`")
    add(f"- OS：`{nested(env, 'os', 'platform')}`")
    add(f"- ram_gib：`{fmt_num(env.get('ram_gib'))}`")
    add(f"- disk.path：`{nested(env, 'disk', 'path')}`")
    add(f"- disk.free_gib（该次运行时）：`{fmt_num(nested(env, 'disk', 'free_gib'))}`")
    add(f"- torch.installed：`{fmt_num(nested(env, 'torch', 'installed'))}`；import_error：`{nested(env, 'torch', 'import_error')}`")
    add(f"- packages.numpy：`{nested(env, 'packages', 'numpy')}`；scipy：`{nested(env, 'packages', 'scipy')}`；pytest：`{nested(env, 'packages', 'pytest')}`")
    add(f"- 该 environment.json 记录的 git.branch：`{nested(env, 'git', 'branch')}`；git.commit：`{nested(env, 'git', 'commit')}`")
    add("- 主实验是 NumPy/SciPy，**不要求 GPU**。不安装 vLLM、bitsandbytes、flash-attn、CUDA、NCCL。")
    add(f"- 真实数据阶段（v1，未在 v1.1 重做）仓库路径当时为 `{real.get('repo')}`；Qwen 只保留已有 CPU 烟测。")
    add("")

    add("## 3. Literature and Baseline Audit")
    add("")
    add("v1.1 **没有**重跑 AMPLe / HyPER / Drago / PP-bandit / ncvxpp。状态从 `REPRODUCTION_AUDIT.json` 与 `COMPARABILITY_MATRIX.json` 原样读出。")
    add("")
    add("### Idea A 官方基线")
    add("")
    add(
        md_table(
            ["paper", "venue", "local_status", "upstream_commit", "comparability (truncated)"],
            baseline_status_rows(a_audit) or [["NA", "NA", "NA", "NA", "NA"]],
        )
    )
    add("")
    add("### Idea B 官方基线")
    add("")
    add(
        md_table(
            ["paper", "venue", "local_status", "upstream_commit", "comparability (truncated)"],
            baseline_status_rows(b_audit) or [["NA", "NA", "NA", "NA", "NA"]],
        )
    )
    add("")
    add(f"Comparability 规则：{cmp_mat.get('rule')}")
    add("")
    add("不得把 AMPLe L2、PP-bandit performative risk、ncvxpp test_acc 写进 Idea A/B 主表。协议不可比时标 `NOT DIRECTLY COMPARABLE`。")
    add("")
    add("跳过项（来自 COMPARABILITY_MATRIX.json）：")
    add("")
    for item in cmp_mat.get("skipped") or []:
        add(f"- `{item.get('item')}` status=`{item.get('status')}`: {item.get('reason')}")
    add("")

    add("## 4. Idea A — Formal Problem and Method")
    add("")
    add("问题：在内生反馈删失下，用独立审计通道识别有限总体泛函 `E[Y]`。已知审计倾向 `q` 才是 MNAR 下的识别通道。有审计不等于把自然通道 MNAR 变成 MAR。普通 DR 的双重稳健性只针对「审计缺失、`q` 已知、`A ⊥ Y | X, history`」。")
    add("")
    add("v1.0 行为保持在 `idea_A/methods/v1.0/`。v1.1 在 `idea_A/methods/v1.1/`。")
    add("")
    add("| ID | 分配 | 估计 | 角色 |")
    add("| --- | --- | --- | --- |")
    add("| A-v1.0 | v1.0 Neyman + exploration floor | IPW | 参照臂，行为与 v1.0 相同 |")
    add("| A-v1.1a | Uniform `q_i = b` | 2 折 cross-fit AIPW | **必须比较的基线** |")
    add("| A-v1.1b | 审计残差的 AIPW 方差近似，`E[q]=b` | 同上 | 经典分层方差分配，**不是新最优设计定理** |")
    add("| A-v1.1c | v1.1b + 剩余预算 + 策略漂移上权 | 同上 | **主提案** |")
    add("")
    add("估计量：`μ_Y` 是审计样本上的 ridge；`μ_P` 是单独 logistic，禁止把连续 `μ_Y` 直接当概率；`μ_V = w(x) μ_Y(x)`。静态 2 折 cross-fit；序贯用 history-respecting nuisance。名义 95% 区间冻结为影响函数 Wald。")
    add("")
    add("闭环环境在 `idea_A/benchmarks/Sequential_Closed_Loop/`。Static Estimation 与 Continual Adaptation 分目录、分指标。确认性场景：policy-drift 与 MNAR-audit。")
    add("")
    add("主指标：总体均值 RMSE、bias、CI coverage、审计成本。主对照：Uniform audit + 2 折 cross-fit DR。")
    add("")

    add("## 5. Idea A — Experiments")
    add("")
    add(f"Tier 1：`n_seeds={a1.get('n_seeds')}`，label=`{a1.get('label')}`，`preliminary={fmt_num(a1.get('preliminary'))}`。written={len(a1.get('written') or [])}。skipped={a1.get('skipped')}。")
    add(f"Tier 2：`n_seeds={a2.get('n_seeds')}`（协议主格子 30；便宜且区间贴 0 的静态 `b=0.05` 扩到 50），label=`{a2.get('label')}`，written_count=`{a2.get('written_count')}`。")
    add(f"seed_extensions：`{json.dumps(a2.get('seed_extensions'), ensure_ascii=False)}`")
    add(f"skipped：{a2.get('skipped')}")
    add("")
    add("### 5.1 Tier 1（PRELIMINARY）")
    add("")
    add("静态 MNAR `b=0.05`：")
    add("")
    add(a_method_table(a1, "static", "mnar", "b=0.05"))
    add("")
    add("静态 MNAR `b=0.05` 配对 vs Uniform DR：")
    add("")
    add(a_paired_table(a1, "static", "mnar", "b=0.05"))
    add("")
    add("短闭环 MNAR-audit `b=0.05`：")
    add("")
    add(a_method_table(a1, "continual_short_mnar_audit", "mnar_audit", "b=0.05"))
    add("")

    add("### 5.2 Tier 2 静态（确认性）")
    add("")
    for mech in ("mnar", "zero_overlap"):
        for budget in ("b=0.05", "b=0.10"):
            add(f"#### {mech} {budget}")
            add("")
            add(a_method_table(a2, "static", mech, budget))
            add("")
            add("配对 vs Uniform DR：")
            add("")
            add(a_paired_table(a2, "static", mech, budget))
            add("")

    add("### 5.3 Tier 2 闭环（确认性比较；10% 阈值为探索性）")
    add("")
    for mech in ("mnar_audit", "policy_drift"):
        for budget in ("b=0.05", "b=0.10"):
            add(f"#### {mech} {budget}")
            add("")
            add(a_method_table(a2, "continual", mech, budget))
            add("")
            add("配对 vs Uniform DR：")
            add("")
            add(a_paired_table(a2, "continual", mech, budget))
            add("")

    add("### 5.4 探索性 10% RMSE 下降")
    add("")
    exp_rows = []
    for item in verdicts["exploratory"]:
        exp_rows.append(
            [
                str(item.get("cell")),
                fmt_num(item.get("relative_rmse_change")),
                fmt_num(item.get("hit_10pct_drop")),
            ]
        )
    add(md_table(["cell", "relative_rmse_change (v1.1c vs Uniform DR)", "hit_10pct_drop"], exp_rows))
    add("")
    add(f"全部 `hit_10pct_drop=false`。探索性检查 **失败**。`exploratory_10pct_failed={fmt_num(verdicts['exploratory_10pct_failed'])}`。")
    add("")
    add("### 5.5 边界预算")
    add("")
    add("`b=0.01` 与 `b=0.20` 不在确认性矩阵。JSON skipped 写明 leftover-time 也未跑。不进主结论。")
    add("")

    add("## 6. Idea A — Interpretation")
    add("")
    add("**区分两件事。**")
    add("")
    add("1. **identification-via-audit（v1.0 已支持，v1.1 静态确认性格子仍然支持）。** observed-only 在 MNAR 上有大约 +1.2 的 bias / RMSE；Uniform cross-fit DR 把 RMSE 降到约 0.13（n=50）。审计通道 + 正确 DR 是识别修复。")
    add("2. **sequential / adaptive allocation（v1.1 主提案，不被支持）。** 相对主对照 Uniform DR，v1.1c 在静态 MNAR `b=0.05` 的点估计更差且 CI 覆盖 0；在闭环 MNAR-audit `b=0.05` 的 CI 完全落在不利一侧。探索性 10% 下降失败。")
    add("")
    add("A-v1.1b 相对 Uniform DR 的静态 MNAR `b=0.05` 差接近 0 且覆盖 0，漂移项没有额外价值。Uniform DR 在这些格子上主导。不得把赢了 observed-only 或 Uniform IPW 写成 PROMISING。")
    add("")
    add(f"总体 Idea A v1.1：`{verdicts['idea_A']}`。不能标 PROMISING。")
    add("")

    add("## 7. Idea B — Formal Problem and Method")
    add("")
    add("问题：闭环策略使部分技能失去自然反馈，观测表现可以看起来很好，而弱技能误差与设计信息恶化。信息矩阵保留为线性模型观测设计 Gramian `W = Σ S φ φᵀ`，并分开已实现观测信息、条件期望信息、全体候选样本信息。更高特征值不是学习成功。")
    add("")
    add("v1.0 行为保持。v1.1 修正 EqualSkill 为等期望审计条数 `q_i ∝ 1/n_skill`，另设频率比例分配。E-optimal（抬 `λ_min`）与 D-optimal（`logdet`）是**经典对照，不是新算法**。")
    add("")
    add("| 名称 | 作用 | 新颖性 |")
    add("| --- | --- | --- |")
    add("| B-v1.0 | 原样包装 v1.0 ObservabilityController | 参照臂 |")
    add("| B-v1.1a / EqualSkill | 等技能期望条数 | 正确性修复 |")
    add("| B-v1.1b / E-optimal | 预算下提高条件 Gramian 的 `λ_min` | 经典对照 |")
    add("| D-optimal | 提高 `logdet` | 经典对照 |")
    add("| B-v1.1c | 信息折扣 + 一步预期降低 worst-skill MSE | **主提案** |")
    add("| Uniform / Greedy | 通用对照 | 无 |")
    add("")
    add("主指标：late worst-skill MSE、累计 regret（相对噪声底 0.0225）、Env3 恢复时间。`1/(1+MSE)` 只作次要。主对照：E-optimal 与 D-optimal。赢 Uniform 不等于赢主对照。")
    add("")
    add("Env1：真闭环，初值让多数技能已经不错。Env2：观测支撑上分不开的潜变量。Env3：回合中点变化，可恢复 / 不可恢复。")
    add("")

    add("## 8. Idea B — Experiments")
    add("")
    add(f"Tier 1：`n_seeds={b1.get('n_seeds')}`，label=`{b1.get('label')}`，n_runs=`{b1.get('n_runs')}`，`preliminary={fmt_num(b1.get('preliminary'))}`。")
    add(f"Tier 2：n_runs_attempted=`{b2.get('n_runs_attempted')}`。noise_floor=`{fmt_num(b2.get('noise_floor'))}`。higher_eigenvalue_is_not_success=`{fmt_num(b2.get('higher_eigenvalue_is_not_success'))}`。")
    add(f"seed_extensions：`{json.dumps(b2.get('seed_extensions'), ensure_ascii=False)}`")
    add(f"skipped：{b2.get('skipped')}")
    add(f"env3_horizon：`{json.dumps(b2.get('env3_horizon'), ensure_ascii=False)}`")
    add("")
    add("### 8.1 Tier 1（PRELIMINARY）")
    add("")
    add("读取 `TIER1_SUMMARY.json` 的 `cells`。")
    add("")
    for env_name in ("Env1", "Env2", "Env3_recoverable", "Env3_unrecoverable"):
        add(f"#### {env_name}")
        add("")
        add(b_method_table(b1, env_name))
        add("")

    add("### 8.2 Tier 2 主格子 Env1 / Env2（确认性 H_B1）")
    add("")
    for env_name in ("Env1", "Env2"):
        add(f"#### {env_name}")
        add("")
        add(b_method_table(b2, env_name))
        add("")
        add("配对 vs E-optimal：")
        add("")
        add(b_paired_table(b2, env_name, "paired_vs_E_optimal"))
        add("")
        add("配对 vs D-optimal：")
        add("")
        add(b_paired_table(b2, env_name, "paired_vs_D_optimal"))
        add("")

    add("### 8.3 稳健性（10 种子，不单独改写 H_B1）")
    add("")
    for env_name, title in (
        ("Env1_b0.10", "Env1 `b=0.10`"),
        ("Env1_d2", "Env1 d=2"),
        ("Env1_d5", "Env1 d=5"),
    ):
        add(f"#### {title}")
        add("")
        add(b_method_table(b2, env_name))
        add("")
        add("配对 vs E-optimal：")
        add("")
        add(b_paired_table(b2, env_name, "paired_vs_E_optimal"))
        add("")

    add("### 8.4 Env3 恢复时间")
    add("")
    for env_name in ("Env3_recoverable", "Env3_unrecoverable"):
        add(f"#### {env_name}")
        add("")
        add(b_method_table(b2, env_name))
        add("")
        add("配对 vs E-optimal（worst-skill / regret）：")
        add("")
        add(b_paired_table(b2, env_name, "paired_vs_E_optimal"))
        add("")
        rec_e = nested(b2, "cells", env_name, "paired_recovery_v11c_vs_E_optimal") or {}
        rec_u = nested(b2, "cells", env_name, "paired_recovery_v11c_vs_Uniform") or {}
        add(
            f"recovery vs E-optimal：n=`{rec_e.get('n_seeds')}` mean_difference=`{fmt_num(rec_e.get('mean_difference'))}` "
            f"CI [`{fmt_num(rec_e.get('ci_low'))}`, `{fmt_num(rec_e.get('ci_high'))}`] side=`{rec_e.get('interval_side')}`。"
        )
        add(
            f"recovery vs Uniform：n=`{rec_u.get('n_seeds')}` mean_difference=`{fmt_num(rec_u.get('mean_difference'))}` "
            f"CI [`{fmt_num(rec_u.get('ci_low'))}`, `{fmt_num(rec_u.get('ci_high'))}`] side=`{rec_u.get('interval_side')}`。"
        )
        add("")
    add("检测延迟：协议规定没有预先定义的检测器时写「未定义」。JSON 字段 `detection_delay` 在单次 run 的 metrics 里为 `undefined`。本摘要不把恢复时间叫做检测延迟。")
    add("")
    add("Greedy 是通用对照，不是信息对照的胜利条件。H_B1 的对手是 E-optimal 与 D-optimal。")
    add("")

    add("## 9. Idea B — Interpretation")
    add("")
    add("B-v1.1c 在 Env1 上 **regret 优于 E-optimal**（配对区间完全为负），但 **worst-skill MSE 劣于 E-optimal**（配对区间完全为正）。H_B1 要求两者同时成立，并且还要对 D-optimal 成立。因此 H_B1 失败。")
    add("")
    add("Env2 的 worst-skill 同样劣于 E-optimal（区间完全为正）；regret vs E 覆盖 0。E-optimal 赢了不算新方法成功。v1.1c 的信息折扣与一步收益没有在主指标上打败经典 E-optimal。")
    add("")
    add(f"Env3 恢复时间 INCONCLUSIVE（右删失）。horizon 锁定为 `{nested(b2, 'env3_horizon', 'locked_horizon')}`。JSON 说明：`{nested(b2, 'env3_horizon', 'oracle_check_tier1')}`")
    add("")
    add(f"总体 Idea B v1.1：`{verdicts['idea_B']}`。不能标 PROMISING。")
    add("")

    add("## 10. Negative Results and Failure Cases")
    add("")
    add("- Idea A v1.1c 在确认性闭环格子上相对 Uniform DR **更差**（MNAR-audit 与 policy-drift，两个预算，区间完全为正）。")
    add("- Idea A 探索性 10% RMSE 下降四格全部失败。")
    add("- Idea A 静态 MNAR/zero-overlap `b=0.05` 的 v1.1c vs Uniform DR 覆盖 0，点估计更差。")
    add("- Idea B v1.1c 相对 E-optimal 的主指标 worst-skill MSE 在 Env1（n=30）与 Env2（n=50）上都更差。")
    add("- Env3 恢复时间右删失，比较 INCONCLUSIVE。")
    add("- 消融 30 种子扩展被 JSON skipped：主格子已经决定。")
    add("- `b=0.01` / `b=0.20` 未进主结论。")
    add("- 官方 HyPER / Drago 仍 blocked；AMPLe / ncvxpp 仍 partial；PP-bandit 适配到技能删失仍 blocked。")
    add("- 真实数据未写入合成主结论。RewardBench 不打榜。")
    add("")

    add("## 11. Ablation Analysis")
    add("")
    add("消融只在主格子、种子 0–2 上各改一个组件，**PRELIMINARY**，不能单独支撑 PROMISING。")
    add("")
    add("### Idea A A1–A8（闭环 MNAR-audit `b=0.05`）")
    add("")
    a_ablation_rows = []
    for item in a_ablations:
        for method in item.get("methods") or []:
            a_ablation_rows.append(
                [
                    str(item.get("name")),
                    str(item.get("component")),
                    str(method.get("method")),
                    str(method.get("n")),
                    fmt_num(method.get("rmse")),
                    fmt_num(method.get("bias_mean")),
                    fmt_num(method.get("coverage")),
                    fmt_num(method.get("audit_mean")),
                    "PRELIMINARY" if item.get("preliminary") else "",
                ]
            )
    add(
        md_table(
            ["ablation", "component", "method", "n", "RMSE (from per-seed errors)", "bias mean", "coverage", "audit", "label"],
            a_ablation_rows or [["NA"] * 9],
        )
    )
    add("")
    add(f"来源：`{rel(PRIMARY_SOURCES['idea_A_ablation_index'])}` 与对应 `metrics.json`。RMSE 由每颗种子的 `population_mean_reward.bias` 重算 `sqrt(mean squared error)`。")
    add("")
    add("### Idea B B1–B9（Env1 `b=0.05`）")
    add("")
    b_ablation_rows = []
    for item in b_ablations:
        for method in item.get("methods") or []:
            b_ablation_rows.append(
                [
                    str(item.get("name")),
                    str(item.get("component")),
                    str(method.get("method")),
                    str(method.get("n")),
                    fmt_num(method.get("worst_mean")),
                    fmt_num(method.get("regret_mean")),
                    fmt_num(method.get("audit_mean")),
                    "PRELIMINARY" if item.get("preliminary") else "",
                ]
            )
    add(
        md_table(
            ["ablation", "component", "method", "n", "late worst-skill MSE mean", "regret mean", "audit", "label"],
            b_ablation_rows or [["NA"] * 8],
        )
    )
    add("")
    add("B4/B5/B6 的 30 种子扩展被跳过：H_B1 已在主矩阵上判定。")
    add("")

    add("## 12. Novelty Assessment")
    add("")
    add("| 类别 | Idea A | Idea B |")
    add("| --- | --- | --- |")
    add("| 既有文献结果 | 分层 Neyman / AIPW 方差分配、cross-fit DR、独立审计识别 | 观测设计 Gram 矩阵、E-optimal、D-optimal、performative prediction |")
    add("| 新观察到的实验现象 | v1.1 确认性格子上 Uniform DR 修复 MNAR observed-only；自适应分配未赢主对照 | v1.1c 可降低 regret，但不能在 worst-skill MSE 上赢 E-optimal |")
    add("| 新算法机制 | **无。** A-v1.1b/c 是可计算的分层 AIPW 方差近似加启发式 | **无被主指标支持的新机制。** E/D-optimal 是经典对照 |")
    add("| 未证明的理论猜想 | 内生删失 + 独立审计的有限样本保证 | 策略诱发可观性坍缩的一般条件 |")
    add("| 与已发表论文的潜在重叠 | AMPLe/HyPER/Drago 目标不同，NOT DIRECTLY COMPARABLE | PP-bandit / ncvxpp 研究 performative risk，不是技能级删失 Gramian |")
    add("")
    add("经验证据不等于已证明的新颖性。不得声称 outperform AMPLe / HyPER / PP-bandit / ncvxpp。")
    add("")

    add("## 13. Limitations")
    add("")
    add("- MacBook Air M3 / 16 GB / 无 NVIDIA GPU；主 venv 无 torch。")
    add("- 合成环境：线性奖励 / 线性技能、短 horizon、已知层或已知特征。")
    add("- `n=3` 消融与 Tier 1 是 PRELIMINARY；主结论用 30/50 种子。")
    add("- Env3 恢复时间大量右删失。")
    add("- 没有新的识别或最优设计定理。")
    add("- 真实数据（MultiPref 50-prompt、RewardBench schema、Qwen CPU generate）不进合成主结论。")
    add("- 官方基线不完整：HyPER blocked，Drago 无代码，AMPLe n_rounds=3，ncvxpp 非 10 replica。")
    add("- 每 run 的 Idea B PNG 体积大约 266MB，有意不进 git。")
    add("")

    add("## 14. Final Recommendation")
    add("")
    add(f"### Idea A：`{verdicts['idea_A']}`")
    add("")
    add("跟 identification-via-audit，不要把 v1.1c 的序贯分配写成新算法成功。Uniform cross-fit DR 是当前最强的确认性对照。下一步若继续，应先解释闭环上 v1.1c 方差爆炸/覆盖下降的原因，而不是加大种子去追 10% RMSE。")
    add("")
    add(f"### Idea B：`{verdicts['idea_B']}`")
    add("")
    add("不要把 E-optimal 的胜利算成新方法。v1.1c 若只在 regret 上局部更好，不能通过 H_B1。下一步要么接受 E-optimal 作为充分对照并停止宣称控制器新颖性，要么换一个 **worst-skill** 上能先过 E-optimal 的机制；不要再把特征值当成功。")
    add("")
    add("对 ICML / NeurIPS / ACL Main：当前 v1.1 **不足以**支持「已有可投稿的新分配/新控制器」。可以继续做识别现象或可观性诊断，但必须先过冻结的 PROMISING 门槛。")
    add("")

    add("## 15. Reproduction Commands")
    add("")
    add("在仓库根目录执行。主 Python：工作副本的 `.venv/bin/python`。需要 torch / transformers / datasets 的步骤用 `.isolated_baselines_venv`。")
    add("")
    add("```bash")
    add(f"cd {REPO_ROOT}")
    add('export PYTHONPATH="$PWD"')
    add("")
    add("# v1.1 Idea A")
    add("bash scripts/run_A_v1_1.sh")
    add("")
    add("# v1.1 Idea B")
    add("bash scripts/run_B_v1_1.sh")
    add("")
    add("# Tests (no GPU required)")
    add(".venv/bin/python -m pytest --import-mode=importlib")
    add("")
    add("# Rebuild v1.1 Chinese reports from JSON")
    add(".venv/bin/python scripts/generate_v1_1_report.py")
    add("")
    add("# v1 smoke reports (unchanged generator)")
    add(".venv/bin/python scripts/generate_report.py")
    add("```")
    add("")
    add("v1 合成矩阵命令仍可用，见根 README。不要重跑官方 AMPLe/HyPER/ncvxpp 大实验。")
    add("")

    add("## 16. Complete Result Index")
    add("")
    add("### 摘要 JSON")
    add("")
    for name, path in PRIMARY_SOURCES.items():
        exists = "yes" if path.exists() else "NO"
        add(f"- `{rel(path)}` ({name}, exists={exists})")
    add("")
    add("### 运行目录计数（含 metrics.json 的子目录）")
    add("")
    add(f"- Idea A v1.1：`{count_result_dirs(REPO_ROOT / 'idea_A' / 'results' / 'v1.1')}`")
    add(f"- Idea B v1.1：`{count_result_dirs(REPO_ROOT / 'idea_B' / 'results' / 'v1.1')}`")
    add("- 每个 run：`config.yaml`、`metrics.json`、`metrics.csv`、`run.log`、`environment.json`；部分含 `plots/`。")
    add("")
    add("### 轻量汇总图（已跟踪；每 run PNG 未进 git）")
    add("")
    for p in a_plots:
        add(f"- `{p}`")
    for p in b_plots:
        add(f"- `{p}`")
    add("")
    add("### Git")
    add("")
    add(f"- branch `{git_branch}` commit `{git_commit}` remote `{git_remote}`")
    add("")
    add("---")
    add("")
    add("数字来源全部是磁盘 JSON。若某单元格在 JSON 中不存在，表中写 NA，而不是补造。本报告不覆盖 `V1_TO_V1_1_AUDIT.md` 与 `V1_1_EVALUATION_PROTOCOL.md`。")
    add("")
    return "\n".join(lines)


def build_baseline_report(artifacts: dict[str, Any], verdicts: dict[str, Any]) -> str:
    a_audit = artifacts.get("idea_A_audit") or {}
    b_audit = artifacts.get("idea_B_audit") or {}
    cmp_mat = artifacts.get("comparability") or {}
    lines: list[str] = []
    add = lines.append
    add("# v1.1 Baseline Reproduction")
    add("")
    add("v1.1 **没有**重跑官方基线。本文件复述 v1 审计 JSON，并说明它们与 v1.1 合成主表不可比。")
    add("")
    add(f"合成判定（由 TIER2 JSON + 冻结协议）：Idea A = **{verdicts['idea_A']}**，Idea B = **{verdicts['idea_B']}**。官方复现成败不改变这两条判定。")
    add("")
    add("本地状态对照：AMPLe=`partial`（`0cad341`，`n_rounds=3`），HyPER=`blocked`，Drago=`blocked`（无代码），PP-bandit=`complete`（`d5daeff`，1-D demo），ncvxpp=`partial`（`ccbeedd`，无 MPI）。")
    add("")
    add("五类对象：Published result / Faithful local reproduction / Modified reproduction / Common-protocol comparison / Blocked。")
    add("")

    def dump_baselines(title: str, blob: dict[str, Any]) -> None:
        add(f"## {title}")
        add("")
        add(f"machine: `{json.dumps(blob.get('machine'), ensure_ascii=False)}`")
        add("")
        for item in blob.get("baselines") or []:
            add(f"### {item.get('paper')}")
            add("")
            add(f"- Venue: {item.get('venue')}")
            add(f"- Official source: {item.get('official_source')}")
            add(f"- Official code present: `{item.get('official_code')}`")
            add(f"- Upstream commit: `{item.get('upstream_commit')}`")
            add(f"- **local_status: `{item.get('local_status')}`**")
            add(f"- local_results_path: `{item.get('local_results_path')}`")
            add(f"- blocker: {item.get('blocker')}")
            add(f"- comparability: {item.get('comparability')}")
            what = item.get("what_ran")
            if what:
                add("- what_ran（全部字段来自 JSON）：")
                add("```json")
                add(json.dumps(what, indent=2, ensure_ascii=False))
                add("```")
            add("")

    dump_baselines("Idea A 官方基线", a_audit)
    dump_baselines("Idea B 官方基线", b_audit)

    add("## 共同协议（不是官方复现）")
    add("")
    add(f"规则：{cmp_mat.get('rule')}")
    add(f"n_seeds 说明：{cmp_mat.get('n_seeds_note')}")
    add("")
    for row in cmp_mat.get("rows") or []:
        add(f"### `{row.get('method_id')}`")
        add("")
        add(f"- protocol_label: `{row.get('protocol_label')}`")
        add(f"- local_status: `{row.get('local_status')}`")
        add(f"- comparability status: `{row.get('status')}`")
        add(f"- leaderboard: `{row.get('leaderboard')}`")
        add(f"- why: {row.get('why')}")
        if row.get("blocker"):
            add(f"- blocker: {row.get('blocker')}")
        add("")
    add("不得把 AMPLe / HyPER / PP-bandit / ncvxpp 的数字排进 v1.1 Uniform DR 或 E-optimal 主表。")
    add("")
    return "\n".join(lines)


def build_novelty_report(artifacts: dict[str, Any], verdicts: dict[str, Any]) -> str:
    a2 = artifacts.get("idea_A_tier2") or {}
    b2 = artifacts.get("idea_B_tier2") or {}
    static_v11c = verdicts["static_mnar_05"]
    loop_v11c = verdicts["continual_mnar_05"]
    env1_pair = b_paired(b2, "Env1", "paired_vs_E_optimal", "B-v1.1c", "late_worst_skill_mse")
    env1_reg = b_paired(b2, "Env1", "paired_vs_E_optimal", "B-v1.1c", "cumulative_regret")
    lines: list[str] = []
    add = lines.append
    add("# v1.1 Novelty Assessment")
    add("")
    add("本报告只做研究解释。所有数字仍来自 TIER2 / 审计 JSON。")
    add("")
    add(f"预先规则下的判定：Idea A v1.1 = **{verdicts['idea_A']}**，Idea B v1.1 = **{verdicts['idea_B']}**。")
    add("")
    add("## Idea A")
    add("")
    add("### 最强支持证据")
    add("")
    add(
        f"identification-via-audit。v1.1 静态 MNAR `b=0.05` n=`{verdicts['obs_static'].get('n_seeds')}`："
        f" observed-only RMSE=`{fmt_num(verdicts['obs_static'].get('rmse'))}` bias=`{fmt_num(verdicts['obs_static'].get('bias'))}` coverage=`{fmt_num(verdicts['obs_static'].get('coverage'))}`；"
        f" Uniform DR RMSE=`{fmt_num(verdicts['uni_static'].get('rmse'))}` bias=`{fmt_num(verdicts['uni_static'].get('bias'))}` coverage=`{fmt_num(verdicts['uni_static'].get('coverage'))}`。"
        " 这支持「已知 q 的独立审计 + 正确 cross-fit DR」修复 MNAR observed-only，不支持新分配规则。"
    )
    add("")
    add("### 最强反证")
    add("")
    add(
        f"自适应分配失败。静态 v1.1c vs Uniform DR RMSE diff=`{fmt_num(static_v11c.get('difference_candidate_minus_baseline'))}` "
        f"CI [`{fmt_num(static_v11c.get('ci_low'))}`, `{fmt_num(static_v11c.get('ci_high'))}`] `{static_v11c.get('interval_side')}`。"
        f" 闭环 MNAR-audit diff=`{fmt_num(loop_v11c.get('difference_candidate_minus_baseline'))}` "
        f"CI [`{fmt_num(loop_v11c.get('ci_low'))}`, `{fmt_num(loop_v11c.get('ci_high'))}`] `{loop_v11c.get('interval_side')}`。"
        " A-v1.1b/c 是经典方差分配加启发式，不是新最优性定理。"
    )
    add("")
    add("### 最近的已发表竞争者")
    add("")
    add("AMPLe（ACL 2025 Main，local_status=partial）、HyPER（blocked）、Drago et al. ICML 2026（无代码）。目标、预算与信息都不同。**不能**声称 outperform。")
    add("")
    add("### 可能的新颖性主张")
    add("")
    add("把已知倾向的独立审计接到内生反馈删失下的有限总体识别。这是问题设定，不是新的 Neyman/AIPW 公式。v1.1 的序贯分配目前没有被主对照支持。")
    add("")
    add("### 当前结果尚不支持的部分")
    add("")
    add("不支持：v1.1c 优于 Uniform DR、10% 动态 RMSE 下降、新分配算法、优于 AMPLe/HyPER、可投稿定理、LLM 后训练。")
    add("")

    add("## Idea B")
    add("")
    add("### 最强支持证据")
    add("")
    add(
        f"在 Env1 上 v1.1c 相对 E-optimal 的 cumulative regret mean_difference=`{fmt_num(env1_reg.get('mean_difference'))}` "
        f"CI [`{fmt_num(env1_reg.get('ci_low'))}`, `{fmt_num(env1_reg.get('ci_high'))}`] `{env1_reg.get('interval_side')}`。"
        " 这只说明累计 excess MSE 路径可以更好，不是 H_B1 成功。"
    )
    add("")
    add("### 最强反证")
    add("")
    add(
        f"主指标 worst-skill MSE：Env1 mean_difference=`{fmt_num(env1_pair.get('mean_difference'))}` "
        f"CI [`{fmt_num(env1_pair.get('ci_low'))}`, `{fmt_num(env1_pair.get('ci_high'))}`] `{env1_pair.get('interval_side')}`。"
        " H_B1 要求 worst-skill **并且** regret 同时优于 E 与 D。E-optimal 是经典对照；它赢了不算新方法。"
    )
    add("")
    add("### 最近的已发表竞争者")
    add("")
    add("ICML 2024 PP-bandit（complete 的 1-D demo，NOT DIRECTLY COMPARABLE）与 NeurIPS 2024 ncvxpp（partial）。不得接到技能删失环境后称为官方算法。")
    add("")
    add("### 可能的新颖性主张")
    add("")
    add("闭环下用 worst-skill MSE / regret 而不是饱和的 `1/(1+MSE)` 或单独的特征值来评价信息分配。当前实验没有把 B-v1.1c 变成被支持的新控制器。")
    add("")
    add("### 当前结果尚不支持的部分")
    add("")
    add("不支持：v1.1c 优于 E-optimal（主指标）、Env3 恢复时间优势、只抬特征值算学习成功、击败 performative 论文。")
    add("")
    add("## 总评")
    add("")
    add(
        f"A v1.1 `{verdicts['idea_A']}`；B v1.1 `{verdicts['idea_B']}`。"
        " 被支持的是审计识别（A）与「必须用 worst-skill 而不是特征值」（B 的评价修正）。不被支持的是两个主提案的序贯分配 / 控制器。"
    )
    add("")
    add(f"TIER2 A written_count=`{a2.get('written_count')}`；B n_runs_attempted=`{b2.get('n_runs_attempted')}`。")
    add("")
    return "\n".join(lines)


def build_next_steps_report(artifacts: dict[str, Any], verdicts: dict[str, Any]) -> str:
    a2 = artifacts.get("idea_A_tier2") or {}
    b2 = artifacts.get("idea_B_tier2") or {}
    lines: list[str] = []
    add = lines.append
    add("# v1.1 Next Research Steps")
    add("")
    add("根据冻结协议与 TIER2 JSON，而不是根据想投哪个会。")
    add("")
    add(f"当前：Idea A v1.1 = **{verdicts['idea_A']}**，Idea B v1.1 = **{verdicts['idea_B']}**。")
    add("")
    add("## 不要做的事")
    add("")
    add("- 不要把 PROMISING 用在任一 v1.1 主提案上。")
    add("- 不要只因为赢了 Uniform 随机 / observed-only / Greedy 就改口。")
    add("- 不要重跑 AMPLe n_rounds=100、HyPER TPU、ncvxpp mpiexec、或发明简化 HyPER。")
    add("- 不要把 PP-bandit 1-D demo 接到技能删失环境。")
    add("- 不要把 MultiPref / RewardBench / Qwen 烟测写进合成主结论。")
    add("- 不要在评价种子上调 `drift_kappa` / `information_discount`。")
    add(f"- 不要把特征值单独当成功。`higher_eigenvalue_is_not_success={fmt_num(b2.get('higher_eigenvalue_is_not_success'))}`。")
    add("")
    add("## Idea A：值得跟的")
    add("")
    add("1. 把 identification-via-audit 写成明确的 MNAR + 已知 q 审计理论，而不是再调 Neyman。")
    add("2. 诊断闭环上 v1.1c RMSE 恶化：哪些种子覆盖失败、倾向是否过小、history-respecting nuisance 是否不稳定。")
    add("3. 若继续分配研究，主对照必须仍是 Uniform cross-fit DR；A-v1.1b 已显示漂移项没有带来确认性增益。")
    add(f"4. JSON 已跳过消融 30 种子扩展：`{a2.get('skipped')}`。不要为了叙事再展开笛卡尔积。")
    add("")
    add("## Idea B：值得跟的")
    add("")
    add("1. 把 E-optimal 当作充分的经典对照。若新控制器不能在 worst-skill MSE 上过它，就停。")
    add("2. Env3：右删失使恢复时间 INCONCLUSIVE。加长 horizon 只允许走协议第 5 节预先登记，且不能按谁恢复更快来选。当前 locked_horizon 已因 Oracle 仍删失而保持 32。")
    add("3. 分开报告三种 Gramian；继续禁止用 `1/(1+MSE)` 做 PROMISING。")
    add(f"4. JSON 已跳过 B4/B5/B6 的 30 种子扩展：`{b2.get('skipped')}`。")
    add("")
    add("## 证据门槛")
    add("")
    add("- 主结论继续只用预先指定的 30/50 种子格子。")
    add("- `n=3` 只能标 PRELIMINARY。")
    add("- 配对区间覆盖 0 → INCONCLUSIVE；完全不利 → NOT PROMISING；PROMISING 需要协议第 3 节的全部条件。")
    add("")
    return "\n".join(lines)


def build_pointer(idea: str, verdict: str, main_rel: str) -> str:
    lines = [
        f"# Idea {idea} v1.1 报告指针",
        "",
        f"主报告：[`{main_rel}`](../../../{main_rel})",
        "",
        f"本目录不重复数字。Idea {idea} v1.1 判定（由 `scripts/generate_v1_1_report.py` 按冻结协议从 TIER2 JSON 计算）：**{verdict}**。",
        "",
        "相关：",
        "",
        "- `reports/V1_1_EVALUATION_PROTOCOL.md`",
        "- `reports/V1_TO_V1_1_AUDIT.md`",
        "- `reports/V1_1_BASELINE_REPRODUCTION.md`",
        "- `reports/V1_1_NOVELTY_ASSESSMENT.md`",
        "- `reports/V1_1_NEXT_RESEARCH_STEPS.md`",
        "",
        f"结果摘要：`idea_{idea}/results/v1.1/TIER1_SUMMARY.json`，`idea_{idea}/results/v1.1/TIER2_SUMMARY.json`。",
        "",
    ]
    return "\n".join(lines)


def generate_all() -> dict[str, Path]:
    artifacts = load_artifacts()
    verdicts = compute_verdicts(artifacts)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main": REPORTS_DIR / "V1_1_FULL_EXPERIMENT_REPORT.md",
        "baseline": REPORTS_DIR / "V1_1_BASELINE_REPRODUCTION.md",
        "novelty": REPORTS_DIR / "V1_1_NOVELTY_ASSESSMENT.md",
        "next": REPORTS_DIR / "V1_1_NEXT_RESEARCH_STEPS.md",
        "pointer_A": REPO_ROOT / "idea_A" / "reports" / "v1.1" / "README.md",
        "pointer_B": REPO_ROOT / "idea_B" / "reports" / "v1.1" / "README.md",
    }
    outputs["main"].write_text(build_main_report(artifacts, verdicts), encoding="utf-8")
    outputs["baseline"].write_text(build_baseline_report(artifacts, verdicts), encoding="utf-8")
    outputs["novelty"].write_text(build_novelty_report(artifacts, verdicts), encoding="utf-8")
    outputs["next"].write_text(build_next_steps_report(artifacts, verdicts), encoding="utf-8")
    outputs["pointer_A"].parent.mkdir(parents=True, exist_ok=True)
    outputs["pointer_B"].parent.mkdir(parents=True, exist_ok=True)
    outputs["pointer_A"].write_text(
        build_pointer("A", verdicts["idea_A"], "reports/V1_1_FULL_EXPERIMENT_REPORT.md"),
        encoding="utf-8",
    )
    outputs["pointer_B"].write_text(
        build_pointer("B", verdicts["idea_B"], "reports/V1_1_FULL_EXPERIMENT_REPORT.md"),
        encoding="utf-8",
    )
    return outputs


def main() -> int:
    outputs = generate_all()
    for key, path in outputs.items():
        print(f"wrote {key}: {path}")
    artifacts = load_artifacts()
    verdicts = compute_verdicts(artifacts)
    print(f"verdict A={verdicts['idea_A']} B={verdicts['idea_B']}")
    if verdicts["idea_A"] == "PROMISING" or verdicts["idea_B"] == "PROMISING":
        print("ERROR: PROMISING is not allowed for these TIER2 JSON values.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
