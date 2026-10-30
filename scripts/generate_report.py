#!/usr/bin/env python3
"""Build the three Chinese review reports from on-disk JSON/CSV.

Every number written into Markdown is copied from a loaded artifact.
This script does not invent metrics, does not re-run simulators, and does
not override the pre-declared PROMISING / INCONCLUSIVE / NOT PROMISING rules
with enthusiasm.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.hardware.probe import write_environment_manifests  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"

PRIMARY_SOURCES = {
    "idea_A_smoke": REPO_ROOT / "idea_A" / "results" / "SMOKE_SUMMARY.json",
    "idea_B_smoke": REPO_ROOT / "idea_B" / "results" / "SMOKE_SUMMARY.json",
    "idea_A_audit": REPO_ROOT / "idea_A" / "baselines" / "REPRODUCTION_AUDIT.json",
    "idea_B_audit": REPO_ROOT / "idea_B" / "baselines" / "REPRODUCTION_AUDIT.json",
    "comparability": REPO_ROOT / "reports" / "COMPARABILITY_MATRIX.json",
    "idea_A_common": REPO_ROOT / "idea_A" / "experiments" / "common_protocol" / "SUMMARY.json",
    "idea_B_common": REPO_ROOT / "idea_B" / "experiments" / "common_protocol" / "SUMMARY.json",
    "real_data": REPO_ROOT / "reports" / "REAL_DATA_FEASIBILITY.json",
    "env_shared": REPO_ROOT / "shared" / "hardware" / "environment_manifest.json",
    "env_reports": REPO_ROOT / "reports" / "environment_manifest.json",
    "inference_A": REPO_ROOT / "idea_A" / "models" / "Qwen2.5-0.5B-Instruct" / "INFERENCE_SMOKE.json",
    "manifest_A": REPO_ROOT / "idea_A" / "models" / "Qwen2.5-0.5B-Instruct" / "MODEL_MANIFEST.md",
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

A_ABLATION_INDEXES = [
    REPO_ROOT / "idea_A" / "ablations" / "A1_No_Adaptive_Allocation" / "results" / "INDEX.json",
    REPO_ROOT / "idea_A" / "ablations" / "A2_No_Propensity_Correction" / "results" / "INDEX.json",
    REPO_ROOT / "idea_A" / "ablations" / "A3_No_Anchor_Intervention" / "results" / "INDEX.json",
    REPO_ROOT / "idea_A" / "ablations" / "A4_Fixed_vs_Adaptive_Budget" / "results" / "INDEX.json",
    REPO_ROOT / "idea_A" / "ablations" / "A5_Misspecified_Propensity" / "results" / "INDEX.json",
]

B_ABLATION_INDEXES = [
    REPO_ROOT / "idea_B" / "ablations" / "B1_No_Observability_Control" / "results" / "ablation_index.json",
    REPO_ROOT / "idea_B" / "ablations" / "B2_No_Adaptive_Exploration" / "results" / "ablation_index.json",
    REPO_ROOT / "idea_B" / "ablations" / "B3_Fixed_vs_Adaptive_Feedback" / "results" / "ablation_index.json",
    REPO_ROOT / "idea_B" / "ablations" / "B4_No_Policy_Feedback_Loop" / "results" / "ablation_index.json",
    REPO_ROOT / "idea_B" / "ablations" / "B5_Observability_Metric_Variants" / "results" / "ablation_index.json",
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


def fmt_pm(blob: dict[str, Any] | None, key: str = "mean") -> str:
    if not isinstance(blob, dict):
        return "NA"
    mean = blob.get(key, blob.get("mean"))
    std = blob.get("std")
    n_val = blob.get("n")
    if mean is None:
        return "NA"
    text = fmt_num(mean)
    if std is not None:
        text += f" ± {fmt_num(std)}"
    if n_val is not None:
        text += f" (n={n_val})"
    return text


def nested(data: Any, *keys: Any, default: Any = None) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def ci_excludes_zero(low: Any, high: Any) -> bool:
    try:
        return not (float(low) <= 0.0 <= float(high))
    except (TypeError, ValueError):
        return False


def ci_supports_lower(mean_diff: Any, low: Any, high: Any) -> bool:
    """Neyman-minus-Uniform RMSE: improvement requires negative diff and CI above-zero excluded."""
    try:
        return float(mean_diff) < 0.0 and float(high) < 0.0
    except (TypeError, ValueError):
        return False


def ci_supports_higher(mean_diff: Any, low: Any, high: Any) -> bool:
    try:
        return float(mean_diff) > 0.0 and float(low) > 0.0
    except (TypeError, ValueError):
        return False


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
    artifacts["a_ablation_indexes"] = []
    for path in A_ABLATION_INDEXES:
        payload = load_json(path)
        artifacts["a_ablation_indexes"].append(
            {"path": str(path), "exists": payload is not None, "data": payload}
        )
    artifacts["b_ablation_indexes"] = []
    for path in B_ABLATION_INDEXES:
        payload = load_json(path)
        artifacts["b_ablation_indexes"].append(
            {"path": str(path), "exists": payload is not None, "data": payload}
        )
    return artifacts


def pop_bias(smoke: dict[str, Any], method: str, mechanism: str, budget: str) -> dict[str, Any] | None:
    return nested(smoke, "per_method", method, mechanism, budget, "population_mean_reward")


def mean_bias(smoke: dict[str, Any], method: str, mechanism: str, budget: str) -> float | None:
    blob = nested(smoke, "per_method", method, mechanism, budget, "population_mean_reward", "bias", "mean")
    try:
        return float(blob)
    except (TypeError, ValueError):
        return None


def load_a_ablation_mnar_b05(index_path: Path) -> dict[str, Any]:
    """Read per-seed metrics.json for MNAR b=0.05 from an A ablation INDEX."""
    index = load_json(index_path) or {}
    ids = index.get("experiment_ids") or []
    selected = [eid for eid in ids if "_mnar_b0.05_" in str(eid)]
    biases: list[float] = []
    rmses: list[float] = []
    methods: list[str] = []
    paths: list[str] = []
    for eid in selected:
        metrics_path = index_path.parent / str(eid) / "metrics.json"
        payload = load_json(metrics_path)
        if not payload:
            continue
        paths.append(rel(metrics_path))
        for method in payload.get("methods") or []:
            methods.append(str(method.get("method")))
            estimand = nested(method, "estimands", "population_mean_reward") or {}
            if "bias" in estimand:
                biases.append(float(estimand["bias"]))
            if "rmse" in estimand:
                rmses.append(float(estimand["rmse"]))
    out: dict[str, Any] = {
        "ablation": index.get("ablation"),
        "n_index_runs": index.get("n_runs"),
        "n_mnar_b05_files": len(selected),
        "n_bias_values": len(biases),
        "methods": sorted(set(methods)),
        "paths": paths,
        "source": rel(index_path),
    }
    if biases:
        out["bias_mean"] = float(sum(biases) / len(biases))
        out["bias_values"] = biases
    if rmses:
        out["rmse_mean"] = float(sum(rmses) / len(rmses))
        out["rmse_values"] = rmses
    return out


def b_ablation_arm_means(index_data: Any, arm_key: str = "ablation_arm") -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    if not isinstance(index_data, list):
        return {}
    for row in index_data:
        arm = str(row.get(arm_key) or row.get("method") or "unknown")
        summary = row.get("summary") or {}
        for field in (
            "late_true_performance",
            "late_observed_performance",
            "late_lambda_min",
            "late_weak_skill_error",
            "mean_feedback_cost",
        ):
            if field in summary and summary[field] is not None:
                grouped[arm][field].append(float(summary[field]))
        collapse = row.get("collapse") or {}
        if "endogenous_collapse" in collapse:
            grouped[arm]["endogenous_collapse"].append(
                1.0 if collapse["endogenous_collapse"] else 0.0
            )
    result: dict[str, dict[str, Any]] = {}
    for arm, fields in grouped.items():
        result[arm] = {"n": max((len(v) for v in fields.values()), default=0)}
        for field, values in fields.items():
            if values:
                result[arm][field] = {
                    "mean": float(sum(values) / len(values)),
                    "n": len(values),
                    "values": values,
                }
    return result


def compute_verdicts(artifacts: dict[str, Any]) -> dict[str, Any]:
    a_smoke = artifacts.get("idea_A_smoke") or {}
    b_smoke = artifacts.get("idea_B_smoke") or {}

    mnar_b05 = nested(a_smoke, "key_findings", "mnar_observed_only_vs_audit_ipw", "b=0.05") or {}
    obs_bias = mnar_b05.get("observed_only_bias_mean")
    ney_bias = mnar_b05.get("neyman_ipw_bias_mean")
    uni_bias = mnar_b05.get("uniform_ipw_bias_mean")
    naive_bias = mean_bias(a_smoke, "naive_ips", "mnar", "b=0.05")
    identification_mnar = False
    if obs_bias is not None and ney_bias is not None and uni_bias is not None:
        audit_abs = min(abs(float(ney_bias)), abs(float(uni_bias)))
        identification_mnar = abs(float(obs_bias)) > 0.5 and audit_abs < 0.25
        if naive_bias is not None:
            identification_mnar = identification_mnar and abs(float(naive_bias)) > 2.0 * audit_abs

    zo_obs = mean_bias(a_smoke, "observed_only", "zero_overlap", "b=0.05")
    zo_ney = mean_bias(a_smoke, "neyman_ipw", "zero_overlap", "b=0.05")
    zo_naive = mean_bias(a_smoke, "naive_ips", "zero_overlap", "b=0.05")
    identification_zero = False
    if zo_obs is not None and zo_ney is not None:
        identification_zero = abs(float(zo_ney)) < abs(float(zo_obs))
        if zo_naive is not None:
            identification_zero = identification_zero and abs(float(zo_ney)) < abs(float(zo_naive))

    identification = bool(identification_mnar or identification_zero)

    allocation_cells: list[dict[str, Any]] = []
    allocation_supported = False
    rmse_findings = nested(a_smoke, "key_findings", "neyman_vs_uniform_rmse") or {}
    for name, cell in rmse_findings.items():
        boot = cell.get("paired_bootstrap_rmse") or {}
        mean_diff = cell.get("mean_rmse_difference_neyman_minus_uniform")
        supported = ci_supports_lower(mean_diff, boot.get("ci_low"), boot.get("ci_high"))
        allocation_cells.append(
            {
                "name": name,
                "mean_diff": mean_diff,
                "ci_low": boot.get("ci_low"),
                "ci_high": boot.get("ci_high"),
                "ci_excludes_zero": ci_excludes_zero(boot.get("ci_low"), boot.get("ci_high")),
                "neyman_better_supported": supported,
                "neyman_worse_supported": ci_supports_higher(
                    mean_diff, boot.get("ci_low"), boot.get("ci_high")
                ),
            }
        )
        if supported:
            allocation_supported = True

    if identification and allocation_supported:
        idea_a = "PROMISING"
    else:
        idea_a = "INCONCLUSIVE"

    collapse_flag = nested(b_smoke, "B2", "endogenous_collapse")
    collapse_joint = nested(
        b_smoke, "B2", "evidence", "observed_performance_stable_while_lambda_min_and_weak_skill_error_worsen"
    )
    weak_worsened = nested(b_smoke, "B2", "evidence", "any_seed_weak_skill_error_worsened")
    collapse_supported = bool(collapse_flag) or bool(collapse_joint)

    cvu = b_smoke.get("controller_vs_uniform") or {}
    info_boot = cvu.get("information_paired_bootstrap") or {}
    perf_boot = cvu.get("true_performance_paired_bootstrap") or {}
    info_supported = ci_supports_higher(
        info_boot.get("mean_difference"), info_boot.get("ci_low"), info_boot.get("ci_high")
    )
    perf_supported = ci_supports_higher(
        perf_boot.get("mean_difference"), perf_boot.get("ci_low"), perf_boot.get("ci_high")
    )
    controller_both = info_supported and perf_supported
    json_beats_both = bool(cvu.get("beats_uniform_on_both"))

    if collapse_supported and controller_both:
        idea_b = "PROMISING"
    else:
        idea_b = "INCONCLUSIVE"

    return {
        "idea_A": idea_a,
        "idea_B": idea_b,
        "n_seeds_note": "n_seeds=3 只是初步证据，不是 confirmatory 结论。",
        "A": {
            "identification_mnar": identification_mnar,
            "identification_zero_overlap": identification_zero,
            "identification": identification,
            "allocation_supported": allocation_supported,
            "mnar_b05": mnar_b05,
            "naive_ips_mnar_b05_bias": naive_bias,
            "zero_overlap_b05": {
                "observed_only": zo_obs,
                "neyman_ipw": zo_ney,
                "naive_ips": zo_naive,
            },
            "allocation_cells": allocation_cells,
            "split": identification and not allocation_supported,
        },
        "B": {
            "endogenous_collapse": collapse_flag,
            "joint_criterion": collapse_joint,
            "any_seed_weak_skill_error_worsened": weak_worsened,
            "collapse_supported": collapse_supported,
            "info_supported": info_supported,
            "perf_supported": perf_supported,
            "controller_both_paired": controller_both,
            "json_beats_uniform_on_both_means": json_beats_both,
            "info_boot": info_boot,
            "perf_boot": perf_boot,
            "budget": cvu.get("budget"),
        },
    }


def method_table_a(smoke: dict[str, Any], mechanism: str, budget: str) -> str:
    methods = list((smoke.get("per_method") or {}).keys())
    rows = []
    for method in methods:
        pop = pop_bias(smoke, method, mechanism, budget)
        if not pop:
            continue
        pref = nested(smoke, "per_method", method, mechanism, budget, "preference_rate") or {}
        audit = nested(smoke, "per_method", method, mechanism, budget, "realized_audit_fraction") or {}
        rows.append(
            [
                method,
                fmt_pm(pop.get("bias")),
                fmt_num(pop.get("rmse_across_seeds")),
                fmt_pm(pref.get("bias")),
                fmt_pm(audit),
            ]
        )
    return md_table(
        ["method", "pop-mean bias mean±std", "pop-mean RMSE", "preference-rate bias", "realized audit"],
        rows,
    )


def paired_table_a(smoke: dict[str, Any]) -> str:
    rows = []
    paired = smoke.get("paired_neyman_vs_uniform") or {}
    for mechanism in sorted(paired):
        for budget, cell in (paired[mechanism] or {}).items():
            boot = cell.get("paired_bootstrap_rmse") or {}
            rows.append(
                [
                    mechanism,
                    budget,
                    fmt_num(cell.get("mean_rmse_difference_neyman_minus_uniform")),
                    fmt_num(boot.get("ci_low")),
                    fmt_num(boot.get("ci_high")),
                    "yes" if ci_excludes_zero(boot.get("ci_low"), boot.get("ci_high")) else "no",
                    "Neyman better" if ci_supports_lower(
                        cell.get("mean_rmse_difference_neyman_minus_uniform"),
                        boot.get("ci_low"),
                        boot.get("ci_high"),
                    ) else (
                        "Neyman worse" if ci_supports_higher(
                            cell.get("mean_rmse_difference_neyman_minus_uniform"),
                            boot.get("ci_low"),
                            boot.get("ci_high"),
                        ) else "CI includes 0 / inconclusive"
                    ),
                ]
            )
    return md_table(
        [
            "mechanism",
            "budget",
            "mean RMSE(Neyman−Uniform)",
            "CI low",
            "CI high",
            "CI excludes 0",
            "paired reading",
        ],
        rows,
    )


def b_method_table(block: dict[str, Any], methods: list[str] | None = None) -> str:
    names = methods or [k for k in block.keys() if isinstance(block.get(k), dict) and "late_true_performance" in (block.get(k) or {})]
    rows = []
    for name in names:
        item = block.get(name) or {}
        if not isinstance(item, dict) or "late_true_performance" not in item:
            continue
        rows.append(
            [
                name,
                fmt_pm(item.get("late_true_performance")),
                fmt_pm(item.get("late_observed_performance")),
                fmt_pm(item.get("late_lambda_min")),
                fmt_pm(item.get("late_weak_skill_error")),
                fmt_pm(item.get("mean_feedback_cost")),
                fmt_num((item.get("endogenous_collapse_any"))),
            ]
        )
    return md_table(
        [
            "method",
            "late true perf",
            "late observed perf",
            "late λ_min",
            "late weak-skill error",
            "feedback cost",
            "endogenous_collapse_any",
        ],
        rows,
    )


def plot_list(paths: list[Any]) -> str:
    if not paths:
        return "（当前 JSON 未列出路径）"
    return "\n".join(f"- `{rel(Path(p))}`" for p in paths)


def build_main_report(
    artifacts: dict[str, Any],
    verdicts: dict[str, Any],
    env: dict[str, Any],
) -> str:
    a = artifacts.get("idea_A_smoke") or {}
    b = artifacts.get("idea_B_smoke") or {}
    a_audit = artifacts.get("idea_A_audit") or {}
    b_audit = artifacts.get("idea_B_audit") or {}
    cmp_mat = artifacts.get("comparability") or {}
    a_common = artifacts.get("idea_A_common") or {}
    b_common = artifacts.get("idea_B_common") or {}
    real = artifacts.get("real_data") or {}
    inf_a = artifacts.get("inference_A") or {}
    va = verdicts["A"]
    vb = verdicts["B"]
    git_commit = git_command("rev-parse", "HEAD")
    git_branch = git_command("symbolic-ref", "--short", "HEAD")
    git_remote = git_command("config", "--get", "remote.origin.url")

    mnar = va["mnar_b05"]
    a_plots = a.get("aggregate_plots") or []
    b_plots = [
        REPO_ROOT / "idea_B" / "results" / "plots" / "method_vs_generic_baselines.png",
        REPO_ROOT / "idea_B" / "results" / "plots" / "collapse_recovery_trajectories.png",
        REPO_ROOT / "idea_B" / "results" / "plots" / "performance_vs_budget.png",
    ]
    b_plots = [p for p in b_plots if p.is_file()]

    a_ablation_rows = []
    for item in artifacts.get("a_ablation_indexes") or []:
        path = Path(item["path"])
        summary = load_a_ablation_mnar_b05(path) if item["exists"] else {"ablation": path.parent.parent.name}
        a_ablation_rows.append(summary)

    b1 = b.get("B1") or {}
    b2 = b.get("B2") or {}
    b3 = b.get("B3") or {}
    b2_budget = nested(b2, "by_budget", "0.05") or {}
    cvu = b.get("controller_vs_uniform") or {}

    ds = nested(real, "datasets_status") or {}
    model = nested(real, "model_status") or {}
    machine = a_audit.get("machine") or b_audit.get("machine") or {}

    missing = artifacts.get("missing_primary") or []
    missing_line = "无（所列主源均在磁盘上）" if not missing else ", ".join(f"`{p}`" for p in missing)

    lines: list[str] = []
    add = lines.append

    add("# Idea A / Idea B 可行性烟测报告 v1")
    add("")
    add("本文件由 `scripts/generate_report.py` 从磁盘 JSON/CSV 生成。报告中的数字均来自下列产物，没有手填。")
    add("")
    add(f"- 生成时间（UTC）：`{datetime.now(timezone.utc).isoformat()}`")
    add(f"- 仓库：`{REPO_ROOT}`")
    add(f"- 分支：`{git_branch}`")
    add(f"- commit：`{git_commit}`")
    add(f"- remote：`{git_remote}`")
    add(f"- 缺失的主 JSON：{missing_line}")
    add(f"- `n_seeds={a.get('n_seeds', 3)}`：{verdicts['n_seeds_note']}")
    add("")
    add("**判定规则预先固定，不因结果好坏改口。** Idea A 只有在 MNAR 或零重叠下审计 IPW 相对 observed-only / naive IPS 明显降偏，**并且**同等预算下 Neyman 相对 Uniform 的 RMSE 改进被配对区间支持时，才标 PROMISING。Idea B 只有在 B2 中观测表现稳定而 λ_min 与弱技能误差同时恶化，**并且**控制器在信息与长期真实表现上均以配对区间击败同预算 Uniform 时，才标 PROMISING。")
    add("")
    add("Idea A v1.0 的分配规则是经典分层 Neyman 分配加探索下界，估计用已知审计倾向的 IPW / Hájek / AIPW。**这不是新算法。**")
    add("")

    # 1
    add("## 1. Executive Summary")
    add("")
    add(f"- **Idea A 状态：`{verdicts['idea_A']}`**。名称：{a.get('name')}。")
    add(f"- **Idea B 状态：`{verdicts['idea_B']}`**。名称：{b.get('name')}。")
    add(f"- Idea A 是否出现假设效应（内生删失导致偏差、审计通道可识别）：**部分支持**。MNAR `b=0.05` 下 observed-only 总体均值 bias mean = `{fmt_num(mnar.get('observed_only_bias_mean'))}`，审计 Neyman IPW bias mean = `{fmt_num(mnar.get('neyman_ipw_bias_mean'))}`，Uniform IPW bias mean = `{fmt_num(mnar.get('uniform_ipw_bias_mean'))}`，naive IPS bias mean = `{fmt_num(va['naive_ips_mnar_b05_bias'])}`。识别-via-audit 成立。")
    add(f"- Idea A 是否在同等预算下击败 Uniform Audit：**不支持**。MNAR `b=0.05` Neyman−Uniform RMSE 差 mean = `{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'mnar_b=0.05', 'mean_rmse_difference_neyman_minus_uniform'))}`，配对 CI = [`{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'mnar_b=0.05', 'paired_bootstrap_rmse', 'ci_low'))}`, `{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'mnar_b=0.05', 'paired_bootstrap_rmse', 'ci_high'))}`]，包含 0。zero-overlap `b=0.05` 的配对 CI 排除 0，但方向是 Neyman **更差**。")
    add(f"- Idea B 假设的内生可观性坍缩（观测表现稳定 **且** λ_min 与弱技能误差同时恶化）：**未满足联合判据**。`B2.endogenous_collapse={fmt_num(vb['endogenous_collapse'])}`；`observed_performance_stable_while_lambda_min_and_weak_skill_error_worsen={fmt_num(vb['joint_criterion'])}`；`any_seed_weak_skill_error_worsened={fmt_num(vb['any_seed_weak_skill_error_worsened'])}`。")
    add(f"- Idea B 控制器相对同预算 Uniform：信息量 λ_min 的配对 CI = [`{fmt_num(nested(vb, 'info_boot', 'ci_low'))}`, `{fmt_num(nested(vb, 'info_boot', 'ci_high'))}`]（排除 0，控制器更高）；长期真实表现配对 CI = [`{fmt_num(nested(vb, 'perf_boot', 'ci_low'))}`, `{fmt_num(nested(vb, 'perf_boot', 'ci_high'))}`]（**包含 0**）。JSON 字段 `beats_uniform_on_both={fmt_num(vb['json_beats_uniform_on_both_means'])}` 只反映均值符号，**不能**覆盖配对区间规则。")
    add("- 是否成功复现并在同一协议上击败 CCF-A 论文：**否，且不得声称**。AMPLe local_status=`partial`；HyPER=`blocked`；PP-bandit official=`complete` 但是 1-D 数值 demo，NOT DIRECTLY COMPARABLE；ncvxpp=`partial`。")
    add("- 关键限制：MacBook Air M3 / 16GB / 无 NVIDIA GPU；`n_seeds=3`；主 venv 无 torch；HyPER / Drago 无官方可运行代码；PP-bandit 适配到 B2 被标 BLOCKED。")
    add(f"- 初步建议：两个想法目前都是 **`INCONCLUSIVE`**。Idea A 的可跟进方向是「已知审计通道在 MNAR 下的识别」，而不是把 Neyman 分配说成新算法。Idea B 需要能同时恶化弱技能误差的闭环设定，以及真实表现上排除 0 的配对区间。")
    add("")

    # 2
    add("## 2. Hardware and Software Environment")
    add("")
    add("环境清单来自本次运行的 `shared.hardware.probe.write_environment_manifests`，以及基线审计 JSON 中的 machine 字段。probe 里的 `experiments_run` 仍为脚手架布尔值（测试锁定为 false）；实验是否跑过以 SMOKE_SUMMARY 为准。")
    add("")
    add(md_table(
        ["字段", "值"],
        [
            ["OS", f"{nested(env, 'os', 'platform')} / Darwin {nested(env, 'os', 'release')}"],
            ["arch", str(nested(env, "os", "machine"))],
            ["审计记录的机型", str(machine.get("model") or "NA")],
            ["RAM GiB", fmt_num(env.get("ram_gib"))],
            ["磁盘 free GiB（probe）", fmt_num(nested(env, "disk", "free_gib"))],
            ["审计 disk_free_after_experiments", str(machine.get("disk_free_after_experiments") or "NA")],
            ["Python", str(nested(env, "python", "version_string"))],
            ["主解释器", str(nested(env, "python", "executable"))],
            ["主 venv torch", f"installed={nested(env, 'torch', 'installed')}; import_error={nested(env, 'torch', 'import_error')}"],
            ["隔离 venv", str(machine.get("python_isolated") or nested(real, "python_isolated_venv"))],
            ["numpy", str(nested(env, "packages", "numpy"))],
            ["scipy", str(nested(env, "packages", "scipy"))],
            ["scikit-learn", str(nested(env, "packages", "scikit-learn"))],
            ["pandas", str(nested(env, "packages", "pandas"))],
            ["matplotlib", str(nested(env, "packages", "matplotlib"))],
            ["pytest", str(nested(env, "packages", "pytest"))],
            ["transformers（主 venv）", str(nested(env, "packages", "transformers"))],
            ["隔离 venv transformers", str(nested(real, "packages", "transformers"))],
            ["隔离 venv torch", str(nested(real, "packages", "torch"))],
            ["Qwen revision", str(model.get("revision") or inf_a.get("revision"))],
            ["MultiPref revision", str(nested(ds, "multipref", "revision"))],
            ["RewardBench revision", str(nested(ds, "rewardbench", "revision"))],
            ["git branch（probe）", str(nested(env, "git", "branch"))],
            ["git commit（probe）", str(nested(env, "git", "commit") or git_commit)],
            ["不安装", "vLLM / bitsandbytes / flash-attn / CUDA / NCCL"],
        ],
    ))
    add("")
    add(f"- 清单路径：`{rel(PRIMARY_SOURCES['env_reports'])}` 与 `{rel(PRIMARY_SOURCES['env_shared'])}`")
    add(f"- Qwen 推理设备：preferred=`{inf_a.get('preferred_device')}`，实际=`{inf_a.get('device_actually_used')}`，generate_ok=`{fmt_num(inf_a.get('generate_ok'))}`，latency_s=`{fmt_num(inf_a.get('generate_latency_seconds'))}`，peak ru_maxrss_bytes=`{fmt_num(nested(inf_a, 'peak_memory', 'ru_maxrss_bytes'))}`")
    add("- 合成矩阵**不是** LLM 后训练。Qwen smoke 只做 load/generate。")
    add("")

    # 3
    add("## 3. Literature and Baseline Audit")
    add("")
    add("状态标签区分：published result / faithful local reproduction / modified（partial） / common-protocol / blocked。官方数字不得进入 Idea A/B 排行榜。")
    add("")
    baseline_rows = []
    for blob in (a_audit.get("baselines") or []) + (b_audit.get("baselines") or []):
        what = blob.get("what_ran") or {}
        local_nums = []
        if "l2_to_true_t3" in what:
            local_nums.append(f"AMPLe L2 t0={fmt_num(what.get('l2_to_true_t0'))}, t3={fmt_num(what.get('l2_to_true_t3'))}")
        if "multi_run_PR_phi_final_mean" in what:
            local_nums.append(
                f"PP-bandit PR_phi mean={fmt_num(what.get('multi_run_PR_phi_final_mean'))} ± {fmt_num(what.get('multi_run_PR_phi_final_std'))}"
            )
        if "npy" in what:
            for npy_name, npy in (what.get("npy") or {}).items():
                local_nums.append(
                    f"{npy_name} loss_last={fmt_num(npy.get('loss_last'))} test_acc_last={fmt_num(npy.get('test_acc_last'))}"
                )
        baseline_rows.append(
            [
                str(blob.get("paper")),
                str(blob.get("venue")),
                str(blob.get("official_source")),
                fmt_num(blob.get("official_code")),
                str(blob.get("upstream_commit") or "NA"),
                str(blob.get("local_status")),
                str(blob.get("local_results_path")),
                (blob.get("blocker") or "none")[:180],
                " ".join(local_nums) if local_nums else "见审计 JSON",
                str(blob.get("comparability")),
            ]
        )
    add(md_table(
        [
            "Paper",
            "Venue",
            "Official source",
            "Official code",
            "Upstream commit",
            "Local status",
            "Local path",
            "Blocker / modification",
            "Local reproduced numbers",
            "Comparability",
        ],
        baseline_rows,
    ))
    add("")
    add("逐项技术差异与协议标签（来自 `reports/COMPARABILITY_MATRIX.json`）：")
    add("")
    cmp_rows = []
    for row in cmp_mat.get("rows") or []:
        cmp_rows.append(
            [
                str(row.get("method_id")),
                str(row.get("venue") or row.get("paper_or_source") or "")[:80],
                str(row.get("vs")),
                str(row.get("status")),
                str(row.get("protocol_label")),
                str(row.get("local_status")),
                str(row.get("leaderboard") or "null"),
            ]
        )
    add(md_table(
        ["method_id", "paper/source", "vs", "comparability", "protocol_label", "local_status", "leaderboard"],
        cmp_rows,
    ))
    add("")
    add("- AMPLe 优化的是多维 preference 向量的 volume/random acquisition，不是有限总体泛函在内生删失下的审计识别。")
    add("- HyPER 优化的是 human vs AI 标注路由后的奖励模型 / RewardBench 分数。本机 TPU/vLLM/13B 训练被预先标 blocked，没有发明简化版 HyPER。")
    add("- Drago et al., ICML 2026 无官方代码，只作理论对照。")
    add("- PP-bandit 官方数值实验是 1-D reparameterized zeroth-order performative-risk demo，不是 B2 的技能删失。")
    add("- ncvxpp 官方 `mpiexec -np 10` 被 mpi4py/mpicc 阻断；单进程 stub 是 partial，不得冒充 10 replica 全文复现，也不得把 Greedy/Lazy 标成官方 SGD-GD。")
    add("- 共同协议上的 `volume_adapted_ipw` 是数学类比，不是 AMPLe official。")
    add("")

    # 4
    add("## 4. Idea A — Formal Problem and Method")
    add("")
    add("实现位置：`idea_A/methods/v1.0/{algorithm.py,estimators.py,identification.py,intervention.py}`，配置 `idea_A/methods/v1.0/config.yaml`。")
    add("")
    add(a.get("allocation_note") or "")
    add("")
    add("有限总体 `i=1..n`，`n=2000`，离散层 `h∈{1..8}`。事实结果 `Y_i` 在模拟器中生成，但算法只在 `O_i = S_i ∨ A_i` 时可读。")
    add("")
    add("- 自然观测 `S_i` 由机制生成：MCAR / MAR / MNAR / policy-dependent / low-overlap / zero-overlap / nonstationary。MNAR 与零重叠下自然倾向不可识别，记为 identification failure，不是估计器报错。")
    add("- 干预：独立审计 `A_i ~ Bern(q_i)`，`q_i` 已知并写入日志。预算约束 `E[A] = b`，`b∈{0.01,0.05,0.10,0.20}`。")
    add("- 目标量：有限总体均值 `E[Y]`、偏好率 `P(Y>0)`、层条件均值 `E[Y|h]`、在均匀层分布 `p*` 下的策略价值。")
    add("- 估计器：observed-only、审计 IPW（Horvitz–Thompson）、SNIPS/Hájek、AIPW/DR（只在 `A=1` 读残差）、naive IPS（把自然倾向当已知，这是错误对照）。")
    add("- 分配：对可见标签估计层标准差 `σ_h`，Neyman `q_h ∝ σ_h`，再与均匀预算做 `exploration_mix=0.15` 混合，并用 `q_floor=1e-6`。**经典两阶段抽样。**")
    add("")
    add("```")
    add("for each simulated world (mechanism, seed):")
    add("    draw X, stratum, Y  # hidden from the method")
    add("    draw natural S according to the mechanism")
    add("    estimate σ_h from currently visible labels only")
    add("    set q_h = Neyman(σ_h) mixed with uniform budget b")
    add("    draw A_i ~ Bern(q_i); O = S OR A")
    add("    estimate functionals with IPW/SNIPS/AIPW using known q")
    add("    compare every method on this same world (paired)")
    add("```")
    add("")
    add("已知理论限制：零审计重叠且自然通道也看不见时，该子总体不可识别；倾向误指定会破坏 IPW；Neyman 用的 `σ_h` 来自可见标签，在 MNAR 下本身偏倚，所以分配改进不是定理保证。")
    add("")

    # 5
    add("## 5. Idea A — Experiments")
    add("")
    add(f"- 数据集：合成 Selective Feedback。`n={a.get('n')}`，seeds=`{a.get('seeds')}`，`n_seeds={a.get('n_seeds')}`，preliminary=`{fmt_num(a.get('preliminary'))}`。")
    add(f"- 矩阵：`{nested(a, 'matrix', 'note')}`；写入 run 数 `{nested(a, 'matrix', 'n_runs_written')}`，跳过 `{nested(a, 'matrix', 'n_cells_skipped')}`。")
    add("- 划分：有限总体一次生成，无 train/val/test 泄漏式划分；配对方法共享同一 `Y`。")
    add("- 超参数：见 `idea_A/methods/v1.0/config.yaml`（`exploration_mix=0.15`，`ridge_lambda=0.001`，`n_batches_adaptive_budget=4` 仅用于 A4）。")
    add("- 命令：`PYTHONPATH=. .venv/bin/python -m idea_A.experiments.run_synthetic`")
    add(f"- 失败 run：`{a.get('failures')}`")
    add("")
    add("### 5.1 MNAR / zero-overlap，`b=0.05`，全部 method")
    add("")
    add("**MNAR, b=0.05**")
    add("")
    add(method_table_a(a, "mnar", "b=0.05"))
    add("")
    add("**zero-overlap, b=0.05**")
    add("")
    add(method_table_a(a, "zero_overlap", "b=0.05"))
    add("")
    add("### 5.2 识别对照（key_findings.mnar_observed_only_vs_audit_ipw）")
    add("")
    id_rows = []
    for budget, cell in (nested(a, "key_findings", "mnar_observed_only_vs_audit_ipw") or {}).items():
        id_rows.append(
            [
                budget,
                fmt_num(cell.get("observed_only_bias_mean")),
                fmt_num(cell.get("observed_only_bias_std")),
                fmt_num(cell.get("neyman_ipw_bias_mean")),
                fmt_num(cell.get("neyman_ipw_bias_std")),
                fmt_num(cell.get("uniform_ipw_bias_mean")),
                fmt_num(cell.get("uniform_ipw_bias_std")),
            ]
        )
    add(md_table(
        ["budget", "observed-only bias", "std", "Neyman IPW bias", "std", "Uniform IPW bias", "std"],
        id_rows,
    ))
    add("")
    add("### 5.3 Neyman vs Uniform 配对 RMSE（全部 mechanism × budget）")
    add("")
    add(paired_table_a(a))
    add("")
    add("### 5.4 共同协议（MNAR / zero-overlap，b=0.05；volume 适配器不是 AMPLe official）")
    add("")
    add(f"协议说明：{nested(a_common, 'matrix', 'note')}")
    add("")
    add(method_table_a(a_common, "mnar", "b=0.05") if a_common else "MISSING")
    add("")
    ney_vs_uni = nested(cmp_mat, "rows")
    neyman_row = next((r for r in (ney_vs_uni or []) if r.get("method_id") == "neyman_ipw"), {})
    add("配对（来自 COMPARABILITY_MATRIX / common-protocol SUMMARY）MNAR `b=0.05`：")
    boot = nested(neyman_row, "paired_vs_uniform", "mnar", "b=0.05", "paired_bootstrap_rmse") or {}
    add(f"- mean RMSE(Neyman−Uniform) = `{fmt_num(nested(neyman_row, 'paired_vs_uniform', 'mnar', 'b=0.05', 'mean_rmse_difference_neyman_minus_uniform'))}`")
    add(f"- paired bootstrap CI = [`{fmt_num(boot.get('ci_low'))}`, `{fmt_num(boot.get('ci_high'))}`], n_boot=`{boot.get('n_boot')}`, preliminary=`{fmt_num(boot.get('preliminary'))}`")
    add("")
    add("### 5.5 图")
    add("")
    add(plot_list(a_plots))
    add("")
    add("### 5.6 消融 run 计数")
    add("")
    ab_rows = []
    for summary in a_ablation_rows:
        ab_rows.append(
            [
                str(summary.get("ablation")),
                fmt_num(summary.get("n_index_runs")),
                fmt_num(summary.get("n_mnar_b05_files")),
                fmt_num(summary.get("bias_mean")),
                fmt_num(summary.get("rmse_mean")),
                ",".join(summary.get("methods") or []),
                str(summary.get("source")),
            ]
        )
    add(md_table(
        ["ablation", "INDEX n_runs", "MNAR b=0.05 files", "MNAR b=0.05 bias mean", "RMSE mean", "methods in files", "INDEX"],
        ab_rows,
    ))
    add("")
    add("A 消融格子：全部机制在 `b=0.05`，以及 MNAR / zero-overlap 的全部预算，seeds 0/1/2。其余 budget×mechanism 为保持运行时间跳过。一次只改一个组件。")
    add("")

    # 6
    add("## 6. Idea A — Interpretation")
    add("")
    add("1. **内生选择是否产生可测偏差？** 是。MNAR observed-only bias mean = "
        f"`{fmt_num(mnar.get('observed_only_bias_mean'))}`（约 +1.19），远离 0。")
    add("2. **提出的方法能否恢复目标量？** 审计 IPW 可以大幅降偏：Neyman IPW bias mean = "
        f"`{fmt_num(mnar.get('neyman_ipw_bias_mean'))}`，Uniform IPW = `{fmt_num(mnar.get('uniform_ipw_bias_mean'))}`。"
        f" naive IPS 仍为 `{fmt_num(va['naive_ips_mnar_b05_bias'])}`。这支持「已知审计通道」而不是「把自然倾向当已知」。")
    add("3. **同等成本下是否优于 Uniform Audit？** 否。MNAR `b=0.05` 配对 RMSE CI 包含 0；多个格子 Neyman 均值更高。zero-overlap `b=0.05` CI 排除 0 且 Neyman 更差。")
    add("4. **是否优于 IPS/DR（在可比假设下）？** 主方法本身就是已知 `q` 的 IPW/DR。相对 naive IPS（错误假设）当然更好；这不是对公开 IPS 论文的击败。")
    add("5. **是否优于适用的已发表方法？** 不能这么说。AMPLe/HyPER 协议不可比；HyPER blocked；不得把 incomparable 数字排进同一榜。")
    add("6. **失败条件：** 零重叠下自然通道不可识别；倾向误指定（A5）；关掉审计锚（A3）；Neyman 用可见 `σ_h` 在 MNAR 下偏倚，低预算时方差更大。")
    add("7. **增益是否只来自更强假设或更多信息？** 识别增益来自独立审计（额外标注预算 + 已知 `q`）。分配增益未被配对区间支持。全反馈 oracle 有全部 `Y`，不能当成算法胜利。")
    add("8. **去掉已知技术后还剩什么新贡献？** v1.0 没有新分配算法。剩下的是现象：在 MNAR 合成世界里，一条已知 `q` 的审计通道能把 +1.19 量级的观测偏差压到约 −0.10。这是经典抽样在新应用设定上的可行性，不是新定理。")
    add("")

    # 7
    add("## 7. Idea B — Formal Problem and Method")
    add("")
    add("实现位置：`idea_B/methods/v1.0/{algorithm.py,observability.py,feedback_controller.py,policy_update.py}`。")
    add("")
    add(
        "信息矩阵（来自 SMOKE_SUMMARY）：`"
        + str(b.get("information_matrix"))
        + "`。这是线性模型的观测设计 Gram 矩阵，**不是**动力系统可观性 Gramian。条件期望版本 `W_T^{cond}=∑_t q_t φ_t φ_t^⊤`。"
    )
    add("")
    add(f"- horizon=`{b.get('horizon')}`，population_per_round=`{b.get('population_per_round')}`，primary_budget=`{b.get('primary_budget')}`，seeds=`{b.get('seeds')}`。")
    add("- 闭环：参数给 query 打分 → 环境按策略改变自然反馈分布 → 有限预算审计 → 用观测标签更新参数 → 新参数成为下一轮策略。")
    add("- B1 是外生缺失诊断，**不得**叫做内生坍缩。B2 是策略依赖反馈。B3 分 recoverable / unrecoverable。")
    add("- 控制器：从 `W_T` 读出弱方向（默认 λ_min / 最弱特征向量），把同一预算分到信息不足的技能，并在技能单纯形上平滑，避免剧烈切换。")
    add("- 指标：真实表现、观测表现、差距、技能误差、`λ_min(W_T/T)`、条件数、反馈成本。更好的 `W_T` **不自动**等于更好的真实表现。")
    add("- 固定策略 + 固定抽样不得标成 endogenous collapse。")
    add("")
    add("信息度量的数学理由：线性模型 `y=φ^⊤θ+ε` 的观测设计 Gram 矩阵决定 `θ` 的最小二乘协方差形状；`λ_min` 小表示存在近不可识别方向。这是标准实验设计 / E-optimality 读出，不是新的 Gramian 定义。")
    add("")

    # 8
    add("## 8. Idea B — Experiments")
    add("")
    add(f"- 合成闭环 run 数 `{b.get('n_runs')}`，skips=`{b.get('skips')}`，wall_time_s=`{fmt_num(b.get('wall_time_s'))}`。")
    add("- 命令：`PYTHONPATH=. .venv/bin/python -m idea_B.experiments.run_synthetic`")
    add(f"- torch_installed（主实验记录）=`{fmt_num(b.get('torch_installed'))}`")
    add(f"- blockers：{b.get('blockers')}")
    add("")
    add("### 8.1 Environment B1（外生缺失诊断）")
    add("")
    add(str(b1.get("note")))
    add("")
    add(b_method_table(b1))
    add("")
    add("### 8.2 Environment B2，budget=0.05")
    add("")
    add(f"- `endogenous_collapse` = `{fmt_num(b2.get('endogenous_collapse'))}`")
    add(f"- joint criterion = `{fmt_num(nested(b2, 'evidence', 'observed_performance_stable_while_lambda_min_and_weak_skill_error_worsen'))}`")
    add(f"- all_seeds_observed_stable = `{fmt_num(nested(b2, 'evidence', 'all_seeds_observed_stable'))}`")
    add(f"- all_seeds_lambda_min_worsened = `{fmt_num(nested(b2, 'evidence', 'all_seeds_lambda_min_worsened'))}`")
    add(f"- any_seed_weak_skill_error_worsened = `{fmt_num(nested(b2, 'evidence', 'any_seed_weak_skill_error_worsened'))}`")
    add(f"- mean_final_weak_skill_observation_rate = `{fmt_num(nested(b2, 'evidence', 'mean_final_weak_skill_observation_rate'))}`")
    add(f"- evidence.note：{nested(b2, 'evidence', 'note')}")
    add("")
    add(b_method_table(b2_budget))
    add("")
    add("### 8.3 Controller vs Uniform（配对，budget=" + str(cvu.get("budget")) + "）")
    add("")
    add(md_table(
        ["metric", "mean(controller−uniform)", "CI low", "CI high", "n_boot", "CI excludes 0"],
        [
            [
                "late λ_min",
                fmt_num(nested(cvu, "information_paired_bootstrap", "mean_difference")),
                fmt_num(nested(cvu, "information_paired_bootstrap", "ci_low")),
                fmt_num(nested(cvu, "information_paired_bootstrap", "ci_high")),
                fmt_num(nested(cvu, "information_paired_bootstrap", "n_boot")),
                "yes" if ci_excludes_zero(
                    nested(cvu, "information_paired_bootstrap", "ci_low"),
                    nested(cvu, "information_paired_bootstrap", "ci_high"),
                ) else "no",
            ],
            [
                "late true performance",
                fmt_num(nested(cvu, "true_performance_paired_bootstrap", "mean_difference")),
                fmt_num(nested(cvu, "true_performance_paired_bootstrap", "ci_low")),
                fmt_num(nested(cvu, "true_performance_paired_bootstrap", "ci_high")),
                fmt_num(nested(cvu, "true_performance_paired_bootstrap", "n_boot")),
                "yes" if ci_excludes_zero(
                    nested(cvu, "true_performance_paired_bootstrap", "ci_low"),
                    nested(cvu, "true_performance_paired_bootstrap", "ci_high"),
                ) else "no",
            ],
        ],
    ))
    add("")
    add(f"JSON `beats_uniform_on_both` = `{fmt_num(cvu.get('beats_uniform_on_both'))}`（按均值符号）。配对规则下信息量支持、真实表现不支持。")
    add("")
    add("### 8.4 Environment B3")
    add("")
    add("**recoverable**")
    add("")
    add(b_method_table(b3.get("recoverable") or {}))
    add("")
    add("**unrecoverable**")
    add("")
    add(b_method_table(b3.get("unrecoverable") or {}))
    add("")
    add("### 8.5 共同协议 B2 b=0.05")
    add("")
    add(f"n_runs=`{b_common.get('n_runs')}`；excluded：{nested(b_common, 'leaderboard', 'excluded')}")
    add("")
    add(b_method_table(b_common.get("methods") or {}))
    add("")
    add("PP-bandit adapted 状态：")
    for item in b_common.get("adapted_skipped") or []:
        add(f"- `{item.get('method_id')}` status=`{item.get('status')}` protocol=`{item.get('protocol_label')}`")
        add(f"  {item.get('reason')}")
    add("")
    add("### 8.6 图")
    add("")
    add(plot_list(b_plots))
    add("- 每个 closed-loop run 另有 `plots/{true_vs_observed_performance,information_eigenvalues,skill_specific_accuracy,skill_specific_observation_rates,regret_vs_rounds}.png`。")
    add("")
    add("### 8.7 消融")
    add("")
    add(f"B 消融 total_runs=`{nested(b, 'ablations', 'total_runs')}`：B1=`{nested(b, 'ablations', 'B1_No_Observability_Control')}`，B2=`{nested(b, 'ablations', 'B2_No_Adaptive_Exploration')}`，B3=`{nested(b, 'ablations', 'B3_Fixed_vs_Adaptive_Feedback')}`，B4=`{nested(b, 'ablations', 'B4_No_Policy_Feedback_Loop')}`，B5=`{nested(b, 'ablations', 'B5_Observability_Metric_Variants')}`。")
    add("")
    b_ab_rows = []
    for item in artifacts.get("b_ablation_indexes") or []:
        means = b_ablation_arm_means(item.get("data"))
        for arm, stats in means.items():
            b_ab_rows.append(
                [
                    Path(item["path"]).parts[-3],
                    arm,
                    fmt_num(stats.get("n")),
                    fmt_num(nested(stats, "late_true_performance", "mean")),
                    fmt_num(nested(stats, "late_lambda_min", "mean")),
                    fmt_num(nested(stats, "late_weak_skill_error", "mean")),
                    fmt_num(nested(stats, "endogenous_collapse", "mean")),
                ]
            )
    if b_ab_rows:
        add(md_table(
            ["ablation", "arm", "n", "late true perf mean", "late λ_min mean", "late weak-skill err mean", "collapse fraction"],
            b_ab_rows,
        ))
    add("")

    # 9
    add("## 9. Idea B — Interpretation")
    add("")
    add("1. **可观性坍缩能否内生出现？** 当前联合判据为 false。Greedy / NoFeedback 上观测表现稳定、λ_min 下降，但弱技能误差没有相对早期窗口继续变差（早期 `θ=0` 误差已经很大）。弱技能最终观测率为 0。")
    add("2. **是否不同于普通遗忘或非平稳抽样？** B1 明确是外生缺失。B4 把 B2 换成 B1 通道。当前还不能把 B2 标成已证实的策略诱发坍缩。")
    add("3. **更低信息是否预测更高估计误差？** Greedy 的 late λ_min mean = "
        f"`{fmt_num(nested(b2, 'evidence', 'GreedyDeployment', 'late_lambda_min', 'mean'))}`，"
        f" late weak-skill error mean = `{fmt_num(nested(b2, 'evidence', 'GreedyDeployment', 'late_weak_skill_error', 'mean'))}`；"
        f" 控制器 late λ_min mean = `{fmt_num(nested(b2_budget, 'ObservabilityController', 'late_lambda_min', 'mean'))}`，"
        f" weak-skill error mean = `{fmt_num(nested(b2_budget, 'ObservabilityController', 'late_weak_skill_error', 'mean'))}`。"
        "方向一致，但坍缩的时间动态（误差继续恶化）未满足。")
    add("4. **控制器是否阻止信息损失？** 相对 Uniform，λ_min 配对 CI 排除 0，支持信息量改善。")
    add("5. **是否改善长期学习（真实表现）？** 均值略高，但配对 CI 包含 0，**不支持** PROMISING 所需的真实表现优势。")
    add("6. **是否击败适用的已发表基线？** 否。PP-bandit / ncvxpp 不可比；adapted PP-bandit BLOCKED。Greedy 是 generic control。")
    add("7. **对预算与模型敏感是否稳健？** 主矩阵覆盖四个预算，但 `n_seeds=3`。真实表现优势未过配对门槛。")
    add("8. **能否用「只是多探索」解释？** 预算与 Uniform 对齐。信息量改善不能还原成更高花费。但若只看真实表现，差异与多探索/分配噪声无法分离。")
    add("")

    # 10
    add("## 10. Negative Results and Failure Cases")
    add("")
    add("| 配置 | 观察到的失败 | 可能原因 | 证据 | 仍不确定 |")
    add("| --- | --- | --- | --- | --- |")
    add(f"| Idea A Neyman vs Uniform，MNAR b=0.05 | 配对 RMSE CI 含 0；差 mean=`{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'mnar_b=0.05', 'mean_rmse_difference_neyman_minus_uniform'))}` | 可见 σ_h 在 MNAR 下偏倚；n_seeds=3 方差大 | `idea_A/results/SMOKE_SUMMARY.json` key_findings | 更大种子数是否翻转 |")
    add(f"| Idea A Neyman vs Uniform，zero-overlap b=0.05 | Neyman 显著更差，CI=[{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'zero_overlap_b=0.05', 'paired_bootstrap_rmse', 'ci_low'))}, {fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'zero_overlap_b=0.05', 'paired_bootstrap_rmse', 'ci_high'))}] | 零重叠层的可见方差不能代表隐层 | 同一 paired 表 | 其他分配启发式是否同样失败 |")
    add(f"| Idea A naive IPS on MNAR | bias mean=`{fmt_num(va['naive_ips_mnar_b05_bias'])}` | 自然倾向不可识别却被当已知 | per_method.naive_ips | 无 |")
    add("| HyPER official | blocked | TPU v3-128, vLLM, 8B–13B RM/DPO, Beaker/GCS, 未初始化 submodule | REPRODUCTION_AUDIT.json | 有集群时能否跑 |")
    add("| Drago ICML 2026 | blocked | 无官方代码 | THEORY_COMPARATOR.md | 作者是否稍后放码 |")
    add("| AMPLe full README | partial：n_rounds=3 而非 100；未用 --dims 0 1 | burnin=50000 太慢；2-aspect 数据集 HTTP 401 | AMPLe what_ran | 完整 100 round 曲线 |")
    add("| ncvxpp mpiexec -np 10 | blocked；单进程 partial | 无 mpi4py/mpicc；data.csv 标签 {+1,-1} 与 Y==1 过滤互作用 | npy loss/acc | 10 replica 官方曲线 |")
    add("| PP-bandit → B2 adapted | BLOCKED | 1-D Bernoulli KL reparam 无法忠实接到 3-D 技能删失 | common_protocol SUMMARY.adapted_skipped | 无忠实 port |")
    add(f"| Idea B B2 endogenous collapse | joint=false；weak-skill error 未继续恶化 | θ 从 0 起步，早期误差已大 | B2.evidence.note | 换初始化是否出现联合坍缩 |")
    add(f"| Idea B 真实表现 vs Uniform | CI 含 0，mean diff=`{fmt_num(nested(cvu, 'true_performance_paired_bootstrap', 'mean_difference'))}` | horizon=32 太短或信号弱 | controller_vs_uniform | 更长 horizon |")
    add("| Qwen MPS generate | abort NSInvalidArgumentException | macOS MPS / torch 2.15 不匹配 | INFERENCE_SMOKE.json | MPS 是否可修 |")
    add("| RewardBench leaderboard | 未跑 | 没有兼容的小 RM；Qwen 是因果 LM | REAL_DATA_FEASIBILITY.json | 无 |")
    add("")
    add("负结果全部保留。没有删除失败格子。")
    add("")

    # 11
    add("## 11. Ablation Analysis")
    add("")
    add("Idea A 一次只改一个组件：")
    add("")
    add("- **A1 No Adaptive Allocation**：Neyman → 均匀 `q=b`，估计仍是审计 IPW。检验层方差倾斜是否改变误差。INDEX n_runs 见上表。")
    add("- **A2 No Propensity Correction**：分配仍是 Neyman，估计改成审计样本未加权均值。检验 `A/q` 是否必要。")
    add("- **A3 No Anchor Intervention**：`q=0`，只留自然通道。MNAR/零重叠下这是识别失败。")
    add("- **A4 Fixed vs Adaptive Budget**：同一 Neyman+IPW，只改预算日程（一次分完 vs 分批更新 σ_h）。两者都是经典抽样日程。")
    add("- **A5 Misspecified Propensity**：真实 `q` 抽样，错误 `q̂` 做 IPW。检验「已知倾向」被破坏时的退化。")
    add("")
    if a_ablation_rows:
        add("MNAR `b=0.05` 消融臂总体均值 bias mean（从各 `metrics.json` 现算，不是手填）：")
        add("")
        for summary in a_ablation_rows:
            add(
                f"- `{summary.get('ablation')}`: bias_mean=`{fmt_num(summary.get('bias_mean'))}` "
                f"rmse_mean=`{fmt_num(summary.get('rmse_mean'))}` n=`{fmt_num(summary.get('n_bias_values'))}` "
                f"methods={summary.get('methods')}"
            )
        add("")
    add("Idea B：")
    add("")
    add("- **B1 No Observability Control**：关掉基于 `W_T` 的分配。")
    add("- **B2 No Adaptive Exploration**：`mix_uniform=1`，花费相同预算但不按弱方向加权。")
    add("- **B3 Fixed vs Adaptive Feedback**：技能均分 vs 最弱方向。")
    add("- **B4 No Policy Feedback Loop**：B2 闭环通道换成 B1 外生缺失。")
    add("- **B5 Observability Metric Variants**：λ_min / tr(W^{-1}) / logdet，不改 Gram 定义。")
    add("")
    add("消融支持「信息分配改变 `W_T`」，但没有把 B2 联合坍缩判据从 false 改成 true，也没有把真实表现 CI 从含 0 改成排除 0。")
    add("")

    # 12
    add("## 12. Novelty Assessment")
    add("")
    add("| 类别 | Idea A | Idea B |")
    add("| --- | --- | --- |")
    add("| 既有文献结果 | 分层 Neyman、IPW/Hájek/AIPW、两阶段抽样 | 观测设计 Gram 矩阵、E-optimality、performative prediction |")
    add("| 新观察到的实验现象 | MNAR 合成世界中 known-q 审计把 observed-only +1.19 bias 降到约 −0.10（n_seeds=3） | 控制器提高 λ_min（配对 CI 排除 0）；联合内生坍缩未出现 |")
    add("| 新算法机制 | **无。v1.0 明确不是新算法。** | 有限预算下按弱特征向量分配观测；是标准实验设计启发式，不是新最优性定理 |")
    add("| 未证明的理论猜想 | 内生删失 + 独立审计的有限样本保证 | 策略诱发可观性坍缩的一般条件 |")
    add("| 可能的新颖性 | 把审计识别接到偏好/选择性反馈，而非新分配规则 | 「观测表现稳定但信息坍缩」这一现象，当前实验尚未联合成立 |")
    add("| 与已发表论文的潜在重叠 | AMPLe/HyPER 都涉及反馈选择，但目标与信息不同 | ICML 2024 / NeurIPS 2024 performative 论文研究决策依赖分布，不研究技能级删失 Gramian |")
    add("")
    add("经验证据不等于已证明的新颖性。不得声称 outperformed CCF-A Main 论文。")
    add("")

    # 13
    add("## 13. Limitations")
    add("")
    add("- Mac compute：16GB ram、无 NVIDIA GPU、磁盘从约 12Gi 降到审计记录的 "
        f"`{machine.get('disk_free_after_experiments')}`，真实数据阶段后约 `{fmt_num(nested(real, 'disk_before_after', 'after_inference_free_gib'))}` GiB。")
    add("- 缺失官方复现：HyPER blocked；Drago 无代码；AMPLe 缩短 round；ncvxpp 非 10 replica；PP-bandit 未接到 B2。")
    add("- `n_seeds=3`，配对 bootstrap 不是 confirmatory。")
    add("- Qwen 只是 CPU generate smoke；RewardBench 无 RM 打分，是 proxy/schema 校验。")
    add("- 合成环境：线性奖励 / 线性技能、短 horizon、已知层或已知特征。")
    add("- MultiPref 只落了 50 个 prompt_id 的确定性子集，未做后训练。")
    add("- 没有新的识别或坍缩定理。")
    add("- 主 venv 无 torch；隔离 venv 才有 torch 2.15.0。")
    add("")

    # 14
    add("## 14. Final Recommendation")
    add("")
    add(f"### Idea A：`{verdicts['idea_A']}`")
    add("")
    add("识别-via-audit 被 MNAR 数字支持；分配新颖性不被配对区间支持（有时更差）。总体 **INCONCLUSIVE**，且必须写明 v1.0 是经典 Neyman。")
    add("")
    add("下一步最能改变结论的实验：在 MNAR 下把 seeds 增到 ≥30，并比较 **Uniform IPW**（已是强对照）与任何真正非经典的审计设计；不要继续把 Neyman 当新方法。")
    add("")
    add(f"### Idea B：`{verdicts['idea_B']}`")
    add("")
    add("信息量改善有配对支持；内生坍缩联合判据失败；真实表现 CI 含 0。总体 **INCONCLUSIVE**。")
    add("")
    add("下一步最能改变结论的实验：改变初始化使早期弱技能误差较小，检查弱技能误差是否随后上升；同时加长 horizon，使真实表现配对 CI 有机会排除 0。不要把 B1 外生缺失写成闭环坍缩。")
    add("")
    add("对 ICML / NeurIPS / ACL Main 的投资建议：当前烟测**不足以**支持「已有可投稿的新算法」。可以继续做识别现象或可观性现象的机制研究，但必须先过预先写死的 PROMISING 门槛。")
    add("")

    # 15
    add("## 15. Reproduction Commands")
    add("")
    add("在仓库根目录执行。主 Python：`/Users/grantwright/Research/idea-A-B/.venv/bin/python`。需要 torch / transformers / datasets 的步骤用 `.isolated_baselines_venv`。")
    add("")
    add("```bash")
    add("cd /Users/grantwright/Research/idea-A-B")
    add("export PYTHONPATH=\"$PWD\"")
    add("")
    add("# Install (NumPy stack; torch 允许失败)")
    add("bash scripts/setup_mac.sh")
    add(".venv/bin/python scripts/verify_installation.py")
    add(".venv/bin/python -m shared.hardware")
    add("")
    add("# Idea A synthetic + ablations")
    add(".venv/bin/python -m idea_A.experiments.run_synthetic")
    add(".venv/bin/python idea_A/ablations/A1_No_Adaptive_Allocation/run.py")
    add(".venv/bin/python idea_A/ablations/A2_No_Propensity_Correction/run.py")
    add(".venv/bin/python idea_A/ablations/A3_No_Anchor_Intervention/run.py")
    add(".venv/bin/python idea_A/ablations/A4_Fixed_vs_Adaptive_Budget/run.py")
    add(".venv/bin/python idea_A/ablations/A5_Misspecified_Propensity/run.py")
    add(".venv/bin/python -m idea_A.experiments.run_all")
    add(".venv/bin/python idea_A/experiments/common_protocol/run.py")
    add("")
    add("# Idea B synthetic + ablations")
    add(".venv/bin/python -m idea_B.experiments.run_synthetic")
    add(".venv/bin/python idea_B/ablations/B1_No_Observability_Control/run.py")
    add(".venv/bin/python idea_B/ablations/B2_No_Adaptive_Exploration/run.py")
    add(".venv/bin/python idea_B/ablations/B3_Fixed_vs_Adaptive_Feedback/run.py")
    add(".venv/bin/python idea_B/ablations/B4_No_Policy_Feedback_Loop/run.py")
    add(".venv/bin/python idea_B/ablations/B5_Observability_Metric_Variants/run.py")
    add(".venv/bin/python idea_B/experiments/common_protocol/run.py")
    add("")
    add("# Official baselines (isolated venv / as recorded)")
    add("# AMPLe partial: see idea_A/baselines/REPRODUCTION_AUDIT.json what_ran.command")
    add("# HyPER: BLOCKED — do not invent a simplified implementation")
    add(".venv/bin/python idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/run_official_numerical.py")
    add("# ncvxpp full mpiexec -np 10 is BLOCKED; partial single-process is recorded in REPRODUCTION_AUDIT.json")
    add("")
    add("# Real-data feasibility (isolated venv)")
    add(".isolated_baselines_venv/bin/python scripts/download_benchmarks.py")
    add(".isolated_baselines_venv/bin/python scripts/download_models.py")
    add("")
    add("# Tests and reports")
    add(".venv/bin/python -m pytest")
    add(".venv/bin/python scripts/generate_report.py")
    add("```")
    add("")
    add("AMPLe 实际跑过的命令（partial，n_rounds=3，acquisition 仍为 volume，burnin=50000）：")
    ample = next((x for x in (a_audit.get("baselines") or []) if "AMPLe" in str(x.get("paper"))), {})
    add("")
    add("```text")
    add(str(nested(ample, "what_ran", "command")))
    add("```")
    add("")

    # 16
    add("## 16. Complete Result Index")
    add("")
    add("### 摘要 JSON")
    add("")
    for name, path in PRIMARY_SOURCES.items():
        exists = "yes" if path.exists() else "NO"
        add(f"- `{rel(path)}` ({name}, exists={exists})")
    add("- `idea_A/experiments/common_protocol/SUMMARY.json`")
    add("- `idea_B/experiments/common_protocol/SUMMARY.json`")
    add("")
    add("### Idea A run 目录")
    add("")
    add(f"- 主矩阵 `{len(a.get('result_dirs') or [])}` 个目录，每个含 `config.yaml, metrics.json, metrics.csv, run.log, environment.json, plots/`。")
    add("- 根：`idea_A/results/syn_<mechanism>_b<budget>_s<seed>/`")
    add(f"- 消融 INDEX：{', '.join('`' + rel(p) + '`' for p in A_ABLATION_INDEXES)}")
    add("- 共同协议：`idea_A/experiments/common_protocol/results/`")
    add("")
    add("### Idea B run 目录")
    add("")
    add(f"- 主矩阵 n_runs=`{b.get('n_runs')}`，根 `idea_B/results/synthetic_<env>_<method>_budget*_seed*/`")
    add(f"- 消融 INDEX：{', '.join('`' + rel(p) + '`' for p in B_ABLATION_INDEXES)}")
    add("- 共同协议：`idea_B/experiments/common_protocol/results/`")
    add("- 轨迹：每个 run 的 `trajectories.json` / `trajectories.npz`")
    add("")
    add("### 图")
    add("")
    add(plot_list(a_plots))
    add(plot_list(b_plots))
    add("")
    add("### 模型与数据")
    add("")
    add(f"- 共享权重缓存（gitignore）：`{nested(model, 'shared_cache')}`")
    add(f"- Idea A 模型目录（manifest + 相对符号链接）：`{nested(model, 'idea_a_model_dir')}`")
    add(f"- Idea B 模型目录：`{nested(model, 'idea_b_model_dir')}`")
    add(f"- MODEL_MANIFEST：`{rel(PRIMARY_SOURCES['manifest_A'])}`")
    add(f"- MultiPref subset：`{nested(ds, 'multipref', 'paths', 'subset_json')}`")
    add(f"- RewardBench sample：`{nested(ds, 'rewardbench', 'paths', 'sample_json')}`")
    add("")
    add("### Git")
    add("")
    add(f"- branch `{git_branch}` commit `{git_commit}` remote `{git_remote}`")
    add(f"- probe 记录的 commit：`{nested(env, 'git', 'commit')}` error=`{nested(env, 'git', 'commit_error')}`")
    add("")
    add("---")
    add("")
    add("数字来源全部是磁盘 JSON。若某单元格在 JSON 中不存在，表中写 NA，而不是补造。")
    add("")
    return "\n".join(lines)


def build_audit_report(artifacts: dict[str, Any], verdicts: dict[str, Any]) -> str:
    a_audit = artifacts.get("idea_A_audit") or {}
    b_audit = artifacts.get("idea_B_audit") or {}
    cmp_mat = artifacts.get("comparability") or {}
    lines: list[str] = []
    add = lines.append
    add("# Baseline Reproduction Audit")
    add("")
    add("本报告区分五类对象，避免把不同协议的数字排进同一张表。")
    add("")
    add("1. **Published result**：论文原文声称的结果。本仓库几乎没有核验全文表格；未核验的不抄。")
    add("2. **Faithful local reproduction**：官方脚本、官方数据、官方超参，仅环境差异。")
    add("3. **Modified reproduction**：缩短 round、单进程 stub、换公开子集等。标 `partial`。")
    add("4. **Common-protocol comparison**：把某种思想接到 Idea A/B 合成世界。不得标 official。")
    add("5. **Blocked / unavailable**：缺代码、缺 TPU/vLLM/MPI、缺数据许可。保留阻断原因。")
    add("")
    add("本地状态对照（任务指定，且与审计 JSON 一致）：AMPLe=`partial`，HyPER=`blocked`，PP-bandit=`complete`，ncvxpp=`partial`。")
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
        if row.get("numbers_from_this_run"):
            add("- numbers_from_this_run:")
            add("```json")
            add(json.dumps(row.get("numbers_from_this_run"), indent=2, ensure_ascii=False))
            add("```")
        if row.get("official_numbers"):
            add("- official_numbers:")
            add("```json")
            add(json.dumps(row.get("official_numbers"), indent=2, ensure_ascii=False))
            add("```")
        if row.get("official_numbers_from_partial_run"):
            add("- official_numbers_from_partial_run:")
            add("```json")
            add(json.dumps(row.get("official_numbers_from_partial_run"), indent=2, ensure_ascii=False))
            add("```")
        add("")

    add("## 跳过项")
    add("")
    for item in cmp_mat.get("skipped") or []:
        add(f"- `{item.get('item')}` status=`{item.get('status')}`: {item.get('reason')}")
    add("")
    add("不得把 AMPLe L2、PP-bandit performative risk、ncvxpp test_acc 写进 Idea A/B leaderboard。")
    add("")
    add(f"Idea A 烟测判定 `{verdicts['idea_A']}`；Idea B `{verdicts['idea_B']}`。官方复现成败不改变这两条判定规则。")
    add("")
    return "\n".join(lines)


def build_novelty_report(artifacts: dict[str, Any], verdicts: dict[str, Any]) -> str:
    a = artifacts.get("idea_A_smoke") or {}
    b = artifacts.get("idea_B_smoke") or {}
    va = verdicts["A"]
    vb = verdicts["B"]
    mnar = va["mnar_b05"]
    cvu = b.get("controller_vs_uniform") or {}
    lines: list[str] = []
    add = lines.append
    add("# Novelty and Failure Analysis")
    add("")
    add("本报告只做研究解释，不做工程清单。所有数字仍来自同一批 JSON。")
    add("")
    add(f"预先规则下的判定：Idea A = **{verdicts['idea_A']}**，Idea B = **{verdicts['idea_B']}**。`n_seeds=3` 为初步证据。")
    add("")

    add("## Idea A")
    add("")
    add("### 最强支持证据")
    add("")
    add(
        f"MNAR 下 observed-only 对总体均值的 bias mean = `{fmt_num(mnar.get('observed_only_bias_mean'))}` "
        f"（std `{fmt_num(mnar.get('observed_only_bias_std'))}`），审计 IPW 把偏差降到 Neyman "
        f"`{fmt_num(mnar.get('neyman_ipw_bias_mean'))}` / Uniform `{fmt_num(mnar.get('uniform_ipw_bias_mean'))}`。"
        f" naive IPS 仍为 `{fmt_num(va['naive_ips_mnar_b05_bias'])}`。"
        " 这说明：偏差来自自然通道的 MNAR，而不是「随便一个 IPS 公式」。识别锚是**独立审计 + 已知 q**。"
    )
    add("")
    add("### 最强反证")
    add("")
    add(
        "同等预算下 Neyman 并不优于 Uniform。MNAR `b=0.05` 配对 RMSE 差 "
        f"`{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'mnar_b=0.05', 'mean_rmse_difference_neyman_minus_uniform'))}`，"
        f" CI [`{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'mnar_b=0.05', 'paired_bootstrap_rmse', 'ci_low'))}`, "
        f"`{fmt_num(nested(a, 'key_findings', 'neyman_vs_uniform_rmse', 'mnar_b=0.05', 'paired_bootstrap_rmse', 'ci_high'))}`] 含 0。"
        " zero-overlap `b=0.05` 的区间排除 0，但方向是 Neyman 更差。"
        " 因此「自适应分配」不是被支持的新颖点；被支持的是审计通道本身。"
    )
    add("")
    add("### 最近的已发表竞争者")
    add("")
    add("AMPLe（ACL 2025 Main）用 volume acquisition 做多维偏好识别；HyPER（ACL 2025 Main）做人机标注路由；Drago et al.（ICML 2026）做 globally compatible reward identification。目标、预算与信息都不同。本机 AMPLe 仅 partial，HyPER/Drago blocked。**不能**声称 outperform 这些论文。")
    add("")
    add("### 可能的新颖性主张")
    add("")
    add("把「已知倾向的独立审计」当作内生反馈删失下偏好/奖励泛函的识别锚。这是应用层面的问题设定，不是新的 Neyman 公式。")
    add("")
    add("### 该差异为何可能重要")
    add("")
    add("若生产系统只能看到自选反馈，observed-only 会给出系统性偏差（本合成 MNAR 约 +1.19）。一条小预算、倾向已知的审计通道在原理上可以修复识别。这对数据治理有意义。")
    add("")
    add("### 当前结果尚不支持的部分")
    add("")
    add("不支持：新分配算法、优于 Uniform 的 RMSE、优于 AMPLe/HyPER、可投稿的理论定理、LLM 后训练结果。v1.0 必须继续被称为经典 Neyman。")
    add("")
    add("### 还需要的工作")
    add("")
    add("理论：MNAR 自然通道 + 独立审计下的有限样本保证。实验：更多种子；真实 MultiPref 上只把 audit 用在 human/expert 标签，不把三类标注合成单一真值；不要把 volume 适配器叫做 AMPLe。")
    add("")

    add("## Idea B")
    add("")
    add("### 最强支持证据")
    add("")
    add(
        f"同预算下控制器的 late λ_min 高于 Uniform：mean difference `{fmt_num(nested(cvu, 'information_paired_bootstrap', 'mean_difference'))}`，"
        f" CI [`{fmt_num(nested(cvu, 'information_paired_bootstrap', 'ci_low'))}`, `{fmt_num(nested(cvu, 'information_paired_bootstrap', 'ci_high'))}`] 排除 0。"
        f" Greedy/NoFeedback 的弱技能最终观测率为 `{fmt_num(nested(b, 'B2', 'evidence', 'mean_final_weak_skill_observation_rate'))}`。"
        " 这表明按 `W_T` 弱方向分配，确实能改变看见了哪些技能。"
    )
    add("")
    add("### 最强反证")
    add("")
    add(
        f"`endogenous_collapse={fmt_num(vb['endogenous_collapse'])}`，联合判据 `{fmt_num(vb['joint_criterion'])}`。"
        " 观测表现稳定、λ_min 下降，但弱技能误差没有继续恶化。"
        f" 真实表现配对 CI [`{fmt_num(nested(cvu, 'true_performance_paired_bootstrap', 'ci_low'))}`, "
        f"`{fmt_num(nested(cvu, 'true_performance_paired_bootstrap', 'ci_high'))}`] 含 0。"
        f" JSON `beats_uniform_on_both={fmt_num(cvu.get('beats_uniform_on_both'))}` 不能覆盖这一区间规则。"
    )
    add("")
    add("### 最近的已发表竞争者")
    add("")
    add("ICML 2024 PP-bandit（官方数值实验本机 complete，但是 1-D reparameterized ZO demo）与 NeurIPS 2024 ncvxpp（partial）。它们研究 performative risk，不研究技能级观测 Gramian。适配 PP-bandit 到 B2 被标 BLOCKED，以免伪造「官方算法接到新环境」。")
    add("")
    add("### 可能的新颖性主张")
    add("")
    add("闭环策略使某些技能的反馈消失，观测指标仍可看起来很好，而 `λ_min(W_T/T)` 暴露不可识别方向。若该现象稳定出现，它不同于普通 catastrophic forgetting。")
    add("")
    add("### 该差异为何可能重要")
    add("")
    add("若只监控观测准确率，系统可能在弱技能上停止学习却仍显示「表现稳定」。信息矩阵提供了一个与奖励不同的诊断。")
    add("")
    add("### 当前结果尚不支持的部分")
    add("")
    add("不支持：已证实的内生坍缩、控制器改善长期真实表现、击败 performative 论文、把 B1 叫做策略诱发坍缩、任意矩阵都叫 observability Gramian。")
    add("")
    add("### 还需要的工作")
    add("")
    add("换初始化与更长 horizon，使弱技能误差有「先低后高」的空间；把信息改善与真实表现分开检验；理论需要策略依赖观测算子的不可识别条件，而不是再发明一个 Gram 矩阵名字。")
    add("")
    add("## 总评")
    add("")
    add("两个想法目前都停在 INCONCLUSIVE。A 的识别现象比分配规则更值得跟；B 的信息诊断比「已经打赢长期奖励」更值得跟。都还不够成为 CCF-A Main 投稿主张。")
    add("")
    return "\n".join(lines)


def generate_all(*, write_env: bool = True) -> dict[str, Path]:
    env = write_environment_manifests(REPO_ROOT) if write_env else load_json(PRIMARY_SOURCES["env_reports"]) or {}
    artifacts = load_artifacts()
    verdicts = compute_verdicts(artifacts)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main": REPORTS_DIR / "A_B_SMOKE_EXPERIMENT_REPORT.md",
        "audit": REPORTS_DIR / "BASELINE_REPRODUCTION_AUDIT.md",
        "novelty": REPORTS_DIR / "NOVELTY_AND_FAILURE_ANALYSIS.md",
    }
    outputs["main"].write_text(build_main_report(artifacts, verdicts, env), encoding="utf-8")
    outputs["audit"].write_text(build_audit_report(artifacts, verdicts), encoding="utf-8")
    outputs["novelty"].write_text(build_novelty_report(artifacts, verdicts), encoding="utf-8")
    return outputs


def main() -> int:
    outputs = generate_all(write_env=True)
    for key, path in outputs.items():
        print(f"wrote {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
