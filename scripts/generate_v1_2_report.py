#!/usr/bin/env python3
"""Build the v1.2 Chinese published-baseline reports from on-disk artifacts.

Every number is copied from locked audits or existing JSON. This script does
not invent 10/30-seed CCF-A head-to-head metrics, does not fill primary-table
Performance cells for INCOMPARABLE rows, and does not override the Stage-3
lock: ZERO DIRECT / scientifically defensible ADAPTED cells.
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

REPORTS_DIR = REPO_ROOT / "reports" / "v1.2"
VERDICT = "PUBLISHED BASELINE COMPARISON NOT ESTABLISHED"
N_DIRECT = 0
N_ADAPTED = 0

PRIMARY_SOURCES = {
    "audit_00": REPORTS_DIR / "00_EXISTING_EVIDENCE_AUDIT.md",
    "audit_01": REPORTS_DIR / "01_PUBLISHED_BASELINE_REGISTRY.md",
    "audit_02": REPORTS_DIR / "02_ORIGINAL_REPRODUCTION.md",
    "audit_03": REPORTS_DIR / "03_COMPARABILITY_AUDIT.md",
    "ample_metrics": REPO_ROOT
    / "idea_A"
    / "baselines"
    / "AMPLe"
    / "official_reproduction"
    / "metrics.json",
    "pp_metrics": REPO_ROOT
    / "idea_B"
    / "baselines"
    / "Performative_Bandit_ICML2024"
    / "official_reproduction"
    / "metrics.json",
    "idea_A_audit": REPO_ROOT / "idea_A" / "baselines" / "REPRODUCTION_AUDIT.json",
    "idea_B_audit": REPO_ROOT / "idea_B" / "baselines" / "REPRODUCTION_AUDIT.json",
    "selector_metrics": REPO_ROOT
    / "results"
    / "v1.2"
    / "idea_A"
    / "original"
    / "activeuf_selector_unit_check"
    / "metrics.json",
    "selector_config": REPO_ROOT
    / "results"
    / "v1.2"
    / "idea_A"
    / "original"
    / "activeuf_selector_unit_check"
    / "config.yaml",
    "selector_env": REPO_ROOT
    / "results"
    / "v1.2"
    / "idea_A"
    / "original"
    / "activeuf_selector_unit_check"
    / "environment.json",
    "idea_A_tier2": REPO_ROOT / "idea_A" / "results" / "v1.1" / "TIER2_SUMMARY.json",
    "idea_B_tier2": REPO_ROOT / "idea_B" / "results" / "v1.1" / "TIER2_SUMMARY.json",
    "activeuf_manifest": REPO_ROOT
    / "idea_A"
    / "baselines"
    / "ActiveUltraFeedback_MANIFEST.md",
}

MAIN_REPORT_SECTIONS = [
    "1. Executive Summary",
    "2. Verified Published Papers",
    "3. Original Problem Formulations",
    "4. Original Benchmark Reproduction",
    "5. Reproduction Differences",
    "6. Benchmark Compatibility",
    "7. Idea A Head-to-Head Results",
    "8. Idea B Head-to-Head Results",
    "9. Extended Feedback Experiments",
    "10. Cost and Budget Fairness",
    "11. Implementation and Leakage Audit",
    "12. Negative Results",
    "13. Statistical Robustness",
    "14. Novelty Implications",
    "15. Research Decision",
    "16. Next Algorithmic Improvements",
    "17. Exact Reproduction Commands",
    "18. Artifact Index",
]

COMPARABILITY_CELLS = [
    {
        "run_id": "C-AMPLe-A",
        "protocol": "original_AMPLe_vs_IdeaA_RMSE",
        "method": "AMPLe volume vs A-v1.1",
        "metric": "preference_vector_L2 vs functional_RMSE",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-HyPER-A",
        "protocol": "HyPER_route_vs_IdeaA",
        "method": "HyPER vs A-v1.1",
        "metric": "RewardBench vs functional_RMSE",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-Drago-A",
        "protocol": "Drago_theory_vs_IdeaA",
        "method": "Drago et al. vs A-v1.1",
        "metric": "compatible_reward_identification vs audit_RMSE",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-ActiveUF-full-A",
        "protocol": "ActiveUF_full_RM_DPO_vs_IdeaA",
        "method": "ActiveUF full pipeline vs A-v1.1",
        "metric": "GSM8K/RewardBench vs functional_RMSE",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-A2-Av11-ActiveUF",
        "protocol": "Protocol_A2",
        "method": "A-v1.1b/c vs official DRTS/DeltaUCB/InfoMax",
        "metric": "head_to_head_unavailable",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-A2-UniformDR-ActiveUF",
        "protocol": "Protocol_A2_auxiliary",
        "method": "Uniform cross-fit DR vs official selectors",
        "metric": "head_to_head_unavailable",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-UCBOGD-vision",
        "protocol": "UCB-OGD_Table1",
        "method": "UCB-OGD CIFAR/VOC/VQA/testbed vs A/B",
        "metric": "class_acc/mAP vs RMSE/worst-skill",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-UCBOGD-logistic",
        "protocol": "independent_paper_based_logistic",
        "method": "UCB-OGD Algorithm 1 (not instantiated)",
        "metric": "not_run",
        "seed_count": 0,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "DIAGNOSTIC",
    },
    {
        "run_id": "C-PPbandit-B",
        "protocol": "original_PP-bandit_1D_vs_IdeaB",
        "method": "PP-bandit ZO vs B-v1.1",
        "metric": "performative_risk vs worst-skill_MSE",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-B2-Bv11-ZO",
        "protocol": "Protocol_B2",
        "method": "B-v1.1 vs official two-layer ZO + partial feedback",
        "metric": "head_to_head_unavailable",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
    {
        "run_id": "C-ncvxpp-B",
        "protocol": "ncvxpp_SGD_vs_IdeaB",
        "method": "ncvxpp performative SGD vs B-v1.1",
        "metric": "loss/acc vs worst-skill_MSE",
        "seed_count": None,
        "source_file": "reports/v1.2/03_COMPARABILITY_AUDIT.md",
        "classification": "INCOMPARABLE",
    },
]


def load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def rel(path: Path | str | None) -> str:
    if path is None:
        return "MISSING"
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(candidate)


def fmt_num(value: Any, digits: int = 16) -> str:
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


def empty_performance(classification: str) -> None:
    """Primary comparison Performance is filled only for DIRECT/ADAPTED."""
    if classification in {"DIRECT", "ADAPTED"}:
        raise AssertionError("Stage 3 lock forbids DIRECT/ADAPTED rows")
    return None


def load_artifacts() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for key, path in PRIMARY_SOURCES.items():
        if path.suffix == ".json":
            artifacts[key] = load_json(path)
        else:
            artifacts[key] = load_text(path) if path.suffix in {".md", ".yaml"} else path
        artifacts[f"{key}_path"] = path
    artifacts["selector_config_data"] = None
    config_path = PRIMARY_SOURCES["selector_config"]
    if config_path.is_file():
        import yaml

        artifacts["selector_config_data"] = yaml.safe_load(
            config_path.read_text(encoding="utf-8")
        )
    return artifacts


def lock_facts(artifacts: dict[str, Any]) -> dict[str, Any]:
    audit = artifacts["audit_03"] or ""
    if "没有任何格子被判为 DIRECT 或可辩护的 ADAPTED" not in audit:
        raise RuntimeError("03_COMPARABILITY_AUDIT.md no longer locks zero DIRECT/ADAPTED cells")
    if "DIRECT 格子数：0。ADAPTED 格子数：0。" not in audit:
        raise RuntimeError("03_COMPARABILITY_AUDIT.md DIRECT/ADAPTED counts changed")
    return {
        "verdict": VERDICT,
        "n_direct": N_DIRECT,
        "n_adapted": N_ADAPTED,
        "protocol_a2": "BLOCKED_INCOMPARABLE",
        "protocol_b2": "INCOMPARABLE",
        "v13_algorithmic_iteration": False,
    }


def ncvxpp_blob(artifacts: dict[str, Any]) -> dict[str, Any]:
    audit = artifacts["idea_B_audit"] or {}
    for item in audit.get("baselines", []):
        if "ncvxpp" in str(item.get("official_source", "")).lower() or "Nonconvex" in str(
            item.get("paper", "")
        ):
            return item
    return {}


def ample_blob(artifacts: dict[str, Any]) -> dict[str, Any]:
    return artifacts["ample_metrics"] or {}


def pp_blob(artifacts: dict[str, Any]) -> dict[str, Any]:
    return artifacts["pp_metrics"] or {}


def paired_diff(blob: dict[str, Any]) -> Any:
    if not isinstance(blob, dict):
        return None
    if "difference_candidate_minus_baseline" in blob:
        return blob["difference_candidate_minus_baseline"]
    return blob.get("mean_difference")


def v11_internal(artifacts: dict[str, Any]) -> dict[str, Any]:
    """Classical v1.1 numbers. Not CCF-A head-to-head evidence."""
    a2 = artifacts["idea_A_tier2"] or {}
    b2 = artifacts["idea_B_tier2"] or {}
    static = nested(a2, "static", "mnar", "b=0.05", "paired_vs_uniform_dr", "v1.1c", default={}) or {}
    loop = nested(
        a2, "continual", "mnar_audit", "b=0.05", "paired_vs_uniform_dr", "v1.1c", default={}
    ) or {}
    observed = nested(
        a2, "static", "mnar", "b=0.05", "paired_vs_uniform_dr", "observed_only", default={}
    ) or {}
    env1 = nested(
        b2, "cells", "Env1", "paired_vs_E_optimal", "B-v1.1c", "late_worst_skill_mse", default={}
    ) or {}
    env1_regret = nested(
        b2, "cells", "Env1", "paired_vs_E_optimal", "B-v1.1c", "cumulative_regret", default={}
    ) or {}
    env2 = nested(
        b2, "cells", "Env2", "paired_vs_E_optimal", "B-v1.1c", "late_worst_skill_mse", default={}
    ) or {}
    return {
        "a_static": static,
        "a_loop": loop,
        "a_observed": observed,
        "b_env1_worst": env1,
        "b_env1_regret": env1_regret,
        "b_env2_worst": env2,
        "b_env1_c_mean": nested(
            b2, "cells", "Env1", "methods", "B-v1.1c", "late_worst_skill_mse", "mean"
        ),
        "b_env1_e_mean": nested(
            b2, "cells", "Env1", "methods", "E-optimal", "late_worst_skill_mse", "mean"
        ),
    }


def build_comparison_rows(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in COMPARABILITY_CELLS:
        row = dict(cell)
        row["performance"] = empty_performance(cell["classification"])
        rows.append(row)

    ample = ample_blob(artifacts)
    rows.append(
        {
            "run_id": "ample_official_partial",
            "protocol": "original_AMPLe_n_rounds=3",
            "method": "AMPLe volume",
            "metric": "l2_to_true_preference",
            "seed_count": 1,
            "source_file": rel(PRIMARY_SOURCES["ample_metrics"]),
            "classification": "ORIGINAL_SETTING",
            "performance": None,
            "original_setting_value": {
                "l2_to_true_t0": nested(ample, "l2_to_true", "0"),
                "l2_to_true_t3": nested(ample, "l2_to_true", "3"),
                "n_rounds": 3,
                "burnin": 50000,
                "acquisition": "volume",
            },
        }
    )

    pp = pp_blob(artifacts)
    rows.append(
        {
            "run_id": "ppbandit_official_numerical",
            "protocol": "original_PP-bandit_1D",
            "method": "PP-bandit two-layer ZO reparameterization",
            "metric": "performative_risk",
            "seed_count": 1,
            "source_file": rel(PRIMARY_SOURCES["pp_metrics"]),
            "classification": "ORIGINAL_SETTING",
            "performance": None,
            "original_setting_value": {
                "numpy_seed": pp.get("numpy_seed"),
                "single_run_final_PR_phi": nested(pp, "single_run", "final_PR_phi"),
                "single_run_final_PR_theta": nested(pp, "single_run", "final_PR_theta"),
                "multi_run_PR_phi_final_mean": nested(pp, "multi_run", "PR_phi_final_mean"),
                "multi_run_PR_theta_final_mean": nested(pp, "multi_run", "PR_theta_final_mean"),
            },
        }
    )

    ncvx = ncvxpp_blob(artifacts)
    what = ncvx.get("what_ran") or {}
    rows.append(
        {
            "run_id": "ncvxpp_synthetic_iter_single_process",
            "protocol": "original_ncvxpp_synthetic-iter_rank0",
            "method": "ncvxpp performative SGD-GD (size=1 stub)",
            "metric": "loss_last/train_acc_last/test_acc_last",
            "seed_count": 1,
            "source_file": rel(PRIMARY_SOURCES["idea_B_audit"]),
            "classification": "ORIGINAL_SETTING",
            "performance": None,
            "original_setting_value": {
                "local_status": ncvx.get("local_status"),
                "eps_list": what.get("eps_list"),
                "batch": what.get("batch"),
                "wall_sec": what.get("wall_sec"),
                "npy": what.get("npy"),
            },
        }
    )

    sel = artifacts["selector_metrics"] or {}
    rows.append(
        {
            "run_id": "activeuf_selector_unit_check",
            "protocol": None,
            "method": "official DRTS/DeltaUCB/InfoMax",
            "metric": "selected_pairs_on_synthetic_RM_UQ",
            "seed_count": 1,
            "source_file": rel(PRIMARY_SOURCES["selector_metrics"]),
            "classification": "DIAGNOSTIC",
            "performance": None,
            "note": "selector_unit_check is NOT a head-to-head. Protocol A2 remains BLOCKED_INCOMPARABLE.",
            "diagnostic_value": {
                "status": sel.get("status"),
                "comparability_cell": sel.get("comparability_cell"),
                "protocol_a2": sel.get("protocol_a2"),
                "vllm_imported": sel.get("vllm_imported"),
                "all_official_files": sel.get("all_official_files"),
                "selected_pairs": {
                    key: nested(sel, "selectors", key, "selected_pairs")
                    for key in ("drts", "deltaucb", "infomax")
                },
            },
        }
    )
    return rows


def header_block(title: str, artifacts: dict[str, Any], extra: str = "") -> str:
    now = datetime.now(timezone.utc).isoformat()
    branch = git_command("rev-parse", "--abbrev-ref", "HEAD")
    commit = git_command("rev-parse", "HEAD")
    remote = git_command("remote", "get-url", "origin")
    missing = [
        rel(path)
        for key, path in PRIMARY_SOURCES.items()
        if not path.is_file()
    ]
    missing_txt = "无" if not missing else ", ".join(missing)
    lines = [
        f"# {title}",
        "",
        "本文件由 `scripts/generate_v1_2_report.py` 从磁盘 JSON 与已锁定审计写出。数字均来自下列产物，没有手填对照 RMSE，也没有发明 10/30-seed CCF-A head-to-head。",
        "",
        f"- 生成时间（UTC）：`{now}`",
        f"- 仓库：`{REPO_ROOT}`",
        f"- 分支：`{branch}`",
        f"- commit：`{commit}`",
        f"- remote：`{remote}`",
        f"- 缺失的主源：{missing_txt}",
        "- 可比性锁：`reports/v1.2/03_COMPARABILITY_AUDIT.md`",
        f"- 主判定：`{VERDICT}`",
        f"- scientifically valid DIRECT 对照数：`{N_DIRECT}`；可辩护 ADAPTED 对照数：`{N_ADAPTED}`",
    ]
    if extra:
        lines.extend(["", extra])
    return "\n".join(lines)


def write_comparison_summary(artifacts: dict[str, Any], facts: dict[str, Any]) -> Path:
    payload = {
        "verdict": facts["verdict"],
        "n_scientifically_valid_direct_comparisons": facts["n_direct"],
        "n_scientifically_defensible_adapted_comparisons": facts["n_adapted"],
        "protocol_a2": facts["protocol_a2"],
        "protocol_b2": facts["protocol_b2"],
        "v13_algorithmic_iteration_justified": facts["v13_algorithmic_iteration"],
        "performance_rule": "performance is filled only for DIRECT or scientifically defensible ADAPTED head-to-head cells; otherwise null",
        "rows": build_comparison_rows(artifacts),
    }
    path = REPORTS_DIR / "comparison_summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _empty_primary_row(
    method: str, venue: str, protocol: str, budget: str, metric: str, comparable: str
) -> list[str]:
    return [method, venue, protocol, budget, metric, "", "", comparable]


def build_main_report(artifacts: dict[str, Any], facts: dict[str, Any]) -> str:
    ample = ample_blob(artifacts)
    pp = pp_blob(artifacts)
    ncvx = ncvxpp_blob(artifacts)
    ncvx_what = ncvx.get("what_ran") or {}
    ncvx_npy = ncvx_what.get("npy") or {}
    sel = artifacts["selector_metrics"] or {}
    sel_env = artifacts["selector_env"] or {}
    internal = v11_internal(artifacts)
    a_static = internal["a_static"]
    a_loop = internal["a_loop"]
    a_obs = internal["a_observed"]
    b_env1 = internal["b_env1_worst"]
    b_reg = internal["b_env1_regret"]
    b_env2 = internal["b_env2_worst"]
    b_c_mean = internal["b_env1_c_mean"]
    b_e_mean = internal["b_env1_e_mean"]

    idea_a_table = md_table(
        ["Method", "Published venue", "Protocol", "Label budget", "Main metric", "Performance", "Cost", "Comparable?"],
        [
            _empty_primary_row(
                "ActiveUF DRTS",
                "ICML 2026 Main",
                "Protocol A2 BLOCKED",
                "preference pairs; live RM UQ missing",
                "not aligned with Idea A RMSE",
                "INCOMPARABLE",
            ),
            _empty_primary_row(
                "ActiveUF DeltaUCB",
                "ICML 2026 Main",
                "Protocol A2 BLOCKED",
                "preference pairs; live RM UQ missing",
                "not aligned with Idea A RMSE",
                "INCOMPARABLE",
            ),
            _empty_primary_row(
                "ActiveUF InfoMax",
                "ICML 2026 framework (Saha 2021 dueling-bandit; not ActiveUF's new algorithm)",
                "Protocol A2 BLOCKED",
                "preference pairs; live RM UQ missing",
                "not aligned with Idea A RMSE",
                "INCOMPARABLE",
            ),
            _empty_primary_row(
                "AMPLe",
                "ACL 2025 Main",
                "original partial n_rounds=3",
                "query rounds (local 3; README 100); burn-in 50000 separate",
                "preference-vector L2 (original setting only)",
                "INCOMPARABLE",
            ),
            _empty_primary_row(
                "HyPER",
                "ACL 2025 Main",
                "official pipeline blocked",
                "human vs LM annotation counts",
                "RewardBench / downstream RM",
                "INCOMPARABLE",
            ),
            _empty_primary_row(
                "Our A-v1.1",
                "Proposed (this repo)",
                "v1.1 synthetic Selective Feedback; not Protocol A2",
                "audit fraction b",
                "functional RMSE",
                "no valid published head-to-head",
            ),
            _empty_primary_row(
                "Uniform DR",
                "Classical",
                "v1.1 internal control only",
                "same b as A-v1.1",
                "functional RMSE",
                "classical internal; not a CCF-A paper",
            ),
        ],
    )
    idea_b_table = md_table(
        ["Method", "Published venue", "Protocol", "Feedback budget", "Main metric", "Performance", "Cost", "Comparable?"],
        [
            _empty_primary_row(
                "Performative Bandit",
                "ICML 2024",
                "original 1-D notebook; Protocol B2 INCOMPARABLE",
                "deployments / two-point queries",
                "PR(φ), PR(θ)",
                "INCOMPARABLE",
            ),
            _empty_primary_row(
                "Performative SGD",
                "NeurIPS 2024",
                "single-process partial; mpiexec blocked",
                "SGD iterations × batch",
                "loss / acc",
                "INCOMPARABLE",
            ),
            _empty_primary_row(
                "Our B-v1.1",
                "Proposed (this repo)",
                "v1.1 Env1–3; not Protocol B2",
                "expected audit fraction b",
                "late worst-skill MSE",
                "no valid published head-to-head",
            ),
            _empty_primary_row(
                "E-optimal",
                "Classical",
                "v1.1 internal control only",
                "same b as B-v1.1",
                "late worst-skill MSE",
                "classical internal; not a CCF-A paper",
            ),
            _empty_primary_row(
                "D-optimal",
                "Classical",
                "v1.1 internal control only",
                "same b as B-v1.1",
                "late worst-skill MSE",
                "classical internal; not a CCF-A paper",
            ),
            _empty_primary_row(
                "Uniform",
                "Classical",
                "v1.1 internal control only",
                "same b as B-v1.1",
                "late worst-skill MSE",
                "classical internal; not a CCF-A paper",
            ),
        ],
    )

    npy_rows = []
    for name, rec in ncvx_npy.items():
        npy_rows.append(
            [
                name,
                str(rec.get("iter_last", "NA")),
                fmt_num(rec.get("loss_last")),
                fmt_num(rec.get("train_acc_last")),
                fmt_num(rec.get("test_acc_last")),
            ]
        )
    npy_table = md_table(
        ["npy", "iter_last", "loss_last", "train_acc_last", "test_acc_last"],
        npy_rows,
    ) if npy_rows else "（无 ncvxpp npy 记录）"

    sel_rows = []
    for key, label in (
        ("drts", "DRTS / DoubleReverseThompsonSampling"),
        ("deltaucb", "DeltaUCB"),
        ("infomax", "InfoMax"),
    ):
        rec = nested(sel, "selectors", key, default={})
        sel_rows.append(
            [
                label,
                str(rec.get("qualname", "NA")),
                str(rec.get("file_is_official", "NA")),
                json.dumps(rec.get("selected_pairs"), ensure_ascii=False),
            ]
        )
    sel_table = md_table(
        ["选择器", "官方类", "file_is_official", "selected_pairs"],
        sel_rows,
    )

    parts = [
        header_block(
            "v1.2 已发表 CCF-A 基线对照（主交付）",
            artifacts,
            extra=(
                "**判定预先冻结在 `03_COMPARABILITY_AUDIT.md`。** "
                "没有 DIRECT 格子，也没有可辩护的 ADAPTED 格子。"
                "Protocol A2 与 Protocol B2 均为 INCOMPARABLE / BLOCKED。"
                "隔离 venv 的 `selector_unit_check` **不是** head-to-head。"
                "不得把 v1.1 相对 Uniform DR / E-optimal 的结果写成击败 CCF-A 论文。"
                "Idea A / Idea B **不得**标为 PROMISING；v1.1 已相对经典对照判定 **NOT PROMISING**。"
            ),
        ),
        "",
        "## 1. Executive Summary",
        "",
        f"- **相对已发表 CCF-A Main 方法的对照状态：`{VERDICT}`。**",
        f"- scientifically valid DIRECT 对照数：`{facts['n_direct']}`。可辩护 ADAPTED 对照数：`{facts['n_adapted']}`。",
        "- Protocol A2（A-v1.1 vs 官方 DRTS / DeltaUCB / InfoMax）：**BLOCKED / INCOMPARABLE**。缺少 live RM UQ；A-v1.1 的方差分配到“选哪一对响应”的映射在看测试结果之前已被否决。",
        "- Protocol B2（B-v1.1 vs 官方两层零阶 + 部分反馈）：**INCOMPARABLE**。PP-bandit 一维环境没有技能组，也没有可分配的审计动作。",
        "- Idea A v1.1 相对经典对照：`NOT PROMISING`（输给 Uniform cross-fit DR）。这 **不能** 改写成发表基线胜利或失败。",
        "- Idea B v1.1 相对经典对照：`NOT PROMISING`（worst-skill MSE 输给 E-optimal）。同样 **不能** 代替 CCF-A 对照。",
        "- Idea A / Idea B 在本阶段 **均不得** 称为 PROMISING。",
        "- v1.3 算法迭代：**不成立**。本机当前没有合法对照路径（无 RM UQ，也无 skill-free 映射）。",
        "",
        "## 2. Verified Published Papers",
        "",
        md_table(
            ["ID", "题目", "Venue", "仓库 / commit", "复现状态", "与 Idea A/B"],
            [
                [
                    "AMPLe",
                    "Comparison-based Active Preference Learning for Multi-dimensional Personalization",
                    "ACL 2025 Main",
                    "https://github.com/ml-postech/AMPLe `0cad341f6a9e77063ef451c27732bf5c1534c797`",
                    "partial",
                    "INCOMPARABLE",
                ],
                [
                    "HyPER",
                    "Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback",
                    "ACL 2025 Main",
                    "https://github.com/allenai/hybrid-preferences `4d96b27b989828b0b69a430f68d25b6b36e9015a`",
                    "blocked",
                    "INCOMPARABLE",
                ],
                [
                    "Drago",
                    "Online Compatible Reward Identification from Preference Feedback",
                    "ICML 2026",
                    "无官方仓库",
                    "blocked",
                    "INCOMPARABLE",
                ],
                [
                    "ActiveUF",
                    "ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning",
                    "ICML 2026 Main",
                    "https://github.com/lasgroup/ActiveUltraFeedback `3cf700a85bbe344b1b472890a488671e2916a8b4`",
                    "全文 blocked；选择器 unit check 仅 DIAGNOSTIC",
                    "INCOMPARABLE",
                ],
                [
                    "UCB-OGD",
                    "Adaptive Data Collection for Robust Learning Across Multiple Distributions",
                    "ICML 2025 Main",
                    "未找到官方仓库",
                    "independent_paper_based = not done；CIFAR/VOC/VQA blocked",
                    "INCOMPARABLE",
                ],
                [
                    "PP-bandit",
                    "Performative Prediction with Bandit Feedback: Learning through Reparameterization",
                    "ICML 2024",
                    "https://github.com/UCSC-REAL/PP-bandit-feedback `d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5`",
                    "数值脚本已跑完；未对齐图 4–6 端点",
                    "技能分配 INCOMPARABLE",
                ],
                [
                    "ncvxpp",
                    "Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss",
                    "NeurIPS 2024",
                    "https://github.com/Qiang-CU/nips2024-ncvxpp `ccbeeddfbf9237aa06dca3a1579529faabd8fe99`",
                    "单进程 partial；`mpiexec -np 10` blocked",
                    "INCOMPARABLE",
                ],
            ],
        ),
        "",
        "InfoMax 出现在 ActiveUltraFeedback 仓库中，但 README 把它标成已有 dueling-bandit 方法（Saha, 2021），**不是** 该论文的新算法。DRTS（源码类名 `DoubleReverseThompsonSampling`）与 `DeltaUCB` 才是该文的新选择器。",
        "",
        "## 3. Original Problem Formulations",
        "",
        "- **AMPLe**：在 RM 打分的回复对上，用 volume / random acquisition 识别单纯形上的多维个性化偏好向量。指标是到 `true_preference` 的向量误差，不是有限总体标量泛函 RMSE。",
        "- **HyPER**：把实例路由到人类或 LM 标注，优化下游 RM / RewardBench。",
        "- **Drago et al.**：理论工作，全局兼容奖励识别，不是有限预算审计 RMSE。",
        "- **ActiveUltraFeedback**：从 prompt 池生成回复，用不确定度奖励模型选出一对，经 LLM-as-a-Judge 标注，再训练 RM / DPO/CPO。动作是选 completion 下标对，可见信息是 `(rewards, lower_bounds, upper_bounds)`。",
        "- **UCB-OGD**：每轮选一个数据源采样并标注，用 OGD 更新；论文 Table 1 是 CIFAR-10 / VOC2012 / VQAv2 / COSMOS 上的类准确率或 mAP。",
        "- **PP-bandit**：把 performative risk 重参数化到诱导分布参数 `φ`，外层零阶优化 `φ`，内层 `LearnModel` 找能诱导目标分布的 `θ`。官方 notebook 是一维 Bernoulli，没有技能组。",
        "- **ncvxpp**：决策依赖分布下的 performative SGD / SGD-GD。官方 `synthetic-iter` 是线性模型合成实验。",
        "- **Idea A v1.1**：有限总体上对每个单位给出审计概率 `q_i`，估计标量泛函。动作不是选回复对。",
        "- **Idea B v1.1**：在技能群体上分配审计，主指标 late worst-skill MSE。官方 PP-bandit 环境没有这些对象。",
        "",
        "## 4. Original Benchmark Reproduction",
        "",
        "本阶段没有重跑 AMPLe burn-in、PP-bandit notebook、ncvxpp。下列数字来自已提交 JSON。",
        "",
        "### AMPLe partial",
        "",
        f"- 来源：`{rel(PRIMARY_SOURCES['ample_metrics'])}`",
        f"- `acquisition=volume`，`burnin=50000`，`n_rounds=3`（README 示例 100 未跑）",
        f"- 数据集：`minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271`",
        f"- `true_preference = {json.dumps(ample.get('true_preference'))}`",
        f"- `l2_to_true_t0 = {fmt_num(nested(ample, 'l2_to_true', '0'))}`",
        f"- `l2_to_true_t3 = {fmt_num(nested(ample, 'l2_to_true', '3'))}`",
        f"- `estimation_t0 = {json.dumps(nested(ample, 'estimation', '0'))}`",
        f"- `estimation_t3 = {json.dumps(nested(ample, 'estimation', '3'))}`",
        f"- `chosen_indices` 来自 chosen 1/2/3：`{json.dumps([nested(ample, 'chosen', str(i), 'index') for i in (1, 2, 3)])}`",
        f"- `elapsed_wall_sec_approx = {fmt_num(ample.get('elapsed_wall_sec_approx'))}`",
        "- 该 L2 **不可** 与 Idea A RMSE 相减。",
        "",
        "### PP-bandit 数值运行",
        "",
        f"- 来源：`{rel(PRIMARY_SOURCES['pp_metrics'])}`",
        f"- `numpy_seed = {pp.get('numpy_seed')}`，`elapsed_sec = {fmt_num(pp.get('elapsed_sec'))}`",
        f"- 单次 `num_iterations=500`，`x0=0.3`：`final_PR_phi = {fmt_num(nested(pp, 'single_run', 'final_PR_phi'))}`，`final_PR_theta = {fmt_num(nested(pp, 'single_run', 'final_PR_theta'))}`",
        f"- 10 个初值各 100 步：`PR_phi_final_mean = {fmt_num(nested(pp, 'multi_run', 'PR_phi_final_mean'))}`（std `{fmt_num(nested(pp, 'multi_run', 'PR_phi_final_std'))}`），`PR_theta_final_mean = {fmt_num(nested(pp, 'multi_run', 'PR_theta_final_mean'))}`（std `{fmt_num(nested(pp, 'multi_run', 'PR_theta_final_std'))}`）",
        "- 论文 Appendix H 图 4–7 没有可抄端点表。**不声称对齐图 4–6。**",
        "",
        "### ncvxpp 单进程 partial",
        "",
        f"- 来源：`{rel(PRIMARY_SOURCES['idea_B_audit'])}`",
        f"- `local_status = {ncvx.get('local_status')}`；官方 `mpiexec -np 10` blocked",
        f"- `eps_list = {json.dumps(ncvx_what.get('eps_list'))}`，`batch = {ncvx_what.get('batch')}`，`wall_sec = {fmt_num(ncvx_what.get('wall_sec'))}`",
        npy_table,
        "",
        "### ActiveUF 选择器 unit check（不是复现论文表）",
        "",
        f"- 来源：`{rel(PRIMARY_SOURCES['selector_metrics'])}`",
        f"- `check_type = {sel.get('check_type')}`，`comparability_cell = {json.dumps(sel.get('comparability_cell'))}`，`protocol_a2 = {sel.get('protocol_a2')}`",
        f"- 隔离 venv Python：`{sel_env.get('python')}`；`torch_version = {sel_env.get('torch_version')}`；`torch_cuda = {json.dumps(sel_env.get('torch_cuda'))}`；`vllm_imported = {json.dumps(sel.get('vllm_imported'))}`",
        f"- 合成池：`n_prompts = {nested(sel, 'candidate_pool', 'n_prompts')}`，`n_completions_per_prompt = {nested(sel, 'candidate_pool', 'n_completions_per_prompt')}`，每 prompt 预算 1 对",
        sel_table,
        "- 这只证明官方类能在合成 RM UQ 张量上返回一对索引。它 **不是** DRTS vs A-v1.1 的对照。",
        "",
        "### HyPER / Drago / UCB-OGD / ActiveUF 全文",
        "",
        "- HyPER：blocked（TPU v3-128、vLLM、8B/13B RM）。无本地数字。",
        "- Drago：blocked，无官方代码。无本地数字。",
        "- UCB-OGD：无官方仓库；`independent_paper_based = not done`。CIFAR / VOC / VQA blocked。论文 Table 1 数字不抄进对照列。",
        "- ActiveUF 全文响应生成、RM 训练、DPO/CPO：blocked（CUDA 13 / vLLM 0.19）。",
        "",
        "## 5. Reproduction Differences",
        "",
        "- AMPLe：`n_rounds` 从 README 100 改为 3；`--dims` 省略（2-aspect 数据集当时 HTTP 401）；`burnin=50000` 与 `acquisition=volume` 未改。",
        "- PP-bandit：本地脚本加了 `numpy` seed 0；官方 notebook 无 seed。因此 JSON 是同一算法、固定种子的一次执行，不是对论文图像素核对。",
        "- ncvxpp：用 `mpi4py` stub，`rank=0, size=1`。不是 `mpiexec -np 10`。`dataset.py` 按 `Y==1`/`Y==0` 平衡，而 `data.csv` 标签是 `{+1,-1}`，只有 `+1` 类留下——官方代码与数据的交互，未改写。`spam-iter` 未跑。",
        "- ActiveUF：浅克隆、submodule 未初始化；主 venv 无 torch；隔离 venv 只跑选择器 unit check，不跑 vLLM / RM。",
        "- HyPER：浅克隆，submodule 未初始化，无训练。",
        "- UCB-OGD：没有作者代码，因此没有任何“复现差异”可列，只有缺失。",
        "",
        "## 6. Benchmark Compatibility",
        "",
        "完整十三维填写见 `reports/v1.2/03_COMPARABILITY_AUDIT.md` 与 `reports/v1.2/BENCHMARK_COMPATIBILITY.md`。摘要：",
        "",
        f"- DIRECT 格子数：`{facts['n_direct']}`。ADAPTED 格子数：`{facts['n_adapted']}`。",
        "- Protocol A2：**BLOCKED**。动作（`q_i` vs `(i,j)`）、可见信息（审计残差 vs live RM UQ）、预算单位（审计比例 `b` vs 偏好对）均不齐。映射 M1–M5 预先否决。",
        "- Protocol B2：**INCOMPARABLE**。官方环境暴露的是一维 `x` 与完整标量 PR 查询，没有技能标签，也没有分配向量。给 B-v1.1 私有技能标签会破坏信息对齐。",
        "- AMPLe L2、HyPER RewardBench、UCB-OGD 类准确率 / mAP、PP-bandit PR、ncvxpp loss/acc 都 **不得** 放进 Idea A RMSE 或 Idea B worst-skill 的同一列。",
        "",
        "## 7. Idea A Head-to-Head Results",
        "",
        "**没有合法 head-to-head。** 本阶段没有 3-seed smoke，也没有 10/30-seed Protocol A2 主格子。下表 Performance / Cost 列留空，因为格子是 INCOMPARABLE，不能填合成 RMSE，也不能填论文原表数字。",
        "",
        idea_a_table,
        "",
        "selector_unit_check（seed 0，隔离 venv）选出的下标见第 4 节。它的 `comparability_cell` 为 `null`，`protocol_a2 = BLOCKED_INCOMPARABLE`。不能从这些下标构造 DRTS vs A-v1.1 的 RMSE 差或置信区间。",
        "",
        "v1.1 内部经典对照（**不是** CCF-A 对照，不进入上表 Performance 列）来自 `idea_A/results/v1.1/TIER2_SUMMARY.json`：",
        "",
        f"- 静态 MNAR `b=0.05`，n=`{a_static.get('n_seeds')}`：A-v1.1c RMSE `{fmt_num(a_static.get('rmse_candidate'))}` vs Uniform DR `{fmt_num(a_static.get('rmse_baseline'))}`，diff `{fmt_num(paired_diff(a_static))}`，CI [`{fmt_num(a_static.get('ci_low'))}`, `{fmt_num(a_static.get('ci_high'))}`]，`interval_side={a_static.get('interval_side')}`。",
        f"- 闭环 MNAR-audit `b=0.05`，n=`{a_loop.get('n_seeds')}`：A-v1.1c RMSE `{fmt_num(a_loop.get('rmse_candidate'))}` vs Uniform DR `{fmt_num(a_loop.get('rmse_baseline'))}`，diff `{fmt_num(paired_diff(a_loop))}`，CI [`{fmt_num(a_loop.get('ci_low'))}`, `{fmt_num(a_loop.get('ci_high'))}`]，`interval_side={a_loop.get('interval_side')}`。",
        f"- 同静态格 observed-only RMSE `{fmt_num(a_obs.get('rmse_candidate'))}` vs Uniform DR `{fmt_num(a_obs.get('rmse_baseline'))}`：这是 identification-via-audit，不是自适应分配赢了发表方法。",
        "",
        "问题回答（任务第 20 节）：",
        "",
        "1. 我们的方法能否在 ActiveUltraFeedback 原任务上运行？**不能**（无 live RM UQ，动作空间不是选回复对）。",
        "2. 同等标注预算下是否超过 DRTS 或 DeltaUCB？**未建立对照，因此没有该检验。**",
        "3. 若不能，失败条件是什么？映射不成立；live RM UQ 缺失；全文流水线 blocked。",
        "4. 内生反馈删失是否暴露了发表方法的可测弱点？**没有合法实验回答这一点。**",
        "5. identification-aware 干预是否针对该弱点？**没有对照，不能声称。**",
        "6. 改进是否大于 Uniform DR？v1.1 内部结果是 A-v1.1c **没有**赢 Uniform DR；这仍不是 CCF-A 结论。",
        "7. 任何优势是否来自额外数据 / 调参 / 算力？不存在相对发表方法的已测优势。",
        "8. 结果是否支持独立 Method Paper？**不支持。** 发表基线比较未建立，且 v1.1 对经典对照已是 NOT PROMISING。",
        "",
        "## 8. Idea B Head-to-Head Results",
        "",
        "**没有合法 head-to-head。** Protocol B2 未跑。Performance / Cost 列留空。",
        "",
        idea_b_table,
        "",
        "v1.1 内部经典对照（**不是** CCF-A 对照）来自 `idea_B/results/v1.1/TIER2_SUMMARY.json`：",
        "",
        f"- Env1 n=`{b_env1.get('n_seeds')}`：B-v1.1c late worst-skill MSE mean `{fmt_num(b_c_mean)}` vs E-optimal `{fmt_num(b_e_mean)}`；配对 diff `{fmt_num(paired_diff(b_env1))}`，CI [`{fmt_num(b_env1.get('ci_low'))}`, `{fmt_num(b_env1.get('ci_high'))}`]，`interval_side={b_env1.get('interval_side')}`。",
        f"- 同格 cumulative regret 配对 diff `{fmt_num(paired_diff(b_reg))}`，CI [`{fmt_num(b_reg.get('ci_low'))}`, `{fmt_num(b_reg.get('ci_high'))}`]，`interval_side={b_reg.get('interval_side')}`。H_B1 要求 worst-skill **并且** regret 相对 E 与 D 同时完全为负；该条件未满足。",
        f"- Env2 worst-skill 配对 diff `{fmt_num(paired_diff(b_env2))}`，CI [`{fmt_num(b_env2.get('ci_low'))}`, `{fmt_num(b_env2.get('ci_high'))}`]，`interval_side={b_env2.get('interval_side')}`。",
        "",
        "问题回答：",
        "",
        "1. 能否在原 performative benchmark 上合法评估我们的方法？**不能**（原环境无技能组 / 审计动作）。",
        "2. 是否相对发表优化器降低了 performative risk？**未建立对照。**",
        "3. 内生反馈观测扩展是否改变排序？Protocol B2 未实现，无排序。",
        "4. 同等反馈成本下是否改进长期学习？无合法发表对照。",
        "5. 在相关处是否超过经典信息最优控制？v1.1 内部：worst-skill 未超过 E-optimal；这不是 CCF-A 结论。",
        "6. 是否需要发表算法得不到的信息？若强行加技能标签，是的，格子仍 INCOMPARABLE。",
        "7. 发现的失败是在原定理假设内还是外？没有对发表方法做原任务失败诊断；B2 是原环境无法承载分配决策。",
        "8. 是否支持独立算法或理论贡献？**不支持** 以“超过 PP-bandit / ncvxpp”为依据的贡献声明。",
        "",
        "## 9. Extended Feedback Experiments",
        "",
        "没有新的 policy-induced selection 扩展实验。Stage 4 的 3-seed smoke 与 Stage 5 的 10/30-seed 主矩阵按锁 **不启动**。",
        "v1.1 闭环格子仍是合成 Selective Feedback / Env1–3，不是 ActiveUF 或 PP-bandit 的扩展协议。",
        "",
        "## 10. Cost and Budget Fairness",
        "",
        "预算单位继续分列，不合并成一个 budget：",
        "",
        "- 标注次数 / 审计比例 `b`：Idea A/B 合成协议。",
        "- 偏好对：ActiveUF 选择器输出。",
        "- 标注回复：ActiveUF `full` 配置自称每 prompt 30 条生成回复。",
        "- 查询轮数：AMPLe `n_rounds`（本地 3）。burn-in 50000 另计。",
        "- 图像张数 / QA 对：UCB-OGD Table 1。",
        "- 部署次数 / 两点查询：PP-bandit。",
        "- SGD 迭代 × batch：ncvxpp（本地 batch 5，约 99426 次迭代）。",
        "- 人类成本 vs judge 成本：ActiveUF 数据卡写的是 LLM-as-a-Judge / RM score，不是人类偏好。HyPER 人类条数与 LM 条数应分开，但本机未跑。",
        "",
        "因为单位不齐，**不存在** 已匹配的 label budget 或 feedback budget 对照。selector_unit_check 的预算是每 prompt 1 对，候选池 2×4，与 Idea A 的 `b` 不是同一账本。",
        "",
        "## 11. Implementation and Leakage Audit",
        "",
        "- 没有实现 Protocol A2 / B2 对照适配器。测试 `idea_A/tests/test_v1_2_comparability.py` 禁止 `protocol_a2_adapter` 等启发式文件名。",
        "- Uncertainty Sampling **没有**改名为 DRTS / DeltaUCB / HyPER。",
        "- selector_unit_check：`oracle_labels_in_acquisition_scores = "
        f"{json.dumps(sel.get('oracle_labels_in_acquisition_scores'))}`；"
        f"`oracle_labels_passed` 对三个选择器均为 false；`call_args = [\"rewards\", \"lower_bounds\", \"upper_bounds\"]`。",
        f"- `all_official_files = {json.dumps(sel.get('all_official_files'))}`。调用的是官方类文件，不是本地重写。",
        f"- `vllm_imported = {json.dumps(sel.get('vllm_imported'))}`。主 venv 仍无 torch。",
        "- A-v1.1 分配不读隐藏 `Y`；官方选择器 `__call__` 也不读 oracle 标签。这一点本来可以对齐，但任务、动作、信息、预算已经失败，因此不构成 ADAPTED。",
        "- learner 只能看见已付费标签：本阶段没有新对照 run 去违反该规则。",
        "",
        "## 12. Negative Results",
        "",
        "- **主负结果：发表基线比较未建立。** 这是 Stage 3 锁的科学结论，不是执行失败。",
        "- Protocol A2 映射 M1–M5 全部否决；无 live RM UQ 时禁止启发式替身。",
        "- Protocol B2 原环境不能支持分配决策。",
        "- ActiveUF / HyPER / UCB-OGD 视觉实验 / ncvxpp MPI 在本机 blocked 或 partial。",
        "- v1.1 已经判定 A-v1.1c、B-v1.1c 相对经典对照 **NOT PROMISING**。本阶段不把该内部失败改写成“超过发表方法失败”或“超过发表方法成功”。",
        f"- 闭环 A-v1.1c 相对 Uniform DR：`interval_side={a_loop.get('interval_side')}`（候选更差）。",
        f"- Env1 B-v1.1c 相对 E-optimal worst-skill：`interval_side={b_env1.get('interval_side')}`（候选更差）。",
        "",
        "## 13. Statistical Robustness",
        "",
        "- 对已发表方法：**没有** 配对检验、没有 10 个开发种子、没有 30 个测试种子。不得发明这些数字。",
        "- selector_unit_check 只有 seed 0 的一次合成调用。",
        "- AMPLe partial 是 `--seed=1` 的一次 3 轮运行。",
        "- PP-bandit 本地脚本 seed 0；多初值 10 个点各 100 步是同一脚本内的数值重复，不是论文种子协议。",
        "- v1.1 内部 30/50 种子结果只描述经典对照稳健性，不迁移到 CCF-A 主表。",
        "",
        "## 14. Novelty Implications",
        "",
        "在没有合法发表对照时，不能声称相对 DRTS / DeltaUCB / AMPLe / HyPER / PP-bandit / ncvxpp 的方法贡献。",
        "v1.1 仍支持的现象是 identification-via-audit（MNAR 下审计通道降偏），那是相对 observed-only / Uniform DR 的识别事实，不是相对 CCF-A 主动学习论文的新算法胜利。",
        "名字不同（审计 vs 主动偏好、技能可观测性 vs performative risk）本身不是 novelty。",
        "",
        "## 15. Research Decision",
        "",
        "- Idea A：继续称为 PROMISING：**否**。相对发表方法：**比较未建立**。相对经典对照：已是 NOT PROMISING。",
        "- Idea B：同样 **否**。",
        "- 是否同时推进 A 与 B 的 v1.3 算法迭代：**否**。当前 Mac 上没有合法对照路径（需要 live RM UQ，或 skill-free、且不发明私有技能标签的映射）。",
        "- 详细逐条见 `reports/v1.2/A_B_RESEARCH_DECISION.md`。",
        "",
        "## 16. Next Algorithmic Improvements",
        "",
        "本阶段 **不提出** 新分配算法。用户任务明确：先建立合法发表对照，再考虑算法迭代。",
        "若未来硬件/数据允许，唯一与本锁一致的下一步是 **对照路径**，不是再写一个 Neyman 变体：",
        "",
        "1. 获得合法 live RM UQ `(rewards, lower_bounds, upper_bounds)` 与真实 completion 池后，才能重新评估 Protocol A2 是否仍被映射否决。",
        "2. 若 PP-bandit 原环境仍无分配动作，不要为 B-v1.1 私加技能标签。",
        "3. 在对照路径出现之前，不投资 v1.3 算法。",
        "",
        "## 17. Exact Reproduction Commands",
        "",
        "```bash",
        "cd /Users/grantwright/Desktop/科研/10.30新课题/v0",
        "export PYTHONPATH=\"$PWD\"",
        "",
        "# 主 venv 测试（不要求 GPU / torch）",
        ".venv/bin/python -m pytest --import-mode=importlib",
        "",
        "# 从已有 JSON 重建 v1.2 报告（不跑对照实验）",
        ".venv/bin/python scripts/generate_v1_2_report.py",
        "",
        "# ActiveUF 选择器 unit check（隔离 venv，不是 Protocol A2）",
        ".isolated_baselines_venv/bin/python idea_A/experiments/v1.2/run_activeuf_selector_unit_check.py",
        "",
        "# 下列原设定本阶段不重跑；产物已在磁盘上",
        "# AMPLe partial: idea_A/baselines/AMPLe/official_reproduction/metrics.json",
        "# PP-bandit: .venv/bin/python idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/run_official_numerical.py",
        "# ncvxpp: PYTHONPATH=official_reproduction/mpi4py_stub python synthetic-iter/main.py",
        "```",
        "",
        "没有 Protocol A2 / B2 对照命令，因为适配器未实现。",
        "",
        "## 18. Artifact Index",
        "",
        md_table(
            ["路径", "角色"],
            [
                ["reports/v1.2/00_EXISTING_EVIDENCE_AUDIT.md", "已有证据审计"],
                ["reports/v1.2/01_PUBLISHED_BASELINE_REGISTRY.md", "发表基线登记"],
                ["reports/v1.2/02_ORIGINAL_REPRODUCTION.md", "原设定复现记录"],
                ["reports/v1.2/03_COMPARABILITY_AUDIT.md", "可比性锁；DIRECT=0 ADAPTED=0"],
                ["reports/v1.2/CCFA_PUBLISHED_BASELINE_COMPARISON.md", "本文件，主交付"],
                ["reports/v1.2/ORIGINAL_BASELINE_REPRODUCTION.md", "原设定复现汇总"],
                ["reports/v1.2/BENCHMARK_COMPATIBILITY.md", "可比性汇总"],
                ["reports/v1.2/A_B_RESEARCH_DECISION.md", "研究决策"],
                ["reports/v1.2/comparison_summary.json", "机器可读对照表；INCOMPARABLE 的 performance 为空"],
                [
                    "results/v1.2/idea_A/original/activeuf_selector_unit_check/",
                    "selector_unit_check JSON/CSV/log；不是对照格",
                ],
                ["idea_A/baselines/AMPLe/official_reproduction/metrics.json", "AMPLe partial L2"],
                [
                    "idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/metrics.json",
                    "PP-bandit PR 端点",
                ],
                ["idea_A/baselines/REPRODUCTION_AUDIT.json", "Idea A 基线审计"],
                ["idea_B/baselines/REPRODUCTION_AUDIT.json", "Idea B 基线审计（含 ncvxpp）"],
                ["idea_A/results/v1.1/TIER2_SUMMARY.json", "v1.1 内部经典对照；非 CCF-A 主表"],
                ["idea_B/results/v1.1/TIER2_SUMMARY.json", "v1.1 内部经典对照；非 CCF-A 主表"],
            ],
        ),
        "",
        "不提交：完整 ActiveUltraFeedback 克隆、`.venv`、`.isolated_baselines_venv`、`.cache`、v1.1 逐次 PNG、脏的 environment_manifest / smoke 报告。",
        "",
    ]
    return "\n".join(parts)


def _ncvx_table(ncvx_npy: dict[str, Any]) -> str:
    rows = []
    for name, rec in ncvx_npy.items():
        rows.append(
            [
                name,
                str(rec.get("iter_last", "NA")),
                fmt_num(rec.get("loss_last")),
                fmt_num(rec.get("train_acc_last")),
                fmt_num(rec.get("test_acc_last")),
            ]
        )
    return md_table(
        ["npy", "iter_last", "loss_last", "train_acc_last", "test_acc_last"],
        rows,
    )


def build_original_reproduction(artifacts: dict[str, Any], facts: dict[str, Any]) -> str:
    ample = ample_blob(artifacts)
    pp = pp_blob(artifacts)
    ncvx = ncvxpp_blob(artifacts)
    ncvx_what = ncvx.get("what_ran") or {}
    sel = artifacts["selector_metrics"] or {}
    parts = [
        header_block(
            "v1.2 原设定基线复现",
            artifacts,
            extra="本文件主要沿用 `02_ORIGINAL_REPRODUCTION.md` 与已有 AMPLe / PP-bandit / ncvxpp JSON。没有新的全文复现。",
        ),
        "",
        "## 总览",
        "",
        md_table(
            ["对象", "官方设定", "本地做了什么", "保真度", "论文数字能否对照"],
            [
                [
                    "AMPLe",
                    "README n_rounds=100, burnin=50000, volume",
                    "n_rounds=3, burnin 与 volume 不变",
                    "partial",
                    "论文无与本地 L2 可减的端点表；L2 只描述本次 3 轮",
                ],
                [
                    "PP-bandit",
                    "bandit-icml24.ipynb，Appendix H 图 4–6",
                    "同一算法单元 + numpy seed 0",
                    "数值脚本 complete；图端点未建立",
                    "论文图注只有横轴区间，没有端点",
                ],
                [
                    "ncvxpp",
                    "mpiexec -np 10 python main.py",
                    "rank=0 size=1 stub，eps∈{0,0.1,10,100}，batch=5",
                    "partial",
                    "论文图未数字化",
                ],
                ["HyPER", "TPU / vLLM / 8B–13B RM", "未跑", "blocked", "无本地值"],
                ["Drago", "无官方代码", "未实现", "blocked", "无"],
                [
                    "ActiveUF 全文",
                    "CUDA 13, vLLM 0.19, Skywork RM, Qwen3-32B oracle",
                    "未跑",
                    "blocked",
                    "无",
                ],
                [
                    "ActiveUF 选择器",
                    "官方 DRTS/DeltaUCB/InfoMax",
                    "隔离 venv unit check，合成 2×4 张量",
                    "DIAGNOSTIC import/call only",
                    "不是论文 RM/DPO 表",
                ],
                [
                    "UCB-OGD",
                    "Table 1 CIFAR/VOC/VQA/testbed",
                    "无官方代码；independent_paper_based not done",
                    "未复现",
                    "论文表不进对照列",
                ],
            ],
        ),
        "",
        f"主判定仍是 `{facts['verdict']}`。原设定复现成功不等于 head-to-head 成立。",
        "",
        "## AMPLe：官方 vs 本地",
        "",
        md_table(
            ["项", "官方 / README", "本地 JSON"],
            [
                ["commit", "`0cad341f6a9e77063ef451c27732bf5c1534c797`", "同上"],
                ["acquisition", "volume", str(ample.get("notes", "")).split(".")[0] if ample else "volume"],
                ["n_rounds", "100（README 示例）", "3"],
                ["burnin", "50000", "50000"],
                ["seed", "命令 --seed=1", "1（见审计）"],
                ["true_preference", "示例 0.2 0.7 0.1", json.dumps(ample.get("true_preference"))],
                ["l2_t0", "论文未给该次运行", fmt_num(nested(ample, "l2_to_true", "0"))],
                ["l2_t3", "论文未给该次运行", fmt_num(nested(ample, "l2_to_true", "3"))],
                ["wall_sec", "未给", fmt_num(ample.get("elapsed_wall_sec_approx"))],
            ],
        ),
        "",
        f"命令：`{ample.get('command')}`",
        "",
        "## PP-bandit：官方 vs 本地",
        "",
        md_table(
            ["项", "官方 notebook / 论文", "本地 JSON"],
            [
                ["commit", "`d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5`", "同上"],
                ["环境", "1-D Bernoulli PR，无 skill", "相同；源码无 skill"],
                ["seed", "notebook 无 seed", str(pp.get("numpy_seed"))],
                ["单次 500 步 PR(φ)", "图中无端点表", fmt_num(nested(pp, "single_run", "final_PR_phi"))],
                ["单次 500 步 PR(θ)", "图中无端点表", fmt_num(nested(pp, "single_run", "final_PR_theta"))],
                [
                    "10 初值 PR(φ) mean",
                    "无",
                    f"{fmt_num(nested(pp, 'multi_run', 'PR_phi_final_mean'))} (std {fmt_num(nested(pp, 'multi_run', 'PR_phi_final_std'))})",
                ],
                [
                    "10 初值 PR(θ) mean",
                    "无",
                    f"{fmt_num(nested(pp, 'multi_run', 'PR_theta_final_mean'))} (std {fmt_num(nested(pp, 'multi_run', 'PR_theta_final_std'))})",
                ],
                ["elapsed_sec", "未给", fmt_num(pp.get("elapsed_sec"))],
            ],
        ),
        "",
        "保真度评估：算法单元来自官方 notebook，数值可复跑；**不能**写成“复现了图 4–6 的数值”。",
        "",
        "## ncvxpp：官方 vs 本地",
        "",
        md_table(
            ["项", "官方", "本地"],
            [
                ["commit", "`ccbeeddfbf9237aa06dca3a1579529faabd8fe99`", "同上"],
                ["命令", "`cd synthetic-iter && mpiexec -np 10 python main.py`", ncvx_what.get("command", "NA")],
                ["进程数", "10", "1（mpi4py stub）"],
                ["eps", "官方脚本", json.dumps(ncvx_what.get("eps_list"))],
                ["batch", "官方脚本", str(ncvx_what.get("batch"))],
                ["wall_sec", "未在本机对比", fmt_num(ncvx_what.get("wall_sec"))],
                ["状态", "全文", str(ncvx.get("local_status"))],
            ],
        ),
        "",
        _ncvx_table(ncvx_what.get("npy") or {}),
        "",
        "失败 / blocker：没有 `mpi4py`，PATH 上没有 `mpicc` / `mpiexec`。`spam-iter` 依赖 torch，未跑。",
        "",
        "## ActiveUF 选择器 unit check",
        "",
        f"`status={sel.get('status')}`，`check_type={sel.get('check_type')}`，`comparability_cell={json.dumps(sel.get('comparability_cell'))}`，`protocol_a2={sel.get('protocol_a2')}`。",
        f"DRTS pairs `{json.dumps(nested(sel, 'selectors', 'drts', 'selected_pairs'))}`；"
        f"DeltaUCB `{json.dumps(nested(sel, 'selectors', 'deltaucb', 'selected_pairs'))}`；"
        f"InfoMax `{json.dumps(nested(sel, 'selectors', 'infomax', 'selected_pairs'))}`。",
        "主 venv 导入失败：`ModuleNotFoundError: No module named 'torch'`（见 `02_ORIGINAL_REPRODUCTION.md`）。",
        "",
        "## 其余 blocked",
        "",
        "HyPER、Drago、UCB-OGD、ActiveUF 全文：无失败日志中的指标，因为实验未启动。原因见 `02_ORIGINAL_REPRODUCTION.md`。",
        "",
        f"结论：有 partial / complete 的 **原设定** 产物，但 `{facts['verdict']}`。",
        "",
    ]
    return "\n".join(parts)


def build_compatibility(artifacts: dict[str, Any], facts: dict[str, Any]) -> str:
    parts = [
        header_block(
            "v1.2 Benchmark 兼容性",
            artifacts,
            extra="本文件建立在 `03_COMPARABILITY_AUDIT.md` 的锁上，不改口。",
        ),
        "",
        "## 锁",
        "",
        f"- `{facts['verdict']}`",
        f"- DIRECT = `{facts['n_direct']}`，ADAPTED = `{facts['n_adapted']}`",
        f"- Protocol A2 = `{facts['protocol_a2']}`",
        f"- Protocol B2 = `{facts['protocol_b2']}`",
        "",
        "## 目标兼容性",
        "",
        md_table(
            ["Cell", "发表目标", "我们的目标", "判定"],
            [
                ["C-AMPLe-A", "单纯形偏好向量", "标量泛函 RMSE", "INCOMPARABLE"],
                ["C-HyPER-A", "human/LM 路由 + RewardBench", "审计泛函", "INCOMPARABLE"],
                ["C-Drago-A", "全局兼容奖励识别", "有限预算审计", "INCOMPARABLE"],
                ["C-ActiveUF-full-A", "RM/DPO/judge 偏好数据", "总体 RMSE", "INCOMPARABLE"],
                ["C-A2-Av11-ActiveUF", "选回复对", "Bernoulli 审计概率", "INCOMPARABLE"],
                ["C-UCBOGD-vision", "多源稳健视觉/VQA", "审计 / 技能 MSE", "INCOMPARABLE"],
                ["C-PPbandit-B", "一维 performative risk", "worst-skill MSE", "INCOMPARABLE"],
                ["C-B2-Bv11-ZO", "一维两点零阶", "技能分配", "INCOMPARABLE"],
                ["C-ncvxpp-B", "performative SGD loss/acc", "observability Gramian", "INCOMPARABLE"],
            ],
        ),
        "",
        "## 信息访问",
        "",
        "- ActiveUF 选择器需要 live RM UQ 张量。本仓库没有。HF `combined` 成对列缺少 `lower_bounds` / `upper_bounds` 与落选回复，offline replay 不成立。judge 分数不是人类偏好。",
        "- A-v1.1 看见协变量、已知 `q`、已审计残差；不看见隐藏 `Y`。",
        "- PP-bandit 看见完整标量 PR；没有个体特征或技能标签。",
        "- 若给 B-v1.1 技能标签而官方方法没有，信息不对齐。",
        "",
        "## 预算会计",
        "",
        "审计比例 `b`、偏好对、查询轮数、图像张数、SGD 迭代不是同一列。burn-in 50000 不并进 `b`。",
        "",
        "## 适配细节",
        "",
        "- Protocol A2 适配器：**未实现**。映射 M1–M5 否决。",
        "- Protocol B2 部分反馈适配器：**未实现**。原环境无分配动作。",
        "- `volume_adapted_ipw` 仍是合成启发式，不叫 AMPLe。",
        "- Uncertainty Sampling 保持原名。",
        "- selector_unit_check 允许，但是 DIAGNOSTIC，不升格 ADAPTED。",
        "- UCB-OGD logistic DIAGNOSTIC：**未实例化**。即使实例化也不能宣称击败论文。",
        "",
        "完整 13 维表见 `03_COMPARABILITY_AUDIT.md` 第 3 节。",
        "",
    ]
    return "\n".join(parts)


def build_decision(artifacts: dict[str, Any], facts: dict[str, Any]) -> str:
    internal = v11_internal(artifacts)
    a_static = internal["a_static"]
    a_loop = internal["a_loop"]
    b_env1 = internal["b_env1_worst"]
    parts = [
        header_block(
            "v1.2 Idea A / Idea B 研究决策",
            artifacts,
            extra="PROMISING 不能表示只打败均匀随机。本文件也不把内部经典对照改写成发表基线胜利。",
        ),
        "",
        f"总判定：`{facts['verdict']}`。v1.3 算法迭代 justified = `{json.dumps(facts['v13_algorithmic_iteration'])}`。",
        "",
        "## Idea A",
        "",
        "### Current evidence / 当前证据",
        "",
        "- 已发表对照：Protocol A2 **BLOCKED / INCOMPARABLE**。无 DIRECT，无可辩护 ADAPTED。",
        "- 选择器 unit check 只说明隔离 venv 能调用官方 DRTS / DeltaUCB / InfoMax。",
        f"- v1.1 内部：静态 MNAR `b=0.05` n=`{a_static.get('n_seeds')}`，A-v1.1c RMSE `{fmt_num(a_static.get('rmse_candidate'))}` vs Uniform DR `{fmt_num(a_static.get('rmse_baseline'))}`，`interval_side={a_static.get('interval_side')}`。闭环 n=`{a_loop.get('n_seeds')}`：`{fmt_num(a_loop.get('rmse_candidate'))}` vs `{fmt_num(a_loop.get('rmse_baseline'))}`，`interval_side={a_loop.get('interval_side')}`。判定 **NOT PROMISING**。",
        "- identification-via-audit 在 v1.1 静态格上仍成立（observed-only 差于 Uniform DR）。那是审计通道，不是主动选择器胜利。",
        "",
        "### Strongest published competitor / 最强已发表竞争者",
        "",
        "名义上最接近的是 ActiveUltraFeedback 的 **DRTS** 与 **DeltaUCB**（ICML 2026 Main）。InfoMax 是同一仓库里的已有 dueling-bandit 对照，不是该文新算法。AMPLe / HyPER 目标更远。",
        "",
        "### Strongest valid experimental result / 最强合法实验结果",
        "",
        "相对这些发表方法：**没有合法实验结果。** 最强原设定产物是 AMPLe 3 轮 volume 的 L2（见 ORIGINAL 报告），以及 selector_unit_check 的合成下标；两者都不是 A-v1.1 vs 发表方法。",
        "",
        "### Most serious limitation / 最严重限制",
        "",
        "没有从 `q_i` 方差分配到“选哪一对响应”的、保持动作 / 信息 / 预算的映射；也没有 live RM UQ。本机无法跑 ActiveUF 全文。",
        "",
        "### Whether we exceed the published method / 是否超过发表方法",
        "",
        "**NO valid comparison。** 不能回答是否超过 DRTS / DeltaUCB / AMPLe / HyPER。",
        "",
        "### Whether a meaningful research gap remains / 是否仍有研究缺口",
        "",
        "识别意义上的 MNAR 审计通道在 v1.1 内部仍可见，但 **相对 CCF-A 主动学习论文的缺口尚未被合法实验定位。** 在对照路径出现之前，不能把缺口写成“我们补上了 DRTS 做不到的内生删失”。",
        "",
        "### Whether to invest in a new method version / 是否投资 v1.3",
        "",
        "**否。** 合法对照路径当前不存在（本 Mac 无 RM UQ / 无已通过审计的 skill-free 映射）。v1.1 对经典对照已是 NOT PROMISING，不支持再发明分配规则。",
        "",
        "### The one most valuable next experiment / 下一步最有价值的实验",
        "",
        "不是新算法。若未来有带官方 UQ 的 completion 池，先重新打开 Stage 3：映射若仍不成立，对照继续 BLOCKED。在那之前不跑 10/30-seed。",
        "",
        "## Idea B",
        "",
        "### Current evidence / 当前证据",
        "",
        "- 已发表对照：Protocol B2 **INCOMPARABLE**。PP-bandit 原设定数值可沿用，但与技能分配不可比。",
        f"- v1.1 内部 Env1 n=`{b_env1.get('n_seeds')}`：B-v1.1c vs E-optimal worst-skill `interval_side={b_env1.get('interval_side')}`，diff `{fmt_num(paired_diff(b_env1))}`。判定 **NOT PROMISING**。",
        "",
        "### Strongest published competitor / 最强已发表竞争者",
        "",
        "名义上最接近的是 **PP-bandit** 两层零阶重参数化（ICML 2024）。ncvxpp 的 performative SGD 是另一目标，且全文 MPI blocked。",
        "",
        "### Strongest valid experimental result / 最强合法实验结果",
        "",
        "相对发表方法：**没有合法对照结果。** 最强原设定产物是 PP-bandit 1-D `final_PR_phi = "
        f"{fmt_num(nested(pp_blob(artifacts), 'single_run', 'final_PR_phi'))}` 等端点，不能与 worst-skill MSE 做差。",
        "",
        "### Most serious limitation / 最严重限制",
        "",
        "官方一维环境没有技能组，也没有可分配的反馈动作。最小缺失层只能造成标量缺失，不能承载 EqualSkill / E-optimal / worst-skill look-ahead。",
        "",
        "### Whether we exceed the published method / 是否超过发表方法",
        "",
        "**NO valid comparison。**",
        "",
        "### Whether a meaningful research gap remains / 是否仍有研究缺口",
        "",
        "闭环可观测性崩溃是 v1.1 合成现象，尚未在发表 performative benchmark 的合法扩展上定位。在不加私有技能标签的前提下，当前环境 **撑不起** 该缺口的 head-to-head。",
        "",
        "### Whether to invest in a new method version / 是否投资 v1.3",
        "",
        "**否。** 没有 skill-free 的合法对照路径。E-optimal 已是更强的内部主指标基线。",
        "",
        "### The one most valuable next experiment / 下一步最有价值的实验",
        "",
        "不要给 PP-bandit 私加技能标签。若存在作者提供的、本身带部分反馈动作的官方环境，再重新做 Stage 3。否则不进入 v1.3 算法。",
        "",
        "## 联合建议",
        "",
        "两个 idea 都不进入 v1.3 算法迭代。需要的是合法发表对照路径，而不是新的分配启发式。",
        "",
    ]
    return "\n".join(parts)


def generate_all() -> dict[str, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    facts = lock_facts(artifacts)
    outputs = {
        "summary": write_comparison_summary(artifacts, facts),
        "main": REPORTS_DIR / "CCFA_PUBLISHED_BASELINE_COMPARISON.md",
        "original": REPORTS_DIR / "ORIGINAL_BASELINE_REPRODUCTION.md",
        "compat": REPORTS_DIR / "BENCHMARK_COMPATIBILITY.md",
        "decision": REPORTS_DIR / "A_B_RESEARCH_DECISION.md",
    }
    outputs["main"].write_text(build_main_report(artifacts, facts) + "\n", encoding="utf-8")
    outputs["original"].write_text(
        build_original_reproduction(artifacts, facts) + "\n", encoding="utf-8"
    )
    outputs["compat"].write_text(build_compatibility(artifacts, facts) + "\n", encoding="utf-8")
    outputs["decision"].write_text(build_decision(artifacts, facts) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    outputs = generate_all()
    for key, path in outputs.items():
        print(f"{key}: {rel(path)}")
    print(VERDICT)
    print(f"DIRECT={N_DIRECT} ADAPTED={N_ADAPTED}")


if __name__ == "__main__":
    main()
