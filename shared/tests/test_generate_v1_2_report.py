"""v1.2 reports copy locked-audit JSON and do not invent CCF-A head-to-head metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = ROOT / "scripts" / "generate_v1_2_report.py"
_SPEC = importlib.util.spec_from_file_location("idea_ab_generate_v1_2_report", _MODULE_PATH)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)


def test_primary_sources_exist():
    missing = [rel for rel, path in gen.PRIMARY_SOURCES.items() if not path.exists()]
    assert missing == [], missing


def test_lock_facts_zero_direct_adapted():
    artifacts = gen.load_artifacts()
    facts = gen.lock_facts(artifacts)
    assert facts["verdict"] == "PUBLISHED BASELINE COMPARISON NOT ESTABLISHED"
    assert facts["n_direct"] == 0
    assert facts["n_adapted"] == 0
    assert facts["protocol_a2"] == "BLOCKED_INCOMPARABLE"
    assert facts["protocol_b2"] == "INCOMPARABLE"
    assert facts["v13_algorithmic_iteration"] is False


def test_incomparable_rows_have_empty_performance():
    artifacts = gen.load_artifacts()
    rows = gen.build_comparison_rows(artifacts)
    assert rows, "expected comparison_summary rows"
    classifications = {row["classification"] for row in rows}
    assert "DIRECT" not in classifications
    assert "ADAPTED" not in classifications
    for row in rows:
        assert row["performance"] is None, row["run_id"]
        if row["classification"] == "INCOMPARABLE":
            assert "performance" in row and row["performance"] is None


def test_selector_unit_check_is_diagnostic_not_head_to_head():
    artifacts = gen.load_artifacts()
    rows = gen.build_comparison_rows(artifacts)
    unit = next(row for row in rows if row["run_id"] == "activeuf_selector_unit_check")
    assert unit["classification"] == "DIAGNOSTIC"
    assert unit["performance"] is None
    assert unit["protocol"] is None
    metrics = artifacts["selector_metrics"]
    assert metrics["comparability_cell"] is None
    assert metrics["protocol_a2"] == "BLOCKED_INCOMPARABLE"
    assert unit["diagnostic_value"]["selected_pairs"]["drts"] == metrics["selectors"]["drts"][
        "selected_pairs"
    ]


def test_generate_all_writes_chinese_v1_2_reports():
    outputs = gen.generate_all()
    main = outputs["main"].read_text(encoding="utf-8")
    original = outputs["original"].read_text(encoding="utf-8")
    compat = outputs["compat"].read_text(encoding="utf-8")
    decision = outputs["decision"].read_text(encoding="utf-8")
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))

    for heading in gen.MAIN_REPORT_SECTIONS:
        assert f"## {heading}" in main, heading

    assert gen.VERDICT in main
    assert gen.VERDICT == summary["verdict"]
    assert summary["n_scientifically_valid_direct_comparisons"] == 0
    assert summary["n_scientifically_defensible_adapted_comparisons"] == 0
    assert "NOT PROMISING" in main
    assert "不得" in main or "不能" in main
    assert "Idea A / Idea B 在本阶段 **均不得** 称为 PROMISING" in main
    assert "标为 PROMISING" in main
    assert "v1.3" in main
    assert "不成立" in main

    artifacts = gen.load_artifacts()
    ample = artifacts["ample_metrics"]
    l2_t3 = ample["l2_to_true"]["3"]
    assert gen.fmt_num(l2_t3) in main
    assert gen.fmt_num(l2_t3) in original
    pp = artifacts["pp_metrics"]
    assert gen.fmt_num(pp["single_run"]["final_PR_phi"]) in original
    ncvx = next(
        item
        for item in artifacts["idea_B_audit"]["baselines"]
        if "ncvxpp" in str(item.get("official_source", "")).lower()
        or "Nonconvex" in str(item.get("paper", ""))
    )
    ncvx_loss = ncvx["what_ran"]["npy"]["eps0-batch5-res0.npy"]["loss_last"]
    assert gen.fmt_num(ncvx_loss) in original

    internal = gen.v11_internal(artifacts)
    static = internal["a_static"]
    assert gen.fmt_num(static["rmse_candidate"]) in main
    # Primary Idea A table must not put that RMSE in the Performance column.
    table_start = main.index("## 7. Idea A Head-to-Head Results")
    table_end = main.index("## 8. Idea B Head-to-Head Results")
    table_block = main[table_start:table_end]
    performance_cells = []
    for line in table_block.splitlines():
        if not line.startswith("|") or line.startswith("| Method") or "---" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 8:
            performance_cells.append(cells[5])
    assert performance_cells, "expected Idea A primary table rows"
    assert all(cell == "" for cell in performance_cells), performance_cells
    assert "INCOMPARABLE" in table_block
    assert "no valid published head-to-head" in table_block

    assert "NO valid comparison" in decision
    assert "否" in decision
    assert compat.count("INCOMPARABLE") >= 8
    for row in summary["rows"]:
        if row["classification"] == "INCOMPARABLE":
            assert row["performance"] is None


def test_v11_internal_numbers_match_tier2_json():
    artifacts = gen.load_artifacts()
    internal = gen.v11_internal(artifacts)
    static = internal["a_static"]
    loop = internal["a_loop"]
    env1 = internal["b_env1_worst"]
    assert static["interval_side"] == "covers_zero"
    assert loop["interval_side"] == "entirely_positive"
    assert env1["interval_side"] == "entirely_positive"
    assert static["rmse_candidate"] == artifacts["idea_A_tier2"]["static"]["mnar"]["b=0.05"][
        "paired_vs_uniform_dr"
    ]["v1.1c"]["rmse_candidate"]
    assert loop["rmse_candidate"] == artifacts["idea_A_tier2"]["continual"]["mnar_audit"][
        "b=0.05"
    ]["paired_vs_uniform_dr"]["v1.1c"]["rmse_candidate"]
