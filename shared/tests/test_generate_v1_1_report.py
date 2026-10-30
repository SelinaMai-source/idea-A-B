"""v1.1 report generation reads JSON from disk and does not invent metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = ROOT / "scripts" / "generate_v1_1_report.py"
_SPEC = importlib.util.spec_from_file_location("idea_ab_generate_v1_1_report", _MODULE_PATH)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)


def test_primary_sources_exist():
    missing = [rel for rel, path in gen.PRIMARY_SOURCES.items() if not path.exists()]
    assert missing == [], missing


def test_verdicts_follow_frozen_protocol():
    artifacts = gen.load_artifacts()
    verdicts = gen.compute_verdicts(artifacts)
    assert verdicts["idea_A"] != "PROMISING"
    assert verdicts["idea_B"] != "PROMISING"
    assert verdicts["idea_A"] in {"INCONCLUSIVE", "NOT PROMISING"}
    assert verdicts["idea_B"] in {"INCONCLUSIVE", "NOT PROMISING"}
    assert verdicts["a_promising_allowed"] is False
    assert verdicts["b_promising_allowed"] is False
    assert verdicts["identification_v11_supported"] is True
    assert verdicts["sequential_allocation_supported"] is False
    assert verdicts["exploratory_10pct_failed"] is True
    assert verdicts["worst_vs_e_worse"] is True
    assert verdicts["hb1_all_negative"] is False
    assert verdicts["env3_inconclusive"] is True


def test_numbers_copied_from_tier2_json():
    artifacts = gen.load_artifacts()
    a2 = artifacts["idea_A_tier2"]
    paired = a2["static"]["mnar"]["b=0.05"]["paired_vs_uniform_dr"]["v1.1c"]
    text = gen.fmt_num(paired["rmse_candidate"])
    parsed = float(text)
    assert abs(parsed - float(paired["rmse_candidate"])) <= 5e-6 * max(1.0, abs(float(paired["rmse_candidate"])))
    assert paired["interval_side"] == "covers_zero"
    loop = a2["continual"]["mnar_audit"]["b=0.05"]["paired_vs_uniform_dr"]["v1.1c"]
    assert loop["interval_side"] == "entirely_positive"
    b2 = artifacts["idea_B_tier2"]
    worst = b2["cells"]["Env1"]["paired_vs_E_optimal"]["B-v1.1c"]["late_worst_skill_mse"]
    assert worst["interval_side"] == "entirely_positive"


def test_generate_all_writes_chinese_v1_1_reports():
    outputs = gen.generate_all()
    main = outputs["main"].read_text(encoding="utf-8")
    baseline = outputs["baseline"].read_text(encoding="utf-8")
    novelty = outputs["novelty"].read_text(encoding="utf-8")
    nxt = outputs["next"].read_text(encoding="utf-8")
    for heading in gen.MAIN_REPORT_SECTIONS:
        assert f"## {heading}" in main, heading
    artifacts = gen.load_artifacts()
    verdicts = gen.compute_verdicts(artifacts)
    assert verdicts["idea_A"] != "PROMISING"
    assert verdicts["idea_B"] != "PROMISING"
    assert f"**Idea A v1.1 状态：`{verdicts['idea_A']}`**" in main
    assert f"**Idea B v1.1 状态：`{verdicts['idea_B']}`**" in main
    assert verdicts["idea_A"] in main
    assert verdicts["idea_B"] in main
    paired = artifacts["idea_A_tier2"]["static"]["mnar"]["b=0.05"]["paired_vs_uniform_dr"]["v1.1c"]
    assert gen.fmt_num(paired["rmse_candidate"]) in main
    assert gen.fmt_num(paired["rmse_baseline"]) in main
    assert "partial" in baseline
    assert "blocked" in baseline
    assert "complete" in baseline
    assert "NOT DIRECTLY COMPARABLE" in baseline
    assert "识别" in novelty or "identification" in novelty.lower()
    assert "不要" in nxt
    protocol = ROOT / "reports" / "V1_1_EVALUATION_PROTOCOL.md"
    audit = ROOT / "reports" / "V1_TO_V1_1_AUDIT.md"
    assert protocol.is_file()
    assert audit.is_file()
    assert outputs["pointer_A"].is_file()
    assert outputs["pointer_B"].is_file()
    assert "run_A_v1_1.sh" in main
    assert "generate_v1_1_report.py" in main
