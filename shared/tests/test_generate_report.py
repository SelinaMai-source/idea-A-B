"""Report generation reads JSON from disk and does not invent metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_MODULE_PATH = ROOT / "scripts" / "generate_report.py"
_SPEC = importlib.util.spec_from_file_location("idea_ab_generate_report", _MODULE_PATH)
gen = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(gen)


def test_primary_sources_exist():
    missing = [rel for rel, path in gen.PRIMARY_SOURCES.items() if not path.exists()]
    assert missing == [], missing


def test_verdicts_follow_predeclared_rules():
    artifacts = gen.load_artifacts()
    verdicts = gen.compute_verdicts(artifacts)
    assert verdicts["idea_A"] == "INCONCLUSIVE"
    assert verdicts["idea_B"] == "INCONCLUSIVE"
    assert verdicts["A"]["identification"] is True
    assert verdicts["A"]["allocation_supported"] is False
    assert verdicts["B"]["collapse_supported"] is False
    assert verdicts["B"]["info_supported"] is True
    assert verdicts["B"]["perf_supported"] is False


def test_numbers_copied_from_json_not_hand_typed():
    artifacts = gen.load_artifacts()
    a = artifacts["idea_A_smoke"]
    observed = a["key_findings"]["mnar_observed_only_vs_audit_ipw"]["b=0.05"]["observed_only_bias_mean"]
    neyman = a["key_findings"]["mnar_observed_only_vs_audit_ipw"]["b=0.05"]["neyman_ipw_bias_mean"]
    text = gen.fmt_num(observed)
    parsed = float(text)
    assert abs(parsed - float(observed)) <= 5e-6 * max(1.0, abs(float(observed)))
    assert float(observed) > 1.0
    assert abs(float(neyman)) < 0.25


def test_generate_all_writes_three_chinese_reports(tmp_path=None):
    outputs = gen.generate_all(write_env=True)
    main = outputs["main"].read_text(encoding="utf-8")
    audit = outputs["audit"].read_text(encoding="utf-8")
    novelty = outputs["novelty"].read_text(encoding="utf-8")
    for heading in gen.MAIN_REPORT_SECTIONS:
        assert f"## {heading}" in main, heading
    assert "INCONCLUSIVE" in main
    assert "这不是新算法" in main or "不是新算法" in main
    artifacts = gen.load_artifacts()
    observed = artifacts["idea_A_smoke"]["key_findings"]["mnar_observed_only_vs_audit_ipw"]["b=0.05"][
        "observed_only_bias_mean"
    ]
    assert gen.fmt_num(observed) in main
    assert "partial" in audit
    assert "blocked" in audit
    assert "complete" in audit
    assert "识别" in novelty
    assert "不得" in main or "不能" in main
    assert "n_seeds=3" in main
    env = ROOT / "reports" / "environment_manifest.json"
    assert env.is_file()
    payload = env.read_text(encoding="utf-8")
    assert "python" in payload
