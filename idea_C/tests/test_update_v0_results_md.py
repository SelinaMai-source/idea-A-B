"""Updater must survive empty/raw dirs and never invent 0 for missing metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "update_v0_results_md.py"
MD = ROOT / "v0实验版本+结果.md"

SPEC = importlib.util.spec_from_file_location("update_v0_results_md", SCRIPT)
assert SPEC and SPEC.loader
UPD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPD)


MINIMAL_MD = """# t

<!-- TABLE_ID: demo -->

|版本|内容|Data|FDR↓|结果解释|
|---|---|---|---|---|
|v0_a|placeholder|demo|NOT_RUN|尚未运行|
"""


def test_empty_raw_does_not_crash(tmp_path: Path) -> None:
    md = tmp_path / "t.md"
    md.write_text(MINIMAL_MD, encoding="utf-8")
    raw = tmp_path / "raw"
    raw.mkdir()
    seen = tmp_path / "seen.json"
    assert UPD.update_markdown(md, raw, seen) == 0
    assert "NOT_RUN" in md.read_text(encoding="utf-8")
    assert not seen.exists()


def test_missing_raw_dir_does_not_crash(tmp_path: Path) -> None:
    md = tmp_path / "t.md"
    md.write_text(MINIMAL_MD, encoding="utf-8")
    raw = tmp_path / "does-not-exist"
    seen = tmp_path / "seen.json"
    assert UPD.update_markdown(md, raw, seen) == 0


def test_empty_run_subdir_and_bad_json_skipped(tmp_path: Path) -> None:
    md = tmp_path / "t.md"
    md.write_text(MINIMAL_MD, encoding="utf-8")
    raw = tmp_path / "raw"
    (raw / "empty_run").mkdir(parents=True)
    bad = raw / "bad_json"
    bad.mkdir()
    (bad / "metrics.json").write_text("{not json", encoding="utf-8")
    empty_json = raw / "empty_json"
    empty_json.mkdir()
    (empty_json / "metrics.json").write_text("\n", encoding="utf-8")
    header_only = raw / "csv_header"
    header_only.mkdir()
    (header_only / "metrics.csv").write_text("table_id,FDR↓\n", encoding="utf-8")
    seen = tmp_path / "seen.json"
    assert UPD.update_markdown(md, raw, seen) == 0
    text = md.read_text(encoding="utf-8")
    assert text.count("|v0_a|") == 1


def test_appends_run_without_overwriting_explanation(tmp_path: Path) -> None:
    md = tmp_path / "t.md"
    md.write_text(MINIMAL_MD, encoding="utf-8")
    raw = tmp_path / "raw" / "run1"
    raw.mkdir(parents=True)
    (raw / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "run1",
                "table_id": "demo",
                "version": "v0_a",
                "content": "real run seed 0",
                "data": "demo",
                "metrics": {"FDR↓": "0.04"},
                "result_explanation": "人工稍后改",
            }
        ),
        encoding="utf-8",
    )
    seen = tmp_path / "seen.json"
    assert UPD.update_markdown(md, raw.parent, seen) == 0
    text = md.read_text(encoding="utf-8")
    assert "尚未运行" in text
    assert "real run seed 0" in text
    assert "0.04" in text
    assert UPD.update_markdown(md, raw.parent, seen) == 0
    assert text.count("real run seed 0") == md.read_text(encoding="utf-8").count("real run seed 0") == 1


def test_missing_metric_is_not_run_not_zero(tmp_path: Path) -> None:
    md = tmp_path / "t.md"
    md.write_text(MINIMAL_MD, encoding="utf-8")
    raw = tmp_path / "raw" / "run2"
    raw.mkdir(parents=True)
    (raw / "metrics.json").write_text(
        json.dumps(
            {
                "run_id": "run2",
                "table_id": "demo",
                "version": "v0_b",
                "content": "partial",
                "data": "demo",
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    seen = tmp_path / "seen.json"
    assert UPD.update_markdown(md, raw.parent, seen) == 0
    lines = [ln for ln in md.read_text(encoding="utf-8").splitlines() if ln.startswith("|v0_b|")]
    assert lines
    cells = [c.strip() for c in lines[0].strip("|").split("|")]
    assert cells[3] == "NOT_RUN"


def test_repo_markdown_has_eight_sections_and_not_run() -> None:
    text = MD.read_text(encoding="utf-8")
    for heading in (
        "## 1. AAAI Robust Gaussian",
        "## 2. AAAI Heavy-tailed",
        "## 3. e-PS Gaussian",
        "## 4. e-PS Student-t",
        "## 5. SCORE Gaussian Mixture",
        "## 6. UAI Local-dependent",
        "## 7. Overlap A",
        "## 8. Overlap B",
    ):
        assert heading in text
    assert "NOT_RUN" in text
    assert "<!-- TABLE_ID: overlap_a_clean -->" in text
    assert "<!-- TABLE_ID: overlap_a_contaminated -->" in text
    assert "<!-- TABLE_ID: overlap_b_streaming -->" in text


def test_updater_on_real_empty_raw() -> None:
    raw = ROOT / "results" / "v0" / "raw"
    seen = ROOT / "results" / "v0" / "summaries" / "ingested_run_ids.json"
    before = MD.read_text(encoding="utf-8")
    assert UPD.update_markdown(MD, raw, seen) == 0
    assert MD.read_text(encoding="utf-8") == before
