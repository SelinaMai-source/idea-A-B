"""Stage 3 comparability lock and official selector unit check.

No DIRECT/ADAPTED adapter is implemented. These tests lock that fact and,
when the isolated torch venv is present, verify that the unit check calls
the official ActiveUF class files rather than a rewrite.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT = REPO_ROOT / "reports" / "v1.2" / "03_COMPARABILITY_AUDIT.md"
ISOLATED_PYTHON = REPO_ROOT / ".isolated_baselines_venv" / "bin" / "python"
CLONE_ROOT = REPO_ROOT / "idea_A" / "baselines" / "ActiveUltraFeedback"
RUNNER = REPO_ROOT / "idea_A" / "experiments" / "v1.2" / "run_activeuf_selector_unit_check.py"
UNIT_CHECK_DIR = (
    REPO_ROOT
    / "results"
    / "v1.2"
    / "idea_A"
    / "original"
    / "activeuf_selector_unit_check"
)
OFFICIAL = {
    "drts": CLONE_ROOT / "activeuf" / "acquisition_function" / "drts.py",
    "deltaucb": CLONE_ROOT / "activeuf" / "acquisition_function" / "deltaucb.py",
    "infomax": CLONE_ROOT / "activeuf" / "acquisition_function" / "infomax.py",
}


def test_audit_exists_and_declares_no_direct_or_adapted_cells():
    text = AUDIT.read_text(encoding="utf-8")
    assert "没有任何格子被判为 DIRECT 或可辩护的 ADAPTED" in text
    assert "DIRECT 格子数：0。ADAPTED 格子数：0。" in text
    assert "C-A2-Av11-ActiveUF" in text
    assert "C-B2-Bv11-ZO" in text
    assert "INCOMPARABLE" in text
    assert "Protocol A2：**BLOCKED**" in text
    assert "不实现 Protocol A2 对照适配器" in text
    assert "B2 保持 INCOMPARABLE" in text


def test_pre_registered_mapping_is_rejected_in_audit():
    text = AUDIT.read_text(encoding="utf-8")
    assert "预先登记的映射" in text
    assert "M1." in text and "M2." in text and "M5." in text
    assert "否决" in text
    assert "live RM UQ" in text
    assert "不写启发式替身" in text


def test_no_protocol_a2_or_b2_comparison_adapter_modules():
    forbidden_stems = {
        "protocol_a2_adapter",
        "protocol_b2_adapter",
        "drts_heuristic",
        "deltaucb_heuristic",
        "hyper_adapted",
    }
    found = []
    for folder in (
        REPO_ROOT / "idea_A" / "experiments" / "v1.2",
        REPO_ROOT / "idea_A" / "baselines",
        REPO_ROOT / "idea_B" / "experiments",
        REPO_ROOT / "idea_B" / "baselines",
    ):
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.py"):
            if "ActiveUltraFeedback" in path.parts:
                continue
            if path.stem in forbidden_stems:
                found.append(str(path.relative_to(REPO_ROOT)))
    assert found == []


def test_uncertainty_sampling_is_not_renamed_in_v12_audit():
    text = AUDIT.read_text(encoding="utf-8")
    assert "没有把 Uncertainty Sampling 改名为 DRTS / DeltaUCB / HyPER" in text


@pytest.mark.skipif(not ISOLATED_PYTHON.is_file(), reason="isolated torch venv missing")
@pytest.mark.skipif(not OFFICIAL["drts"].is_file(), reason="ActiveUF clone missing")
def test_selector_unit_check_calls_official_files_same_pool_no_oracle(tmp_path):
    out_dir = tmp_path / "activeuf_selector_unit_check"
    proc = subprocess.run(
        [str(ISOLATED_PYTHON), str(RUNNER), "--seed", "0", "--out-dir", str(out_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((out_dir / "config.yaml").read_text(encoding="utf-8"))
    env = json.loads((out_dir / "environment.json").read_text(encoding="utf-8"))

    assert metrics["status"] == "ok"
    assert metrics["check_type"] == "selector_unit_check"
    assert metrics["comparability_cell"] is None
    assert metrics["protocol_a2"] == "BLOCKED_INCOMPARABLE"
    assert metrics["same_candidate_pool"] is True
    assert metrics["budget_pairs_per_prompt"] == 1
    assert metrics["oracle_labels_in_acquisition_scores"] is False
    assert metrics["vllm_imported"] is False
    assert metrics["all_official_files"] is True
    assert config["comparability"] == "not_a_comparison_cell"
    assert env["vllm_imported"] is False

    pool = metrics["candidate_pool"]
    n_prompts = pool["n_prompts"]
    n_comp = pool["n_completions_per_prompt"]
    for key, official in OFFICIAL.items():
        rec = metrics["selectors"][key]
        assert Path(rec["file"]).resolve() == official.resolve()
        assert rec["file_is_official"] is True
        assert rec["n_pairs"] == n_prompts
        assert rec["oracle_labels_passed"] is False
        assert rec["call_args"] == ["rewards", "lower_bounds", "upper_bounds"]
        assert rec["pair_indices_in_range"] is True
        assert rec["n_pairs"] == metrics["budget_pairs_per_prompt"] * n_prompts
        for i, j in rec["selected_pairs"]:
            assert 0 <= i < n_comp and 0 <= j < n_comp
            assert i != j

    artifacts = {
        "config.yaml",
        "metrics.json",
        "metrics.csv",
        "run.log",
        "environment.json",
    }
    assert artifacts <= {p.name for p in out_dir.iterdir()}


def test_committed_unit_check_json_matches_lock_if_present():
    metrics_path = UNIT_CHECK_DIR / "metrics.json"
    if not metrics_path.is_file():
        pytest.skip("unit-check JSON not written yet")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    config = yaml.safe_load((UNIT_CHECK_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert metrics["check_type"] == "selector_unit_check"
    assert metrics["comparability_cell"] is None
    assert config["comparability"] == "not_a_comparison_cell"
    if metrics.get("status") == "ok":
        assert metrics["all_official_files"] is True
        assert metrics["oracle_labels_in_acquisition_scores"] is False
        assert metrics["vllm_imported"] is False


def test_main_venv_still_has_no_torch():
    proc = subprocess.run(
        [sys.executable, "-c", "import torch"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "No module named 'torch'" in (proc.stderr + proc.stdout)
