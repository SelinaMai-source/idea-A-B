import json
import subprocess
import sys
from pathlib import Path

from policies.diffusion_policy_image.adapter import probe
from reporting.export_results import export_rows
from reporting.update_markdown import append_row

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def test_policy_probe_does_not_substitute_a_backbone():
    report = probe()
    assert report["status"] in {"BLOCKED_NO_CUDA", "BLOCKED_NO_CHECKPOINT", "CHECKPOINT_PRESENT_TRAINING_NOT_RUN"}
    assert report["substituted_backbone"] is None
    assert report["family"] == "image_cnn_conditional_unet_diffusion_policy"
    assert report["lowdim_is_formal_image_result"] is False
    assert "octo" in report["forbidden_substitutes"]


def test_markdown_append_and_csv_export(tmp_path: Path):
    doc = tmp_path / "results.md"
    doc.write_text(
        "<!-- TABLE_ID: demo -->\n\n|版本|状态|\n|---|---|\n|kept|NOT_RUN|\n",
        encoding="utf-8",
    )
    append_row(doc, "demo", ["SYNTHETIC_INTERFACE_SMOKE", "not a paper number"])
    text = doc.read_text(encoding="utf-8")
    assert "|kept|NOT_RUN|" in text
    assert "SYNTHETIC_INTERFACE_SMOKE" in text
    append_row(doc, "demo", ["SYNTHETIC_INTERFACE_SMOKE", "duplicate"])
    assert doc.read_text(encoding="utf-8").count("SYNTHETIC_INTERFACE_SMOKE") == 1
    status = export_rows([{"method": "random", "final_success_rate": "NOT_RUN"}], tmp_path)
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "metrics.jsonl").is_file()
    assert "parquet" in status


def test_cli_help_and_dry_run(tmp_path: Path):
    scripts = [
        "run_original_baselines.py",
        "run_unified_baselines.py",
        "run_online_baselines.py",
        "run_continual_learning.py",
        "summarize_results.py",
    ]
    for name in scripts:
        completed = subprocess.run(
            [PYTHON, str(ROOT / "scripts" / name), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "--dry-run" in completed.stdout
        assert "--resume" in completed.stdout
    out = tmp_path / "plan"
    completed = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_unified_baselines.py"),
            "--method",
            "cupid",
            "--benchmark",
            "robomimic_lift_mh",
            "--seed",
            "0",
            "--dry-run",
            "--resume",
            "--output-dir",
            str(out),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["will_train"] is False
    assert plan["will_download"] is False
    assert plan["paper_result"] is False
    assert plan["planned_status"] in {"BLOCKED", "SOURCE_UNAVAILABLE", "NOT_IMPLEMENTED"}
    reference = subprocess.run(
        [
            PYTHON,
            str(ROOT / "references" / "hinflow" / "run_reference.py"),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "hin"),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert reference.returncode == 0, reference.stderr


def test_manifest_schemas():
    import yaml

    from core.status import ARTIFACT_STATUSES, CLONE_STATUSES

    sources = yaml.safe_load((ROOT / "manifests" / "source_repositories.yaml").read_text(encoding="utf-8"))
    names = []
    for repo in sources["repositories"]:
        assert repo["clone_status"] in CLONE_STATUSES
        assert repo["url"].startswith("https://")
        if repo["clone_status"] == "CLONED":
            assert len(repo["commit_sha"]) == 40
        else:
            assert repo["failure_reason"]
        names.append(repo["name"])
    assert "cupid" in names and "libero" in names and "hinflow" in names
    artifacts = yaml.safe_load((ROOT / "manifests" / "artifacts.yaml").read_text(encoding="utf-8"))
    assert artifacts["assets"]
    for asset in artifacts["assets"]:
        assert asset["status"] in ARTIFACT_STATUSES
        assert asset["url"].startswith("https://")
