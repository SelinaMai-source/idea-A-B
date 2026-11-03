"""Integrity checks for pins, transfers, schemas, budgets, metrics, and the CLI."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

from benchmarks.common import detect_variant_mismatch, episode_bounds, validate_schema
from benchmarks.robomimic_lift_mh.adapter import BENCHMARK as LIFT
from core.matrix import build_matrix
from core.registry import discover_baselines
from core.selection_budget import take_prefix
from core.trajectories import make_synthetic_trajectory
from core.types import SelectionBudget, SelectionContext
from evaluation.metrics import metric_applicability, preserve_runs
from policies.diffusion_policy_image.adapter import (
    classify_checkpoint,
    execute_image_dp,
    optimization_step,
    probe,
    read_official_horizons,
    real_device_blocker,
)
from tools.fetch_sources import archive_url, verify_record, write_manifest
from tools.http_transfer import download_with_resume

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


class _Proc:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class _Body:
    def __init__(self, status: int, headers: dict, body: bytes):
        self.status = status
        self.headers = headers
        self._body = body
        self._pos = 0

    def read(self, n: int = -1) -> bytes:
        if self._pos >= len(self._body):
            return b""
        if n is None or n < 0:
            n = len(self._body)
        chunk = self._body[self._pos : self._pos + n]
        self._pos += len(chunk)
        return chunk


def _pool():
    return [
        make_synthetic_trajectory("a", 3, score=0.2, collected_step=1),
        make_synthetic_trajectory("b", 3, score=0.9, collected_step=5),
        make_synthetic_trajectory("c", 3, score=0.4, collected_step=3),
        make_synthetic_trajectory("d", 5, score=0.1, collected_step=4),
    ]


def test_pinned_fetch_does_not_replace_a_different_head(tmp_path: Path):
    pin = "a" * 40
    other = "b" * 40
    dest = tmp_path / "repo"
    dest.mkdir()
    (dest / ".git").write_text("gitdir: fake\n", encoding="utf-8")
    (dest / "eval_demonstration_scores.py").write_text("print('author')\n", encoding="utf-8")

    def runner(cmd, timeout):
        joined = " ".join(cmd)
        if "rev-parse" in joined:
            return _Proc(stdout=other + "\n")
        return _Proc(stdout="")

    record = verify_record(
        {
            "name": "cupid",
            "commit_sha": pin,
            "local_path": "repo",
            "url": "https://github.com/agiachris/cupid",
            "clone_status": "CLONED",
        },
        runner,
        root=tmp_path,
    )
    assert record["commit_sha"] == pin
    assert record["observed_head"] == other
    assert record["pin_verified"] is False
    assert "was not updated" in record["failure_reason"]


def test_manifest_pins_stay_forty_characters():
    payload = yaml.safe_load((ROOT / "manifests" / "source_repositories.yaml").read_text(encoding="utf-8"))
    names = {row["name"] for row in payload["repositories"]}
    assert {"cupid", "deminf", "datamil", "demo_score", "strap", "hinflow", "diffusion_policy", "libero"} <= names
    for row in payload["repositories"]:
        if row["clone_status"] == "CLONED":
            assert len(row["commit_sha"]) == 40


def test_write_manifest_refuses_pin_overwrite(tmp_path: Path):
    path = tmp_path / "sources.yaml"
    with pytest.raises(RuntimeError, match="overwrite pin"):
        write_manifest(
            path,
            [{"name": "cupid", "commit_sha": "c" * 40}],
            {"cupid": "a" * 40},
        )


def test_archive_url_uses_the_pinned_commit():
    url = archive_url("https://github.com/agiachris/cupid", "a" * 40)
    assert url == "https://github.com/agiachris/cupid/archive/" + ("a" * 40) + ".tar.gz"


def test_hinflow_submodule_record_is_not_marked_ready():
    text = (ROOT / "manifests" / "source_repositories.yaml").read_text(encoding="utf-8")
    assert "hinflow" in text
    hin = next(
        row
        for row in yaml.safe_load(text)["repositories"]
        if row["name"] == "hinflow"
    )
    assert hin["submodules"]["status"] != "INITIALIZED"


def test_range_206_appends_and_ignored_range_restarts(tmp_path: Path):
    dest = tmp_path / "file.bin"
    partial = tmp_path / "file.bin.partial"
    partial.write_bytes(b"AAAA")

    def opener(url, method="GET", headers=None):
        if method == "HEAD":
            return _Body(200, {"content-length": "8", "content-type": "application/octet-stream"}, b"")
        assert headers["Range"] == "bytes=4-"
        return _Body(206, {"content-range": "bytes 4-7/8", "content-type": "application/octet-stream"}, b"BBBB")

    report = download_with_resume("https://example.invalid/file", dest, opener, expected_size=8, retries=1)
    assert report["status"] == "DOWNLOADED_VERIFIED"
    assert dest.read_bytes() == b"AAAABBBB"
    assert not partial.exists()

    ignored = tmp_path / "ignored.bin"
    ignored_partial = tmp_path / "ignored.bin.partial"
    ignored_partial.write_bytes(b"AAAA")

    def ignore_range(url, method="GET", headers=None):
        if method == "HEAD":
            return _Body(200, {"content-length": "8"}, b"")
        return _Body(200, {"content-type": "application/octet-stream"}, b"AAAABBBB")

    report = download_with_resume(
        "https://example.invalid/ignored",
        ignored,
        ignore_range,
        expected_size=8,
        retries=1,
    )
    assert report["status"] == "DOWNLOADED_VERIFIED"
    assert ignored.read_bytes() == b"AAAABBBB"


def test_bad_content_range_does_not_append(tmp_path: Path):
    dest = tmp_path / "file.bin"
    partial = tmp_path / "file.bin.partial"
    partial.write_bytes(b"AAAA")

    def opener(url, method="GET", headers=None):
        if method == "HEAD":
            return _Body(200, {"content-length": "8"}, b"")
        return _Body(206, {"content-range": "bytes 0-3/8"}, b"CCCC")

    report = download_with_resume("https://example.invalid/bad", dest, opener, expected_size=8, retries=2)
    assert report["status"] == "DOWNLOAD_FAILED"
    assert report["status"] != "SOURCE_VERIFIED_DEFERRED"
    assert partial.read_bytes() == b"AAAA"
    assert not dest.exists()


def test_checksum_mismatch_is_not_verified(tmp_path: Path):
    dest = tmp_path / "file.bin"

    def opener(url, method="GET", headers=None):
        if method == "HEAD":
            return _Body(200, {"content-length": "4"}, b"")
        return _Body(200, {"content-type": "application/octet-stream"}, b"data")

    report = download_with_resume(
        "https://example.invalid/sum",
        dest,
        opener,
        expected_size=4,
        expected_sha256="0" * 64,
        retries=1,
    )
    assert report["status"] == "DOWNLOAD_FAILED"
    assert not dest.exists()


def test_verified_asset_is_not_downloaded_again(tmp_path: Path):
    dest = tmp_path / "done.bin"
    dest.write_bytes(b"data")
    from tools.http_transfer import sha256_file

    digest = sha256_file(dest)
    dest.with_name(dest.name + ".provenance.json").write_text(
        json.dumps({"status": "DOWNLOADED_VERIFIED", "sha256": digest, "url": "https://example.invalid/done"}),
        encoding="utf-8",
    )

    def opener(url, method="GET", headers=None):
        raise AssertionError("verified asset was fetched again")

    report = download_with_resume("https://example.invalid/done", dest, opener, expected_sha256=digest)
    assert report["status"] == "DOWNLOADED_VERIFIED"
    assert report["redownloaded"] is False


def test_archive_members_are_listed_without_becoming_a_dataset(tmp_path: Path):
    archive = tmp_path / "tiny.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("readme.txt", "not a robot dataset")
    from tools.http_transfer import inspect_archive

    info = inspect_archive(archive)
    assert info["format"] == "zip"
    assert "readme.txt" in info["members_preview"]


def test_macos_image_zip_stays_deferred():
    from tools.download_assets import download_one

    report = download_one("robomimic_image", Path("/tmp/should-not-exist.zip"), allow_large=True, system="Darwin")
    assert report["status"] == "SOURCE_VERIFIED_DEFERRED"
    assert report["status"] != "DOWNLOAD_FAILED"


def test_schema_shapes_bounds_and_ph_mh_mismatch():
    summary = {
        "variant": "ph",
        "demos": [
            {
                "episode_id": "demo_0",
                "actions": [4, 7],
                "length": 4,
                "obs_shapes": {"agentview_image": [4, 84, 84, 3]},
            }
        ],
    }
    checked = validate_schema(LIFT, summary)
    assert checked["status"] == "BLOCKED"
    assert "not Lift MH" in checked["reason"]
    assert detect_variant_mismatch("MH", "ph")
    good = {
        "variant": "mh",
        "demos": [
            {
                "episode_id": "demo_0",
                "actions": [4, 7],
                "length": 4,
                "obs_shapes": {"agentview_image": [4, 84, 84, 3]},
            },
            {
                "episode_id": "demo_1",
                "actions": [2, 7],
                "length": 2,
                "obs_shapes": {"agentview_image": [2, 84, 84, 3]},
            },
        ],
    }
    ok = validate_schema(LIFT, good)
    assert ok["status"] == "SCHEMA_OK"
    assert ok["formal_image_dp_result"] is False
    assert episode_bounds(good["demos"]) == [(0, 4), (4, 6)]
    absent = LIFT.load_dataset(Path("/tmp/data_selection_missing_lift_mh.hdf5"))
    assert absent["status"] == "BLOCKED"
    assert "absent" in absent["reason"]


def test_lowdim_is_not_a_formal_image_result():
    class Low:
        name = "lowdim_probe"
        observation_mode = "lowdim"
        dataset_variant = "mh"

    from benchmarks.common import load_dataset

    report = load_dataset(Low(), "/tmp/whatever.hdf5")
    assert report["formal_image_dp_result"] is False
    assert report["status"] == "INTERFACE_ONLY"


def test_horizons_come_from_the_workspace_yaml_and_ph_is_not_mh():
    horizons = read_official_horizons()
    payload = yaml.safe_load((ROOT / "policies/diffusion_policy_image/official" / horizons["workspace_file"]).read_text())
    assert horizons["obs_horizon"] == payload["n_obs_steps"]
    assert horizons["prediction_horizon"] == payload["horizon"]
    assert horizons["action_horizon"] == payload["n_action_steps"]
    assert horizons["policy_target"].endswith("DiffusionUnetImagePolicy")
    assert horizons["task_yaml_dataset_type"] == "ph"
    assert horizons["mh_inferred_from_task_yaml"] is False


def test_checkpoint_roles_do_not_reuse_a_finished_lift_ckpt_for_can():
    finished = classify_checkpoint(
        "experiments/image/lift_mh/diffusion_policy_cnn/train_0/checkpoints/epoch=0250-test_mean_score=1.000.ckpt",
        task="lift",
    )
    assert finished["role"] == "PRETRAINED_EVALUATION_CHECKPOINT"
    assert finished["usable_as_initial_state"] is False
    other = classify_checkpoint(
        "experiments/image/lift_mh/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt",
        task="can",
    )
    assert other["usable_as_initial_state"] is False
    assert "can" in other["reason"]


def test_fake_optimization_step_calls_backward_and_cuda_test_is_integration():
    class Loss:
        def __init__(self):
            self.backward_called = False

        def backward(self):
            self.backward_called = True

    class Policy:
        def compute_loss(self, batch):
            self.batch = batch
            self.loss = Loss()
            return self.loss

    class Optimizer:
        def __init__(self):
            self.stepped = False

        def step(self):
            self.stepped = True

        def zero_grad(self, set_to_none=False):
            self.cleared = set_to_none

    class Scheduler:
        def __init__(self):
            self.stepped = False

        def step(self):
            self.stepped = True

    policy = Policy()
    optimizer = Optimizer()
    scheduler = Scheduler()
    loss = optimization_step(policy, {"obs": "fake"}, optimizer, scheduler)
    assert loss.backward_called
    assert optimizer.stepped and scheduler.stepped
    assert policy.batch == {"obs": "fake"}


@pytest.mark.integration
def test_real_cuda_policy_forward_is_not_run_on_this_process():
    report = probe()
    if not report["cuda"]:
        pytest.skip("requires NVIDIA CUDA for DiffusionUnetImagePolicy forward and backward")
    pytest.fail("CUDA is available; this integration test does not claim a real forward/backward pass")


def test_published_methods_block_or_consume_author_scores():
    selectors = {item.name: item for item in discover_baselines()}
    pool = _pool()
    budget = SelectionBudget(max_episodes=1)
    context = SelectionContext(seed=0)
    for name in ("cupid", "deminf", "datamil", "demo_score", "strap"):
        result = selectors[name].select(pool, budget, context)
        assert result.status == "BLOCKED"
        assert result.selected_ids == []
        assert result.extra_cost["heuristic_fallback"] is False
        assert result.extra_cost["official_stack_invoked"] is False
    cupid = selectors["cupid"]
    scored = cupid.select(
        pool,
        budget,
        context,
        author_runner=lambda trajectories, budget, context: {"a": 0.1, "b": 5.0, "c": 0.2, "d": 0.3},
    )
    assert scored.extra_cost["official_stack_invoked"] is True
    assert scored.selected_ids == ["b"]
    misuse = selectors["demo_score"].select(
        pool,
        budget,
        SelectionContext(seed=0, dataset_metadata={"failed_actions_as_imitation": True}),
    )
    assert "imitation labels" in misuse.reason
    leaked = selectors["demo_score"].select(
        pool,
        budget,
        SelectionContext(
            seed=0,
            dataset_metadata={"classifier_train_ids": ["a"], "eval_ids": ["a", "b"]},
        ),
    )
    assert "leak" in leaked.reason
    assert "octo" in selectors["datamil"].select(pool, budget, context).reason.lower()
    assert selectors["strap"].selects == "subtrajectory"


def test_whole_episodes_fifo_reservoir_and_subtrajectory_cut():
    pool = _pool()
    kept, truncated, stopped = take_prefix(pool, SelectionBudget(max_transitions=4), allow_partial=False)
    assert truncated is False
    assert stopped
    assert all(item.n_transitions in {3, 5} for item in kept)
    cut, was_cut, _reason = take_prefix(pool[:1], SelectionBudget(max_transitions=2), allow_partial=True)
    assert was_cut is True
    assert cut[0].n_transitions == 2
    selectors = {item.name: item for item in discover_baselines()}
    fifo = selectors["fifo"].select(pool, SelectionBudget(max_episodes=2), SelectionContext(seed=0))
    assert fifo.extra_cost["retained_end"] == "newest"
    assert fifo.selected_ids == ["b", "d"]
    text = (ROOT / "baselines/reservoir/adapter.py").read_text(encoding="utf-8")
    assert "random.shuffle" not in text
    assert "Algorithm R" in text


def test_metric_applicability_auc_and_forgetting():
    assert metric_applicability("P1", "success_auc", "robomimic_lift_mh") == "N/A"
    assert metric_applicability("P2", "success_auc", "robomimic_lift_mh", online_protocol_real=False) == "N/A"
    assert metric_applicability("P2", "success_auc", "robomimic_lift_mh", online_protocol_real=True) == "NOT_RUN"
    assert metric_applicability("P3", "forgetting", "robomimic_lift_mh") == "N/A"
    assert metric_applicability("P3", "backward_transfer", "robomimic_lift_mh") == "N/A"
    assert metric_applicability("P3", "average_success", "libero_continual") == "NOT_RUN"
    assert metric_applicability("P0", "forward_transfer", "libero_continual") == "NOT_DEFINED"
    kept = preserve_runs([{"seed": 0, "status": "FAILED"}], [{"seed": 1, "status": "NOT_RUN"}])
    assert [row["status"] for row in kept] == ["FAILED", "NOT_RUN"]


def test_matrix_does_not_treat_missing_cuda_as_incompatible_code():
    matrix = build_matrix()
    cupid = next(row for row in matrix["rows"] if row["method"] == "cupid")
    assert cupid["image_dp_compatibility"] != "BLOCKED_NO_CUDA"
    assert cupid["RUNTIME_READINESS"] == "BLOCKED_NO_CUDA"
    assert cupid["EXPERIMENT_STATUS"] == "NOT_RUN"
    hin = next(row for row in matrix["rows"] if row["method"] == "hinflow")
    assert hin["image_dp_compatibility"] == "NOT_APPLICABLE"


def test_cli_cuda_flag_real_mode_and_resume(tmp_path: Path):
    assert real_device_blocker("cuda", cuda_ok=True) is None
    assert real_device_blocker("cuda", cuda_ok=False).startswith("BLOCKED_NO_CUDA")
    out = tmp_path / "real"
    completed = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_unified_baselines.py"),
            "--method",
            "all_data",
            "--benchmark",
            "robomimic_lift_mh",
            "--task",
            "lift",
            "--seed",
            "0",
            "--config",
            str(ROOT / "configs/experiments/robomimic_lift_mh_all_data_seed0.yaml"),
            "--output-dir",
            str(out),
            "--device",
            "cuda",
            "--retention-ratio",
            "1.0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2, completed.stderr
    payload = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED_NO_CUDA"
    assert payload["paper_result"] is False
    assert payload["will_train"] is False
    assert payload["training_optimizer_steps"] is None
    assert payload["device"] == "cuda"
    assert payload["seed"] == 0
    assert payload["retention_ratio"] == 1.0
    assert "SYNTHETIC_INTERFACE_SMOKE" not in json.dumps(payload)
    resume = tmp_path / "resume"
    resumed = subprocess.run(
        [
            PYTHON,
            str(ROOT / "scripts" / "run_unified_baselines.py"),
            "--method",
            "all_data",
            "--benchmark",
            "robomimic_lift_mh",
            "--task",
            "lift",
            "--seed",
            "0",
            "--output-dir",
            str(resume),
            "--device",
            "cpu",
            "--resume",
            "--retention-ratio",
            "1.0",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 2
    resume_payload = json.loads((resume / "result.json").read_text(encoding="utf-8"))
    assert resume_payload["status"] == "BLOCKED"
    assert "resume" in resume_payload["reason"]


def test_missing_dataset_does_not_call_the_workspace(tmp_path: Path):
    called = {}

    def factory(request):
        called["ran"] = request
        return {"status": "COMPLETED_VERIFIED", "paper_result": False}

    missing = execute_image_dp(
        {
            "device": "cuda",
            "benchmark": "robomimic_lift_mh",
            "task": "lift",
            "dataset_path": str(tmp_path / "missing.hdf5"),
            "seed": 0,
            "output_dir": str(tmp_path),
        },
        workspace_factory=factory,
        cuda_probe=lambda: (True, ""),
    )
    assert called == {}
    assert missing["status"] == "BLOCKED"
    assert missing["paper_result"] is False
    ph = tmp_path / "lift" / "ph" / "image.hdf5"
    ph.parent.mkdir(parents=True)
    ph.write_bytes(b"ph-bytes")
    mismatched = execute_image_dp(
        {
            "device": "cuda",
            "benchmark": "robomimic_lift_mh",
            "task": "lift",
            "dataset_path": str(ph),
            "seed": 0,
            "output_dir": str(tmp_path),
        },
        workspace_factory=factory,
        cuda_probe=lambda: (True, ""),
    )
    assert called == {}
    assert "ph" in mismatched["reason"]
    assert mismatched.get("reported_as_mh") is False


def test_darwin_scripts_do_not_train():
    env = subprocess.run(["bash", str(ROOT / "scripts" / "setup_environment.sh")], cwd=ROOT, check=False, capture_output=True, text=True)
    assert env.returncode == 2
    assert "Darwin" in env.stderr
    gpu = subprocess.run(["bash", str(ROOT / "scripts" / "setup_gpu.sh")], cwd=ROOT, check=False, capture_output=True, text=True)
    assert gpu.returncode == 2
    lift = subprocess.run(["bash", str(ROOT / "scripts" / "run_lift_mh_all_data.sh")], cwd=ROOT, check=False, capture_output=True, text=True)
    assert lift.returncode == 2


def test_failed_result_is_kept_by_the_summary(tmp_path: Path):
    run = tmp_path / "results" / "job"
    run.mkdir(parents=True)
    (run / "result.json").write_text(json.dumps({"status": "FAILED", "paper_result": False, "seed": 0}) + "\n", encoding="utf-8")
    completed = subprocess.run(
        [PYTHON, str(ROOT / "scripts" / "summarize_results.py"), "--results-dir", str(tmp_path / "results"), "--output-dir", str(tmp_path / "out")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["failed_rows"] == 1
    assert summary["paper_results"] == 0
