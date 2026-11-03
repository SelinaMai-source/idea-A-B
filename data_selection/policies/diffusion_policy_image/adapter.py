"""Thin image CNN Diffusion Policy integration around the official checkout.

The diffusion update itself stays in ``DiffusionUnetImagePolicy``. This module
loads the official workspace yaml, checks the dataset, and either calls
``TrainDiffusionUnetImageWorkspace`` or fails closed.
"""

from __future__ import annotations

import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "policies" / "diffusion_policy_image" / "official"
FAMILY = "image_cnn_conditional_unet_diffusion_policy"
POLICY_TARGET = "diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy"
WORKSPACE_TARGET = "diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace"
FORBIDDEN = (
    "octo",
    "bc_transformer",
    "hinflow_transformer",
    "mlp_only",
    "large_vla",
)
WORKSPACE_CONFIG = "diffusion_policy/config/train_diffusion_unet_image_workspace.yaml"
TASK_CONFIG = "diffusion_policy/config/task/lift_image.yaml"
TASK_ABS_CONFIG = "diffusion_policy/config/task/lift_image_abs.yaml"

CHECKPOINT_ROLES = (
    "INITIAL_CHECKPOINT",
    "PRETRAINED_EVALUATION_CHECKPOINT",
    "RETRAINED_CHECKPOINT",
)


def _cuda() -> tuple[bool, str]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - torch is optional on the Mac
        return False, f"{type(exc).__name__}: {exc}"
    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"
    return available, ""


def _checkpoint() -> str | None:
    folder = ROOT / "policies" / "diffusion_policy_image" / "checkpoints"
    if not folder.is_dir():
        return None
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix in {".ckpt", ".pt", ".pth"}:
            return str(path)
    return None


def _rgb_shape(task_payload: dict):
    obs = ((task_payload.get("shape_meta") or {}).get("obs") or {})
    for value in obs.values():
        if isinstance(value, dict) and value.get("type") == "rgb":
            return value.get("shape")
    return None


def _action_shape(task_payload: dict):
    return ((task_payload.get("shape_meta") or {}).get("action") or {}).get("shape")


def read_official_horizons() -> dict:
    workspace = OFFICIAL / WORKSPACE_CONFIG
    task = OFFICIAL / TASK_CONFIG
    task_abs = OFFICIAL / TASK_ABS_CONFIG
    if not workspace.is_file():
        return {
            "status": "UNVERIFIED",
            "workspace_file": "",
            "task_file": "",
            "obs_horizon": None,
            "prediction_horizon": None,
            "action_horizon": None,
            "image_resolution": None,
            "policy_target": None,
            "task_yaml_dataset_type": None,
        }
    payload = yaml.safe_load(workspace.read_text(encoding="utf-8")) or {}
    policy = payload.get("policy") or {}
    defaults = payload.get("defaults") or []
    default_task = None
    for item in defaults:
        if isinstance(item, dict) and "task" in item:
            default_task = item["task"]
    image_shape = None
    dataset_type = None
    action_shape = None
    abs_dataset_type = None
    abs_action_shape = None
    if task.is_file():
        task_payload = yaml.safe_load(task.read_text(encoding="utf-8")) or {}
        image_shape = _rgb_shape(task_payload)
        dataset_type = task_payload.get("dataset_type")
        action_shape = _action_shape(task_payload)
    if task_abs.is_file():
        abs_payload = yaml.safe_load(task_abs.read_text(encoding="utf-8")) or {}
        abs_dataset_type = abs_payload.get("dataset_type")
        abs_action_shape = _action_shape(abs_payload)
    return {
        "status": "READ_FROM_OFFICIAL_CONFIG",
        "workspace_file": WORKSPACE_CONFIG,
        "task_file": TASK_CONFIG if task.is_file() else "",
        "task_abs_file": TASK_ABS_CONFIG if task_abs.is_file() else "",
        "workspace_default_task": default_task,
        "obs_horizon": payload.get("n_obs_steps"),
        "prediction_horizon": payload.get("horizon"),
        "action_horizon": payload.get("n_action_steps"),
        "image_resolution": image_shape,
        "policy_target": policy.get("_target_"),
        "workspace_target": payload.get("_target_"),
        "optimizer_target": (payload.get("optimizer") or {}).get("_target_"),
        "scheduler_name": (payload.get("training") or {}).get("lr_scheduler"),
        "training_device_in_yaml": (payload.get("training") or {}).get("device"),
        "task_yaml_dataset_type": dataset_type,
        "task_yaml_action_shape": action_shape,
        "task_abs_dataset_type": abs_dataset_type,
        "task_abs_action_shape": abs_action_shape,
        "mh_inferred_from_task_yaml": False,
        "note": (
            "Horizons are the values stored in train_diffusion_unet_image_workspace.yaml. "
            "lift_image.yaml and lift_image_abs.yaml both set dataset_type ph. "
            "That field is not Lift MH. MH requires an mh dataset path."
        ),
    }


def probe() -> dict:
    cuda, torch_error = _cuda()
    checkpoint = _checkpoint()
    horizons = read_official_horizons()
    if not cuda:
        status = "BLOCKED_NO_CUDA"
    elif checkpoint is None:
        status = "BLOCKED_NO_CHECKPOINT"
    else:
        status = "CHECKPOINT_PRESENT_TRAINING_NOT_RUN"
    return {
        "status": status,
        "family": FAMILY,
        "substituted_backbone": None,
        "forbidden_substitutes": list(FORBIDDEN),
        "cuda": cuda,
        "torch_import_error": torch_error,
        "checkpoint": checkpoint,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "horizons": horizons,
        "lowdim_is_formal_image_result": False,
        "official_checkout_present": OFFICIAL.is_dir(),
        "policy_target": POLICY_TARGET,
        "workspace_target": WORKSPACE_TARGET,
    }


def classify_checkpoint(path: str, *, task: str, output_dir: str | None = None, resume: bool = False) -> dict:
    checkpoint = Path(path)
    name = checkpoint.name
    lowered = str(checkpoint).lower()
    task_token = (task or "").lower()
    task_match = bool(task_token) and task_token in lowered
    if "test_mean_score" in name or name.startswith("epoch="):
        role = "PRETRAINED_EVALUATION_CHECKPOINT"
        usable = False
        reason = "a finished evaluation checkpoint is not a from-scratch INITIAL_CHECKPOINT"
    elif resume and output_dir and checkpoint.is_file() and output_dir in str(checkpoint):
        role = "RETRAINED_CHECKPOINT"
        usable = True
        reason = "resume uses a checkpoint stored under this run's output directory"
    elif name == "latest.ckpt":
        role = "UNSPECIFIED_LATEST"
        usable = False
        reason = "latest.ckpt is not treated as the initial state of a from-scratch run"
    else:
        role = "UNSPECIFIED"
        usable = False
        reason = "checkpoint role is not INITIAL_CHECKPOINT"
    if task_token and not task_match:
        usable = False
        reason = f"checkpoint path does not contain task {task}. A Lift checkpoint is not used for another task."
    return {
        "path": str(checkpoint),
        "role": role,
        "usable_as_initial_state": usable,
        "task_match": task_match,
        "reason": reason,
        "known_roles": list(CHECKPOINT_ROLES),
    }


def optimization_step(policy, batch, optimizer, scheduler=None):
    """One official loss backward. The loss function is the policy's, not a rewrite."""
    loss = policy.compute_loss(batch)
    loss.backward()
    optimizer.step()
    zero_grad = getattr(optimizer, "zero_grad", None)
    if zero_grad is not None:
        try:
            zero_grad(set_to_none=True)
        except TypeError:
            zero_grad()
    if scheduler is not None:
        scheduler.step()
    return loss


def save_training_state(path: Path, payload, saver) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saver(payload, path)


def load_training_state(path: Path, loader):
    if not path.is_file():
        raise FileNotFoundError(path)
    return loader(path)


def rollout_successes(env, policy, n_episodes: int, seed: int) -> list[bool]:
    successes = []
    for index in range(n_episodes):
        outcome = env.rollout(policy, episode_index=index, seed=seed + index)
        if "success" not in outcome:
            raise RuntimeError("rollout did not return success")
        successes.append(bool(outcome["success"]))
    return successes


def _blocked(status: str, reason: str, **extra) -> dict:
    payload = {
        "status": status,
        "reason": reason,
        "paper_result": False,
        "family": FAMILY,
        "substituted_backbone": None,
        "training_optimizer_steps": None,
        "will_train": False,
    }
    payload.update(extra)
    return payload


def launch_official_workspace(request: dict) -> dict:
    """Import the author workspace and call ``run``. This is not a reimplementation."""
    import sys

    official = str(OFFICIAL)
    if official not in sys.path:
        sys.path.insert(0, official)
    from diffusion_policy.workspace.train_diffusion_unet_image_workspace import (  # type: ignore
        TrainDiffusionUnetImageWorkspace,
    )
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(OFFICIAL / WORKSPACE_CONFIG))
    cfg.training.seed = int(request["seed"])
    cfg.training.device = request["device"]
    if request.get("dataset_path"):
        cfg.task.dataset.dataset_path = request["dataset_path"]
        cfg.task.dataset_type = request.get("expected_variant", "mh")
    workspace = TrainDiffusionUnetImageWorkspace(cfg, output_dir=request["output_dir"])
    if request.get("resume") and request.get("checkpoint"):
        workspace.load_checkpoint(path=request["checkpoint"])
    workspace.run()
    return {
        "status": "COMPLETED_UNVERIFIED",
        "paper_result": False,
        "reason": "official TrainDiffusionUnetImageWorkspace.run returned. Metrics still require a separate eval log.",
        "training_optimizer_steps": getattr(workspace, "global_step", None),
        "will_train": True,
    }


def execute_image_dp(request: dict, *, inspector=None, workspace_factory=None, cuda_probe=None) -> dict:
    horizons = read_official_horizons()
    device = str(request.get("device") or "cpu")
    if cuda_probe is None:
        cuda_ok, cuda_detail = _cuda()
    else:
        cuda_ok, cuda_detail = cuda_probe()
    if device.startswith("cuda") and not cuda_ok:
        return _blocked(
            "BLOCKED_NO_CUDA",
            f"requested {device} but CUDA is unavailable ({cuda_detail or 'torch.cuda.is_available() is false'})",
            horizons=horizons,
        )
    benchmark = request.get("benchmark") or ""
    task = request.get("task") or ""
    if task and benchmark and task.lower() not in benchmark.lower() and task.lower() not in {"lift", "can", "square", "transport"}:
        return _blocked("BLOCKED", f"task {task} does not match benchmark {benchmark}", horizons=horizons)
    if task and benchmark and task.lower() not in benchmark.lower():
        return _blocked("BLOCKED", f"task {task} does not match benchmark {benchmark}", horizons=horizons)
    expected = "mh" if benchmark.endswith("_mh") else request.get("expected_variant")
    dataset_path = request.get("dataset_path") or ""
    if not dataset_path or not Path(dataset_path).exists():
        return _blocked(
            "BLOCKED",
            f"dataset file absent: {dataset_path or '(empty)'}",
            horizons=horizons,
            task_yaml_dataset_type=horizons.get("task_yaml_dataset_type"),
        )
    from benchmarks.common import detect_variant_mismatch, variant_from_path

    path_variant = variant_from_path(Path(dataset_path))
    mismatch = detect_variant_mismatch((expected or "").upper() if expected in {"mh", "ph"} else (expected or ""), path_variant)
    if path_variant and expected and path_variant != expected.lower():
        return _blocked(
            "BLOCKED",
            f"variant mismatch: benchmark expects {expected}, path reports {path_variant}. "
            "lift_image.yaml dataset_type ph is not Lift MH.",
            horizons=horizons,
            task_yaml_dataset_type=horizons.get("task_yaml_dataset_type"),
            reported_as_mh=False,
        )
    _ = mismatch
    if inspector is None:
        from benchmarks.common import load_dataset

        class _Spec:
            name = benchmark or "image_dp"
            observation_mode = "image"
            dataset_variant = (expected or "").upper() if expected in {"mh", "ph"} else (expected or "")
            family = "robomimic" if "robomimic" in benchmark else "libero"

        inspection = load_dataset(_Spec(), dataset_path)
    else:
        inspection = inspector(dataset_path)
    if inspection.get("status") != "SCHEMA_OK":
        return _blocked(
            "BLOCKED",
            inspection.get("reason") or "dataset failed schema validation",
            horizons=horizons,
            inspection_status=inspection.get("status"),
        )
    observed = (inspection.get("variant") or "").lower()
    if expected and observed and observed != expected.lower():
        return _blocked(
            "BLOCKED",
            f"variant mismatch: benchmark expects {expected}, dataset reports {observed}. "
            "ph is not reported as MH.",
            horizons=horizons,
        )
    checkpoint = request.get("checkpoint")
    if checkpoint:
        decision = classify_checkpoint(
            checkpoint,
            task=task,
            output_dir=request.get("output_dir"),
            resume=bool(request.get("resume")),
        )
        if not decision["usable_as_initial_state"]:
            return _blocked(decision["role"] if decision["role"] != "UNSPECIFIED" else "BLOCKED", decision["reason"], checkpoint=decision, horizons=horizons)
    if request.get("dry_run"):
        return _blocked("DRY_RUN", "dry-run does not train", horizons=horizons, will_train=False)
    factory = workspace_factory or launch_official_workspace
    try:
        started = factory(request)
    except Exception as exc:  # noqa: BLE001
        return _blocked(
            "BLOCKED",
            f"official workspace was not started: {type(exc).__name__}: {exc}",
            horizons=horizons,
        )
    started.setdefault("paper_result", False)
    started.setdefault("family", FAMILY)
    started.setdefault("substituted_backbone", None)
    started["horizons"] = horizons
    return started


def real_device_blocker(device: str, *, cuda_ok: bool, cuda_detail: str = "") -> str | None:
    if str(device).startswith("cuda") and not cuda_ok:
        detail = cuda_detail or "torch.cuda.is_available() is false"
        return f"BLOCKED_NO_CUDA: requested {device} but CUDA is unavailable ({detail})"
    return None
