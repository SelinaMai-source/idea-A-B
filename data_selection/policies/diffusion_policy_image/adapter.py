"""Image CNN Diffusion Policy probe. Never substitutes another backbone."""

from __future__ import annotations

import platform
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "policies" / "diffusion_policy_image" / "official"
FAMILY = "image_cnn_conditional_unet_diffusion_policy"
FORBIDDEN = (
    "octo",
    "bc_transformer",
    "hinflow_transformer",
    "mlp_only",
    "large_vla",
)

WORKSPACE_CONFIG = "diffusion_policy/config/train_diffusion_unet_image_workspace.yaml"
TASK_CONFIG = "diffusion_policy/config/task/lift_image.yaml"


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


def read_official_horizons() -> dict:
    workspace = OFFICIAL / WORKSPACE_CONFIG
    task = OFFICIAL / TASK_CONFIG
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
    image_shape = None
    dataset_type = None
    if task.is_file():
        task_payload = yaml.safe_load(task.read_text(encoding="utf-8")) or {}
        image_shape = _rgb_shape(task_payload)
        dataset_type = task_payload.get("dataset_type")
    return {
        "status": "READ_FROM_OFFICIAL_CONFIG",
        "workspace_file": WORKSPACE_CONFIG,
        "task_file": TASK_CONFIG if task.is_file() else "",
        "obs_horizon": payload.get("n_obs_steps"),
        "prediction_horizon": payload.get("horizon"),
        "action_horizon": payload.get("n_action_steps"),
        "image_resolution": image_shape,
        "policy_target": policy.get("_target_"),
        "task_yaml_dataset_type": dataset_type,
        "note": "lift_image.yaml dataset_type is the literal in that file. MH experiment directories are separate and were not inferred from this field.",
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
    }
