"""Shared RoboMimic and LIBERO dataset, schema, environment, and evaluation hooks.

A missing file is BLOCKED. A PH file is not reported as an MH task. Low-dim
inspection is interface-only and is not a formal image Diffusion Policy result.
"""

from __future__ import annotations

import json
from pathlib import Path


def _blocked(reason: str, **extra) -> dict:
    payload = {
        "status": "BLOCKED",
        "reason": reason,
        "paper_result": False,
        "formal_image_dp_result": False,
    }
    payload.update(extra)
    return payload


def variant_from_path(path: Path) -> str | None:
    parts = {part.lower() for part in path.parts}
    name = path.name.lower()
    if "mh" in parts or "mh" in name.split(".")[0].split("_"):
        return "mh"
    if "ph" in parts or name.startswith("ph"):
        return "ph"
    return None


def detect_variant_mismatch(expected: str, observed: str | None) -> str | None:
    if not expected or observed is None:
        return None
    if expected.lower() in {"constructed", "hinflow_original"}:
        return None
    if expected.lower() != observed.lower():
        return (
            f"variant mismatch: benchmark expects {expected}, file reports {observed}. "
            "lift_image.yaml dataset_type ph is not Lift MH."
        )
    return None


def validate_schema(spec, summary: dict) -> dict:
    """Validate an in-memory or on-disk episode summary."""
    if getattr(spec, "observation_mode", "image") == "lowdim":
        return {
            "status": "INTERFACE_ONLY",
            "reason": "low-dim is interface-only and is not a formal image-DP result",
            "formal_image_dp_result": False,
            "paper_result": False,
        }
    demos = list(summary.get("demos") or [])
    if not demos:
        return _blocked("schema has no episodes")
    action_dims = set()
    problems = []
    for demo in demos:
        actions = demo.get("actions")
        obs = demo.get("obs_shapes") or {}
        if not isinstance(actions, list) or not actions:
            problems.append(f"{demo.get('episode_id')} has no actions")
            continue
        width = len(actions[0]) if not isinstance(actions[0], int) else None
        if width is None:
            # actions stored as a shape tuple (T, D)
            if len(actions) == 2 and all(isinstance(v, int) for v in actions):
                length, width = actions
                action_dims.add(width)
                obs_length = demo.get("length")
                if obs_length is not None and obs_length != length:
                    problems.append(f"{demo.get('episode_id')} obs/action length mismatch")
                continue
            problems.append(f"{demo.get('episode_id')} action shape is unreadable")
            continue
        action_dims.add(len(actions[0]))
        if any(len(step) != len(actions[0]) for step in actions):
            problems.append(f"{demo.get('episode_id')} action dimension changes")
        length = demo.get("length", len(actions))
        if length != len(actions):
            problems.append(f"{demo.get('episode_id')} breaks the episode boundary")
        if not obs:
            problems.append(f"{demo.get('episode_id')} has no observation shapes")
    if len(action_dims) > 1:
        problems.append(f"mixed action dims {sorted(action_dims)}")
    expected_dim = getattr(spec, "expected_action_dim", None)
    if expected_dim is not None and action_dims and expected_dim not in action_dims:
        problems.append(f"action dim {action_dims} != expected {expected_dim}")
    observed = summary.get("variant")
    mismatch = detect_variant_mismatch(getattr(spec, "dataset_variant", ""), observed)
    if mismatch:
        problems.append(mismatch)
    if problems:
        return _blocked("; ".join(problems), variant=observed, action_dims=sorted(action_dims))
    return {
        "status": "SCHEMA_OK",
        "paper_result": False,
        "formal_image_dp_result": False,
        "episodes": len(demos),
        "action_dims": sorted(action_dims),
        "variant": observed,
        "obs_shapes": demos[0].get("obs_shapes"),
        "reason": "schema inspected; no policy was evaluated",
    }


def episode_bounds(demos: list[dict]) -> list[tuple[int, int]]:
    bounds = []
    cursor = 0
    for demo in demos:
        length = demo.get("length")
        if length is None:
            actions = demo.get("actions") or []
            if len(actions) == 2 and all(isinstance(v, int) for v in actions):
                length = actions[0]
            else:
                length = len(actions)
        start = cursor
        cursor += int(length)
        bounds.append((start, cursor))
    return bounds


def _summary_from_hdf5(path: Path) -> dict:
    try:
        import h5py
    except Exception as exc:  # noqa: BLE001
        return _blocked(f"HDF5 is present but h5py is unavailable: {type(exc).__name__}: {exc}")
    demos = []
    variant = variant_from_path(path)
    with h5py.File(path, "r") as handle:
        if "data" not in handle:
            return _blocked("HDF5 has no data group", path=str(path))
        data = handle["data"]
        attrs = {key: data.attrs[key] for key in data.attrs}
        env_args = attrs.get("env_args")
        if isinstance(env_args, bytes):
            env_args = env_args.decode("utf-8", errors="replace")
        if isinstance(env_args, str):
            try:
                parsed = json.loads(env_args)
            except json.JSONDecodeError:
                parsed = {"raw": env_args}
            env_name = str(parsed.get("env_name", ""))
            if "MH" in env_name or env_name.endswith("MH"):
                variant = "mh"
            elif "PH" in env_name:
                variant = "ph"
        for key in data.keys():
            group = data[key]
            actions = group["actions"] if "actions" in group else None
            if actions is None:
                demos.append({"episode_id": key, "actions": [], "obs_shapes": {}})
                continue
            shape = tuple(int(v) for v in actions.shape)
            obs_shapes = {}
            if "obs" in group:
                for obs_key in group["obs"].keys():
                    obs_shapes[obs_key] = tuple(int(v) for v in group["obs"][obs_key].shape)
            demos.append(
                {
                    "episode_id": key,
                    "actions": [shape[0], shape[1]] if len(shape) == 2 else [],
                    "length": shape[0] if shape else 0,
                    "obs_shapes": obs_shapes,
                }
            )
    return {"status": "READ", "path": str(path), "variant": variant, "demos": demos, "format": "hdf5"}


def _summary_from_zarr(path: Path) -> dict:
    try:
        import zarr
    except Exception as exc:  # noqa: BLE001
        return _blocked(f"Zarr is present but zarr is unavailable: {type(exc).__name__}: {exc}")
    root = zarr.open(str(path), mode="r")
    variant = variant_from_path(path)
    # Episode ends are required. A flat array without ends is not a success.
    ends = None
    for key in ("episode_ends", "meta/episode_ends"):
        if key in root:
            ends = [int(v) for v in root[key][:]]
            break
    if not ends:
        return _blocked("zarr has no episode_ends; refusing to treat the array as demonstrations", path=str(path))
    demos = []
    previous = 0
    for index, end in enumerate(ends):
        demos.append(
            {
                "episode_id": f"episode_{index}",
                "actions": [end - previous, 0],
                "length": end - previous,
                "obs_shapes": {},
            }
        )
        previous = end
    return {"status": "READ", "path": str(path), "variant": variant, "demos": demos, "format": "zarr"}


def load_dataset(spec, path: str | Path | None = None) -> dict:
    if getattr(spec, "observation_mode", "image") == "lowdim":
        return {
            "status": "INTERFACE_ONLY",
            "reason": "low-dim loader is interface-only and is not a formal image-DP result",
            "formal_image_dp_result": False,
            "paper_result": False,
            "path": None if path is None else str(path),
        }
    if path is None or str(path) == "":
        return _blocked(f"{spec.name} dataset path is empty")
    dataset = Path(path)
    if not dataset.exists():
        return _blocked(f"dataset file absent: {dataset}")
    if dataset.is_dir() or dataset.suffix == ".zarr" or dataset.name.endswith(".zarr"):
        summary = _summary_from_zarr(dataset)
    elif dataset.suffix in {".hdf5", ".h5"}:
        summary = _summary_from_hdf5(dataset)
    else:
        return _blocked(f"unsupported dataset suffix: {dataset.suffix}")
    if summary.get("status") == "BLOCKED":
        summary["formal_image_dp_result"] = False
        summary["paper_result"] = False
        return summary
    checked = validate_schema(spec, summary)
    checked["path"] = str(dataset)
    checked["format"] = summary.get("format")
    checked["bounds"] = episode_bounds(summary.get("demos") or [])
    return checked


def make_environment(spec, path: str | Path | None = None) -> dict:
    if getattr(spec, "observation_mode", "image") != "image":
        return {
            "status": "INTERFACE_ONLY",
            "reason": "low-dim environment is not a formal image-DP result",
            "formal_image_dp_result": False,
            "paper_result": False,
        }
    loaded = load_dataset(spec, path)
    if loaded.get("status") != "SCHEMA_OK":
        return loaded
    family = getattr(spec, "family", "")
    module_name = "libero" if family == "libero" else "robosuite"
    try:
        __import__(module_name)
    except Exception as exc:  # noqa: BLE001
        return _blocked(
            f"dataset schema is readable but {module_name} is not importable: {type(exc).__name__}: {exc}",
            path=loaded.get("path"),
        )
    return _blocked(
        f"{module_name} imported. make_environment does not roll out and does not report success.",
        path=loaded.get("path"),
        simulator_import=module_name,
    )


def evaluate_policy(spec, policy, env=None, *, n_episodes: int = 0) -> dict:
    if env is None or policy is None:
        return _blocked("evaluate_policy requires a real environment and a policy. No success rate was invented.")
    if n_episodes < 1:
        return _blocked("evaluate_policy requires a positive episode count")
    successes = []
    for index in range(n_episodes):
        outcome = env.rollout(policy, episode_index=index)
        if "success" not in outcome:
            return _blocked("environment rollout did not return a success flag")
        successes.append(bool(outcome["success"]))
    return {
        "status": "EVALUATED",
        "paper_result": False,
        "formal_image_dp_result": getattr(spec, "observation_mode", "") == "image",
        "successes": successes,
        "n_episodes": n_episodes,
        "reason": "success flags come from the injected environment rollout",
    }


class BenchmarkBase:
    name = ""
    family = ""
    task = ""
    dataset_variant = ""
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = ""
    image_keys_status = "UNVERIFIED"
    expected_action_dim = None
    expected_dataset_path = ""

    def schema(self):
        return {
            "image_keys_status": self.image_keys_status,
            "action_convention": self.action_convention,
            "forgetting_applicable": self.forgetting_applicable,
            "formal_result_without_hdf5_or_zarr": False,
            "dataset_variant": self.dataset_variant,
        }

    def load_dataset(self, path=None):
        return load_dataset(self, path if path is not None else self.expected_dataset_path)

    def validate_schema(self, summary):
        return validate_schema(self, summary)

    def make_environment(self, path=None):
        return make_environment(self, path if path is not None else self.expected_dataset_path)

    def evaluate_policy(self, policy, env=None, n_episodes: int = 0):
        return evaluate_policy(self, policy, env, n_episodes=n_episodes)


BENCHMARK = None
