"""Shared CLI.

``--dry-run`` writes a plan and does not train.
``--synthetic`` stays on the synthetic interface.
A real invocation calls the image Diffusion Policy path or fails closed.
``--device cuda`` is blocked only when CUDA is actually unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.registry import discover_baselines, discover_benchmarks  # noqa: E402
from core.seeds import set_seed  # noqa: E402
from core.status import BLOCKED_NO_CUDA, SYNTHETIC_INTERFACE_SMOKE  # noqa: E402
from core.trajectories import make_synthetic_trajectory  # noqa: E402
from core.types import SelectionBudget, SelectionContext  # noqa: E402
from policies.diffusion_policy_image.adapter import (  # noqa: E402
    probe,
    real_device_blocker,
)


def build_parser(protocol: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"data_selection v0 runner ({protocol})")
    parser.add_argument("--method")
    parser.add_argument("--benchmark")
    parser.add_argument("--task")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config")
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retention-ratio", type=float)
    parser.add_argument("--online-steps", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run the CPU synthetic interface only. This is not a paper result.",
    )
    parser.set_defaults(protocol=protocol)
    return parser


def _selector(name: str):
    for spec in discover_baselines():
        if spec.name == name:
            return spec
    raise SystemExit(f"unknown or unregistered method: {name}")


def _benchmark(name: str | None):
    if name is None:
        return None
    for spec in discover_benchmarks():
        if spec.name == name:
            return spec
    raise SystemExit(f"unknown or unregistered benchmark: {name}")


def _load_config(args: argparse.Namespace) -> dict:
    if not args.config:
        return {}
    path = Path(args.config)
    if not path.is_file():
        raise SystemExit(f"config file is absent: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"config must be a mapping: {path}")
    return payload


def _dataset_path(args: argparse.Namespace, bench, config: dict) -> str:
    if config.get("dataset_path"):
        return str(config["dataset_path"])
    env_key = config.get("dataset_env")
    if env_key and os.environ.get(str(env_key)):
        return os.environ[str(env_key)]
    if bench is not None and bench.name == "robomimic_lift_mh" and os.environ.get("ROBOMIMIC_LIFT_MH_DATASET"):
        return os.environ["ROBOMIMIC_LIFT_MH_DATASET"]
    return str(getattr(bench, "expected_dataset_path", "") or "")


def _plan(args: argparse.Namespace, status: str, reason: str) -> dict:
    resume_from = None
    if args.resume and args.output_dir:
        state_path = Path(args.output_dir) / "state.json"
        if state_path.is_file():
            resume_from = json.loads(state_path.read_text(encoding="utf-8")).get("completed_phases")
    return {
        "dry_run": True,
        "protocol": args.protocol,
        "method": args.method,
        "benchmark": args.benchmark,
        "task": args.task,
        "seed": args.seed,
        "config": args.config,
        "device": args.device,
        "checkpoint": args.checkpoint,
        "retention_ratio": args.retention_ratio,
        "online_steps": args.online_steps,
        "resume": bool(args.resume),
        "resume_from": resume_from,
        "planned_status": status,
        "reason": reason,
        "will_train": False,
        "will_download": False,
        "paper_result": False,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _budget(args: argparse.Namespace) -> SelectionBudget:
    max_episodes = None
    if args.retention_ratio == 1.0:
        max_episodes = None
    return SelectionBudget(
        max_episodes=max_episodes,
        max_transitions=args.online_steps if args.protocol != "online_closed_loop" else None,
        retention_ratio=args.retention_ratio,
    )


def _common(args: argparse.Namespace, payload: dict) -> dict:
    payload.update(
        {
            "protocol": args.protocol,
            "method": args.method,
            "benchmark": args.benchmark,
            "task": args.task,
            "seed": args.seed,
            "config": args.config,
            "device": args.device,
            "checkpoint": args.checkpoint,
            "retention_ratio": args.retention_ratio,
            "online_steps": args.online_steps,
            "resume": bool(args.resume),
            "paper_result": False,
            "will_download": False,
        }
    )
    return payload


def execute(args: argparse.Namespace) -> int:
    if args.method is None or args.benchmark is None or args.output_dir is None:
        raise SystemExit("--method, --benchmark, and --output-dir are required")
    if args.seed < 0:
        raise SystemExit("seed must be >= 0")
    spec = _selector(args.method) if args.method != "hinflow" else None
    bench = _benchmark(args.benchmark)
    policy = probe()
    if args.dry_run:
        if args.method == "hinflow":
            status = "NOT_APPLICABLE"
            reason = "HinFlow is not an Image-DP selector. Use scripts/run_reference via references/hinflow."
        elif spec.kind == "published_method":
            blocked = spec.select(
                [make_synthetic_trajectory("probe", 2)],
                SelectionBudget(max_episodes=1),
                SelectionContext(seed=args.seed),
            )
            status = blocked.status
            reason = blocked.reason
        else:
            status = "SYNTHETIC_INTERFACE_ONLY"
            reason = "Dry run does not train. A later --synthetic invocation only exercises the CPU selector."
        cuda_ok = bool(policy.get("cuda"))
        device_note = real_device_blocker(args.device, cuda_ok=cuda_ok, cuda_detail=policy.get("torch_import_error") or "")
        if device_note:
            reason = reason + " " + device_note
        payload = _plan(args, status, reason)
        payload["policy_probe"] = policy["status"]
        payload["benchmark_forgetting_applicable"] = bench.forgetting_applicable
        out = Path(args.output_dir)
        _write(out / "plan.json", payload)
        state = {"completed_phases": ["plan"], "paper_result": False}
        _write(out / "state.json", state)
        return 0

    out = Path(args.output_dir)
    if args.synthetic:
        return _synthetic(args, spec, out)
    return _real(args, spec, bench, policy, out)


def _synthetic(args: argparse.Namespace, spec, out: Path) -> int:
    if spec is None or spec.kind != "control":
        if spec is None:
            _write(
                out / "result.json",
                _common(
                    args,
                    {
                        "status": "NOT_APPLICABLE",
                        "reason": "HinFlow is not a unified selector.",
                    },
                ),
            )
            return 2
        blocked = spec.select(
            [make_synthetic_trajectory("probe", 2)],
            SelectionBudget(max_episodes=1),
            SelectionContext(seed=args.seed),
        )
        _write(
            out / "result.json",
            _common(
                args,
                {
                    "status": blocked.status,
                    "reason": blocked.reason,
                    "selected_ids": blocked.selected_ids,
                },
            ),
        )
        return 2
    if args.protocol == "online_closed_loop":
        from online.loop import OnlineLoop

        rounds = 2
        episodes = 2
        transitions = 4
        eval_steps = [episodes * transitions * (index + 1) for index in range(rounds)]
        result = OnlineLoop().run(
            spec,
            seed=args.seed,
            rounds=rounds,
            episodes_per_round=episodes,
            transitions_per_episode=transitions,
            eval_steps=eval_steps,
            budget=SelectionBudget(max_episodes=2, retention_ratio=args.retention_ratio),
        )
        result["paper_result"] = False
        result["requested_benchmark"] = args.benchmark
        result["method"] = spec.name
        result["training_optimizer_steps"] = 0
        _write(out / "result.json", result)
        _write(out / "state.json", {"completed_phases": ["synthetic_online"], "paper_result": False})
        return 0
    set_seed(args.seed)
    pool = [
        make_synthetic_trajectory(f"ep{i}", 4, score=float(i), collected_step=i, success=(i % 2 == 0))
        for i in range(4)
    ]
    budget = SelectionBudget(
        max_episodes=2,
        max_transitions=args.online_steps,
        retention_ratio=args.retention_ratio,
    )
    selected = spec.select(
        pool,
        budget,
        SelectionContext(
            seed=args.seed,
            scores={item.trajectory_id: item.score or 0.0 for item in pool},
            score_provenance="synthetic_interface_scores",
        ),
    )
    payload = _common(
        args,
        {
            "status": SYNTHETIC_INTERFACE_SMOKE,
            "benchmark": "synthetic_vector",
            "requested_benchmark": args.benchmark,
            "selection_status": selected.status,
            "selected_ids": selected.selected_ids,
            "selected_episodes": selected.selected_episodes,
            "selected_transitions": selected.selected_transitions,
            "training_optimizer_steps": 0,
            "reason": "CPU synthetic selector smoke. Not a published benchmark number.",
        },
    )
    _write(out / "result.json", payload)
    _write(out / "state.json", {"completed_phases": ["plan", "synthetic_select"], "paper_result": False})
    return 0


def _real(args: argparse.Namespace, spec, bench, policy: dict, out: Path) -> int:
    from policies.diffusion_policy_image.adapter import execute_image_dp

    cuda_ok = bool(policy.get("cuda"))
    blocker = real_device_blocker(
        args.device,
        cuda_ok=cuda_ok,
        cuda_detail=policy.get("torch_import_error") or "",
    )
    if blocker:
        _write(
            out / "result.json",
            _common(
                args,
                {
                    "status": BLOCKED_NO_CUDA,
                    "reason": blocker,
                    "policy_probe": policy,
                    "will_train": False,
                    "training_optimizer_steps": None,
                },
            ),
        )
        return 2
    if args.protocol == "online_closed_loop":
        from online.loop import RealOnlineProtocol

        result = RealOnlineProtocol().run(
            seed=args.seed,
            online_steps=args.online_steps,
            policy_ready=False,
            simulator_ready=False,
            supervision_ready=False,
        )
        _write(out / "result.json", _common(args, result))
        return 2
    if spec is None:
        _write(
            out / "result.json",
            _common(args, {"status": "NOT_APPLICABLE", "reason": "HinFlow is not a unified selector.", "will_train": False}),
        )
        return 2
    if args.resume and not args.checkpoint:
        state_path = out / "state.json"
        if not state_path.is_file() and not (out / "checkpoints").is_dir():
            _write(
                out / "result.json",
                _common(
                    args,
                    {
                        "status": "BLOCKED",
                        "reason": "resume requested but no checkpoint or state.json exists in the output directory",
                        "will_train": False,
                        "training_optimizer_steps": None,
                    },
                ),
            )
            return 2
    config = _load_config(args)
    dataset = _dataset_path(args, bench, config)
    loaded = bench.load_dataset(dataset)
    if loaded.get("status") != "SCHEMA_OK":
        payload = {
            "status": "BLOCKED",
            "reason": loaded.get("reason") or "dataset is not ready",
            "dataset_status": loaded.get("status"),
            "dataset_path": dataset,
            "will_train": False,
            "training_optimizer_steps": None,
            "reported_as_mh": False,
        }
        if str(args.benchmark).startswith("robomimic"):
            payload["task_yaml_dataset_type"] = "ph"
        _write(out / "result.json", _common(args, payload))
        return 2
    if spec.kind == "published_method":
        blocked = spec.select(
            [make_synthetic_trajectory("author_probe", 2)],
            SelectionBudget(max_episodes=1),
            SelectionContext(seed=args.seed),
        )
        _write(
            out / "result.json",
            _common(
                args,
                {
                    "status": blocked.status,
                    "reason": blocked.reason,
                    "selected_ids": blocked.selected_ids,
                    "official_stack_invoked": blocked.extra_cost.get("official_stack_invoked"),
                    "heuristic_fallback": False,
                    "dataset_path": dataset,
                    "will_train": False,
                    "training_optimizer_steps": None,
                    "note": "Author stack did not return scores for this dataset. Training was not started.",
                },
            ),
        )
        return 2
    if spec.name != "all_data":
        _write(
            out / "result.json",
            _common(
                args,
                {
                    "status": "BLOCKED",
                    "reason": (
                        "real mode does not select a synthetic probe. "
                        "Materialized benchmark episodes are required before this control can run."
                    ),
                    "will_train": False,
                    "training_optimizer_steps": None,
                    "dataset_path": dataset,
                },
            ),
        )
        return 2
    if spec.name == "all_data" and args.retention_ratio not in (None, 1.0):
        _write(
            out / "result.json",
            _common(
                args,
                {
                    "status": "BLOCKED",
                    "reason": "all_data refuses a retention ratio below 1. Use another method name for a subset.",
                    "will_train": False,
                    "training_optimizer_steps": None,
                },
            ),
        )
        return 2
    request = {
        "method": args.method,
        "benchmark": args.benchmark,
        "task": args.task or getattr(bench, "task", ""),
        "seed": args.seed,
        "device": args.device,
        "dataset_path": dataset,
        "checkpoint": args.checkpoint,
        "resume": bool(args.resume),
        "output_dir": str(out),
        "expected_variant": "mh" if str(args.benchmark).endswith("_mh") else None,
        "dry_run": False,
    }
    trained = execute_image_dp(request)
    trained = _common(args, trained)
    trained["dataset_path"] = dataset
    _write(out / "result.json", trained)
    return 0 if trained.get("status") == "COMPLETED_VERIFIED" else 2
