"""Shared CLI. ``--dry-run`` writes a plan JSON and does not train or download."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.registry import discover_baselines, discover_benchmarks  # noqa: E402
from core.seeds import set_seed  # noqa: E402
from core.status import BLOCKED_NO_CUDA, SYNTHETIC_INTERFACE_SMOKE  # noqa: E402
from core.trajectories import make_synthetic_trajectory  # noqa: E402
from core.types import SelectionBudget, SelectionContext  # noqa: E402
from policies.diffusion_policy_image.adapter import probe  # noqa: E402


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
        if policy["status"] == BLOCKED_NO_CUDA and args.device == "cuda":
            reason = reason + " Requested CUDA device is unavailable."
        payload = _plan(args, status, reason)
        payload["policy_probe"] = policy["status"]
        payload["benchmark_forgetting_applicable"] = bench.forgetting_applicable
        out = Path(args.output_dir)
        _write(out / "plan.json", payload)
        state = {"completed_phases": ["plan"], "paper_result": False}
        _write(out / "state.json", state)
        return 0

    out = Path(args.output_dir)
    if args.device == "cuda" or policy["status"] == BLOCKED_NO_CUDA and not args.synthetic:
        payload = {
            "status": BLOCKED_NO_CUDA,
            "paper_result": False,
            "method": args.method,
            "benchmark": args.benchmark,
            "seed": args.seed,
            "reason": "Image Diffusion Policy training is not run on this machine.",
            "policy_probe": policy,
        }
        _write(out / "result.json", payload)
        return 2
    if not args.synthetic:
        payload = {
            "status": "NOT_RUN",
            "paper_result": False,
            "reason": "No benchmark run was requested. Pass --dry-run or --synthetic.",
        }
        _write(out / "result.json", payload)
        return 2
    if spec is None or spec.kind != "control":
        if spec is None:
            _write(
                out / "result.json",
                {
                    "status": "NOT_APPLICABLE",
                    "paper_result": False,
                    "reason": "HinFlow is not a unified selector.",
                },
            )
            return 2
        blocked = spec.select(
            [make_synthetic_trajectory("probe", 2)],
            SelectionBudget(max_episodes=1),
            SelectionContext(seed=args.seed),
        )
        _write(
            out / "result.json",
            {
                "status": blocked.status,
                "reason": blocked.reason,
                "paper_result": False,
                "selected_ids": blocked.selected_ids,
            },
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
    selected = spec.select(pool, budget, SelectionContext(seed=args.seed, scores={item.trajectory_id: item.score or 0.0 for item in pool}))
    payload = {
        "status": SYNTHETIC_INTERFACE_SMOKE,
        "paper_result": False,
        "protocol": args.protocol,
        "method": spec.name,
        "benchmark": "synthetic_vector",
        "requested_benchmark": args.benchmark,
        "seed": args.seed,
        "selection_status": selected.status,
        "selected_ids": selected.selected_ids,
        "selected_episodes": selected.selected_episodes,
        "selected_transitions": selected.selected_transitions,
        "training_optimizer_steps": 0,
        "reason": "CPU synthetic selector smoke. Not a published benchmark number.",
    }
    _write(out / "result.json", payload)
    _write(out / "state.json", {"completed_phases": ["plan", "synthetic_select"], "paper_result": False})
    return 0
