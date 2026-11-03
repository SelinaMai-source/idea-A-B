"""Build the experiment matrix from adapters and audited facts, not from URLs."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.registry import discover_baselines, discover_benchmarks, discover_references

ROOT = Path(__file__).resolve().parents[1]


def _facts(root: Path) -> dict:
    path = root / "manifests" / "method_facts.yaml"
    if not path.is_file():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("methods") or {}


def _paper_status(fact: dict) -> str:
    status = fact.get("implementation_status", "NOT_YET_AUDITED")
    if status == "READY" and not fact.get("entrypoint_verified"):
        return "NOT_YET_AUDITED"
    if status == "READY" and fact.get("official_stack_executed"):
        return "READY"
    if status == "READY":
        return "BLOCKED"
    return status


def _runtime_readiness() -> str:
    try:
        import torch
    except Exception:
        return "BLOCKED_NO_CUDA"
    try:
        if torch.cuda.is_available():
            return "CUDA_AVAILABLE_EXPERIMENT_NOT_RUN"
    except Exception:
        return "BLOCKED_NO_CUDA"
    return "BLOCKED_NO_CUDA"


def _image_dp_compatibility(spec) -> str:
    if spec.kind == "control":
        return "COMPATIBLE_IMAGE_CNN_DP"
    if spec.name in {"cupid", "demo_score"}:
        return "COMPATIBLE_IMAGE_CNN_DP"
    if spec.name in {"deminf", "datamil", "strap"}:
        return "NOT_YET_ADAPTED"
    return "NOT_YET_ADAPTED"


def build_matrix(root: Path | None = None) -> dict:
    root = root or ROOT
    facts = _facts(root)
    baselines = discover_baselines(root)
    benchmarks = discover_benchmarks(root)
    references = discover_references(root)
    runtime = _runtime_readiness()
    rows = []
    unsupported = []
    for spec in baselines:
        fact = facts.get(spec.name, {})
        if spec.kind == "control":
            implementation = "IMPLEMENTED_CPU_SELECTOR"
            original = "NOT_APPLICABLE"
            unified = "NOT_YET_ADAPTED"
            online = "NOT_IMPLEMENTED"
            source_status = "NOT_APPLICABLE"
        else:
            implementation = _paper_status(fact)
            original = fact.get("original_reproduction", "NOT_YET_AUDITED")
            unified = fact.get("unified_adaptation", "NOT_IMPLEMENTED")
            if unified == "NOT_APPLICABLE":
                unified = "NOT_IMPLEMENTED"
            online = fact.get("online_closed_loop", "NOT_IMPLEMENTED")
            source_status = "CLONED" if fact.get("entrypoint_verified") else "NOT_YET_AUDITED"
        robomimic_state = "SELECTOR_INTERFACE_ONLY" if spec.kind == "control" else fact.get(
            "robomimic", "NOT_YET_AUDITED"
        )
        libero_state = "SELECTOR_INTERFACE_ONLY" if spec.kind == "control" else fact.get(
            "libero", "NOT_YET_AUDITED"
        )
        rows.append(
            {
                "method": spec.name,
                "kind": spec.kind,
                "leaderboard_eligible": False,
                "robomimic": robomimic_state,
                "libero": libero_state,
                "image_dp_compatibility": _image_dp_compatibility(spec),
                "original_paper_reproduction": original,
                "unified_adaptation": unified,
                "online_closed_loop": online,
                "required_datasets": fact.get("required_datasets", []),
                "required_checkpoints": fact.get("required_checkpoints", []),
                "estimated_run_cost": fact.get(
                    "estimated_run_cost",
                    "unmeasured; image DP training is not scheduled on this Mac",
                ),
                "implementation_status": implementation,
                "selects": spec.selects,
                "reason": fact.get("reason", ""),
                "SOURCE_STATUS": source_status,
                "IMPLEMENTATION_STATUS": implementation,
                "DEPENDENCY_STATUS": fact.get("dependency_status", "NOT_INSTALLED"),
                "DATA_STATUS": "NOT_PRESENT",
                "MODEL_STATUS": "NOT_TRAINED",
                "ORIGINAL_FIDELITY_STATUS": original,
                "UNIFIED_ADAPTATION_STATUS": unified,
                "RUNTIME_READINESS": runtime,
                "EXPERIMENT_STATUS": "NOT_RUN",
            }
        )
        unsupported.append(
            {
                "method": spec.name,
                "benchmark": "libero_hinflow",
                "reason": "HinFlow's four-task protocol is not a cell of this selector's Image-DP comparison.",
            }
        )
    for ref in references:
        fact = facts.get(ref.name, {})
        rows.append(
            {
                "method": ref.name,
                "kind": ref.kind,
                "leaderboard_eligible": False,
                "robomimic": "NOT_APPLICABLE",
                "libero": fact.get("libero", "NOT_YET_AUDITED"),
                "image_dp_compatibility": "NOT_APPLICABLE",
                "original_paper_reproduction": fact.get("original_reproduction", "NOT_YET_AUDITED"),
                "unified_adaptation": "NOT_APPLICABLE",
                "online_closed_loop": fact.get("online_closed_loop", "NOT_YET_AUDITED"),
                "required_datasets": fact.get("required_datasets", []),
                "required_checkpoints": fact.get("required_checkpoints", []),
                "estimated_run_cost": fact.get("estimated_run_cost", "unmeasured"),
                "implementation_status": _paper_status(fact) if fact else "NOT_YET_AUDITED",
                "selects": getattr(ref, "selects", "not_a_selector"),
                "reason": fact.get(
                    "reason",
                    "HinFlow is an online-learning reference and is not an Image-DP data-selection baseline.",
                ),
                "SOURCE_STATUS": "CLONED_SUBMODULES_INCOMPLETE",
                "IMPLEMENTATION_STATUS": "NOT_APPLICABLE",
                "DEPENDENCY_STATUS": "NOT_INSTALLED",
                "DATA_STATUS": "NOT_PRESENT",
                "MODEL_STATUS": "NOT_TRAINED",
                "ORIGINAL_FIDELITY_STATUS": fact.get("original_reproduction", "BLOCKED"),
                "UNIFIED_ADAPTATION_STATUS": "NOT_APPLICABLE",
                "RUNTIME_READINESS": runtime,
                "EXPERIMENT_STATUS": "NOT_RUN",
            }
        )
    unsupported.append(
        {
            "method": "hinflow",
            "benchmark": "unified_image_dp_leaderboard",
            "reason": "HinFlow's backbone is not the image CNN Diffusion Policy, so it is excluded from that leaderboard.",
        }
    )
    for bench in benchmarks:
        if not bench.forgetting_applicable:
            unsupported.append(
                {
                    "method": "*",
                    "benchmark": bench.name,
                    "metric": "forgetting",
                    "reason": "Forgetting and BWT apply only to the separate task-incremental LIBERO protocol.",
                }
            )
    runnable = [
        {
            "method": spec.name,
            "benchmark": "synthetic_vector",
            "kind": "cpu_selector_or_online_interface",
            "device": "cpu",
        }
        for spec in baselines
        if spec.kind == "control"
    ]
    return {
        "rows": rows,
        "unsupported": unsupported,
        "runnable_jobs": runnable,
        "benchmarks": [bench.name for bench in benchmarks],
        "note": (
            "Runnable jobs are CPU selector checks only. "
            "Image-DP compatibility is not the same field as RUNTIME_READINESS. "
            "A missing CUDA device does not mean the image CNN policy code is absent."
        ),
    }


def write_matrix(root: Path | None = None) -> Path:
    root = root or ROOT
    payload = build_matrix(root)
    path = root / "configs" / "experiment_matrix.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path
