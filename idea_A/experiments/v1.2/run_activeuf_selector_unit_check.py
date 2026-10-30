#!/usr/bin/env python3
"""Import official ActiveUF selectors and call them on synthetic RM-UQ tensors.

This is selector_unit_check, not a comparison cell. Protocol A2 remains
INCOMPARABLE / BLOCKED: there is no live reward-model UQ pool, and the
pre-registered mapping from A-v1.1 variance allocation to pair selection
was rejected in reports/v1.2/03_COMPARABILITY_AUDIT.md before any numbers.

Does not import vLLM. Does not rewrite DRTS / DeltaUCB / InfoMax.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CLONE_ROOT = REPO_ROOT / "idea_A" / "baselines" / "ActiveUltraFeedback"
OUT_DIR = (
    REPO_ROOT
    / "results"
    / "v1.2"
    / "idea_A"
    / "original"
    / "activeuf_selector_unit_check"
)
OFFICIAL_FILES = {
    "drts": CLONE_ROOT / "activeuf" / "acquisition_function" / "drts.py",
    "deltaucb": CLONE_ROOT / "activeuf" / "acquisition_function" / "deltaucb.py",
    "infomax": CLONE_ROOT / "activeuf" / "acquisition_function" / "infomax.py",
}


def _class_record(cls) -> dict:
    path = Path(inspect.getfile(cls)).resolve()
    return {
        "qualname": cls.__qualname__,
        "module": cls.__module__,
        "file": str(path),
        "file_is_official": path == OFFICIAL_FILES[
            "drts"
            if cls.__qualname__ == "DoubleReverseThompsonSampling"
            else "deltaucb"
            if cls.__qualname__ == "DeltaUCB"
            else "infomax"
        ].resolve()
        if cls.__qualname__ in {
            "DoubleReverseThompsonSampling",
            "DeltaUCB",
            "InfoMax",
        }
        else False,
    }


def run_check(seed: int = 0) -> dict:
    if str(CLONE_ROOT) not in sys.path:
        sys.path.insert(0, str(CLONE_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import numpy as np
    import torch

    from activeuf.acquisition_function.deltaucb import DeltaUCB
    from activeuf.acquisition_function.drts import DoubleReverseThompsonSampling
    from activeuf.acquisition_function.infomax import InfoMax

    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    # Synthetic RM scores and symmetric bounds. These are NOT oracle labels
    # and NOT Idea A outcomes. Official __call__ takes only these three tensors.
    rewards = torch.tensor(
        [
            [0.10, 0.50, 0.20, 0.90],
            [0.00, 1.00, 0.30, 0.40],
        ],
        dtype=torch.float32,
    )
    half_width = torch.tensor(
        [
            [0.20, 0.15, 0.25, 0.10],
            [0.30, 0.05, 0.20, 0.15],
        ],
        dtype=torch.float32,
    )
    lower = rewards - half_width
    upper = rewards + half_width
    n_prompts, n_completions = rewards.shape
    candidate_pool = {
        "n_prompts": int(n_prompts),
        "n_completions_per_prompt": int(n_completions),
        "pairs_per_prompt": 1,
        "note": "same synthetic pool for all three official selectors",
    }

    records = {}
    for key, cls in (
        ("drts", DoubleReverseThompsonSampling),
        ("deltaucb", DeltaUCB),
        ("infomax", InfoMax),
    ):
        rec = _class_record(cls)
        obj = cls()
        selected = obj(rewards, lower, upper)
        rec["selected_pairs"] = [
            [int(a), int(b)] for a, b in selected
        ]
        rec["n_pairs"] = len(rec["selected_pairs"])
        rec["pair_indices_in_range"] = all(
            0 <= i < n_completions and 0 <= j < n_completions and i != j
            for i, j in rec["selected_pairs"]
        )
        rec["call_args"] = ["rewards", "lower_bounds", "upper_bounds"]
        rec["oracle_labels_passed"] = False
        records[key] = rec

    vllm_imported = "vllm" in sys.modules
    return {
        "status": "ok",
        "comparability_cell": None,
        "protocol_a2": "BLOCKED_INCOMPARABLE",
        "check_type": "selector_unit_check",
        "clone_commit_expected": "3cf700a85bbe344b1b472890a488671e2916a8b4",
        "candidate_pool": candidate_pool,
        "budget_pairs_per_prompt": 1,
        "same_candidate_pool": True,
        "oracle_labels_in_acquisition_scores": False,
        "vllm_imported": vllm_imported,
        "selectors": records,
        "all_official_files": all(item["file_is_official"] for item in records.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_lines = [
        "selector_unit_check: official ActiveUF DRTS/DeltaUCB/InfoMax",
        f"python: {sys.executable}",
        f"clone_root: {CLONE_ROOT}",
        "not a DIRECT/ADAPTED comparison cell",
    ]
    payload: dict
    try:
        payload = run_check(seed=int(args.seed))
        log_lines.append("status: ok")
        for key, rec in payload["selectors"].items():
            log_lines.append(
                f"{key}: {rec['qualname']} file={rec['file']} pairs={rec['selected_pairs']}"
            )
    except Exception as exc:
        payload = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "comparability_cell": None,
            "protocol_a2": "BLOCKED_INCOMPARABLE",
            "check_type": "selector_unit_check",
            "vllm_imported": "vllm" in sys.modules,
        }
        log_lines.append(f"status: failed {type(exc).__name__}: {exc}")
        log_lines.append(traceback.format_exc())

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from shared.evaluation.results import write_run

    env = {
        "python": sys.executable,
        "python_version": sys.version,
        "torch_available": True,
        "vllm_imported": bool(payload.get("vllm_imported")),
        "clone_root": str(CLONE_ROOT),
        "repo_root": str(REPO_ROOT),
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:
        env["torch_available"] = False
        env["torch_error"] = f"{type(exc).__name__}: {exc}"

    config = {
        "experiment_id": "activeuf_selector_unit_check",
        "protocol": None,
        "comparability": "not_a_comparison_cell",
        "seed": int(args.seed),
        "official_selectors": ["drts", "deltaucb", "infomax"],
        "note": (
            "Calls official ActiveUF classes on synthetic RM-UQ tensors. "
            "Does not establish Protocol A2."
        ),
    }
    write_run(out_dir, config, payload, env, "\n".join(log_lines) + "\n")
    print(json.dumps({"out_dir": str(out_dir), "status": payload.get("status")}, indent=2))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
