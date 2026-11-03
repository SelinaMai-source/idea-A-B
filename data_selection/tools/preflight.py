#!/usr/bin/env python3
"""Fail closed before any image-DP training command. Does not download or train."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import detect_variant_mismatch, variant_from_path  # noqa: E402
from tools.inspect_environment import collect  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for data_selection v0.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-dataset")
    parser.add_argument("--expect-variant")
    parser.add_argument("--expect-task")
    args = parser.parse_args()
    env = collect()
    blockers = []
    if env["torch"]["cuda"] is not True:
        blockers.append("BLOCKED_NO_CUDA")
    if platform.system() == "Darwin":
        blockers.append("MAC_NO_IMAGE_DP_TRAINING")
    dataset_report = None
    if args.require_dataset:
        dataset = Path(args.require_dataset)
        if not dataset.is_file():
            blockers.append(f"DATASET_ABSENT:{dataset}")
            dataset_report = {"path": str(dataset), "present": False}
        else:
            observed = variant_from_path(dataset)
            mismatch = detect_variant_mismatch(args.expect_variant or "", observed)
            dataset_report = {
                "path": str(dataset),
                "present": True,
                "bytes": dataset.stat().st_size,
                "observed_variant": observed,
                "mismatch": mismatch,
            }
            if mismatch:
                blockers.append(mismatch)
            if args.expect_task and args.expect_task.lower() not in str(dataset).lower():
                blockers.append(f"DATASET_TASK_MISMATCH:{args.expect_task}")
    report = {
        "ok_for_image_dp_training": not blockers,
        "blockers": blockers,
        "machine": env["machine"],
        "dataset": dataset_report,
        "will_train": False,
        "will_download": False,
    }
    print(json.dumps(report, indent=2))
    if args.require_cuda and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
