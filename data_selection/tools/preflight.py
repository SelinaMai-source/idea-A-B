#!/usr/bin/env python3
"""Fail closed before any image-DP training command."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.inspect_environment import collect


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight checks for data_selection v0.")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    env = collect()
    blockers = []
    if env["torch"]["cuda"] is not True:
        blockers.append("BLOCKED_NO_CUDA")
    if platform.system() == "Darwin":
        blockers.append("MAC_NO_IMAGE_DP_TRAINING")
    report = {"ok_for_image_dp_training": not blockers, "blockers": blockers, "machine": env["machine"]}
    print(json.dumps(report, indent=2))
    if args.require_cuda and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
