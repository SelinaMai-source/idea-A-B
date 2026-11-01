#!/usr/bin/env python3
"""Idea C v0 experiment runner stub.

This commit only scaffolds packages. It must not start Monte Carlo, ranking,
or any paper-scale / pilot simulation. Checkpoint and resume will live here
once algorithms exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IDEA_C = Path(__file__).resolve().parents[1]
PLANNED = [
    "p0_aaai_robust_gaussian",
    "p0_eps_gaussian",
    "p0_eps_studentt",
    "p1a_overlap_multiarm_clean",
    "p1a_overlap_multiarm_contaminated",
    "p0_score_gaussian_mixture",
    "p0_uai_local_dependence",
    "p1b_overlap_streaming",
]


def list_experiments() -> int:
    raw = IDEA_C / "results" / "v0" / "raw"
    print("Idea C v0 planned experiments (NOT_RUN):")
    for name in PLANNED:
        print(f"  - {name}")
    print(f"raw dir: {raw} exists={raw.exists()}")
    print("Algorithms are empty packages. Refusing to run Monte Carlo.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Idea C v0 runner (stub)")
    parser.add_argument("--list", action="store_true", help="list planned experiments")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reserved for checkpoint resume; not implemented",
    )
    parser.add_argument("experiment", nargs="?", help="experiment id (refused)")
    args = parser.parse_args(argv)
    if args.list or args.experiment is None:
        return list_experiments()
    print(
        f"Refusing to start {args.experiment}: v0 skeleton has empty algorithm "
        "packages and must not run Monte Carlo.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
