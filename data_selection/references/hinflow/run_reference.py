#!/usr/bin/env python3
"""HinFlow standalone dry-run. Does not launch the official stack."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="HinFlow reference runner")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("HinFlow original training is NOT_RUN. Pass --dry-run to write a plan.")
    if not args.output_dir:
        raise SystemExit("--output-dir is required")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "dry_run": True,
        "method": "hinflow",
        "leaderboard_eligible": False,
        "will_train": False,
        "will_download": False,
        "planned_status": "NOT_RUN",
        "seed": args.seed,
        "device": args.device,
        "resume": bool(args.resume),
        "paper_result": False,
        "reason": "Separate online-learning reference. Image-DP is not substituted.",
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
