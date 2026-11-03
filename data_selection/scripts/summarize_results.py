#!/usr/bin/env python3
"""Summarize JSONL/JSON results that already exist. Does not invent metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize existing data_selection results.")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--output-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def collect(results_dir: Path) -> list[dict]:
    rows = []
    if not results_dir.is_dir():
        return rows
    for path in sorted(results_dir.rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["_source"] = str(path)
                rows.append(row)
    for path in sorted(results_dir.rglob("result.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_source"] = str(path)
        rows.append(row)
    return rows


def main() -> int:
    args = build_parser().parse_args()
    results_dir = Path(args.results_dir)
    rows = collect(results_dir)
    if args.dry_run:
        if not args.output_dir:
            raise SystemExit("--output-dir is required with --dry-run")
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plan = {
            "dry_run": True,
            "will_train": False,
            "results_dir": str(results_dir),
            "files_found": len(rows),
            "resume": bool(args.resume),
        }
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        return 0
    real = [row for row in rows if row.get("paper_result") is True]
    synthetic = [row for row in rows if row.get("status") == "SYNTHETIC_INTERFACE_SMOKE"]
    failed = [row for row in rows if row.get("status") == "FAILED"]
    summary = {
        "paper_results": len(real),
        "synthetic_interface_rows": len(synthetic),
        "failed_rows": len(failed),
        "other_rows": len(rows) - len(real) - len(synthetic),
        "benchmark_cells": "NOT_RUN" if not real else "SEE_ROWS",
        "sources": [row.get("_source") for row in rows],
        "failed_sources": [row.get("_source") for row in failed],
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
