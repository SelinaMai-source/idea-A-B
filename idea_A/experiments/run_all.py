"""Run the Idea A synthetic matrix and ablations A1–A5, then refresh SMOKE_SUMMARY.json."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
RESULTS = REPO_ROOT / "idea_A" / "results"

ABLATIONS = [
    REPO_ROOT / "idea_A" / "ablations" / "A1_No_Adaptive_Allocation" / "run.py",
    REPO_ROOT / "idea_A" / "ablations" / "A2_No_Propensity_Correction" / "run.py",
    REPO_ROOT / "idea_A" / "ablations" / "A3_No_Anchor_Intervention" / "run.py",
    REPO_ROOT / "idea_A" / "ablations" / "A4_Fixed_vs_Adaptive_Budget" / "run.py",
    REPO_ROOT / "idea_A" / "ablations" / "A5_Misspecified_Propensity" / "run.py",
]


def _run(script: Path) -> int:
    print(f"[idea_A] running {script}", flush=True)
    completed = subprocess.run(
        [PYTHON, str(script)],
        cwd=str(REPO_ROOT),
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(REPO_ROOT)},
    )
    return int(completed.returncode)


def _attach_ablations(summary: dict) -> dict:
    summary = dict(summary)
    ablations = {}
    for path in ABLATIONS:
        index_path = path.parent / "results" / "INDEX.json"
        if index_path.is_file():
            ablations[path.parent.name] = json.loads(index_path.read_text(encoding="utf-8"))
        else:
            ablations[path.parent.name] = {"missing": True, "path": str(index_path)}
    summary["ablations"] = ablations
    summary["ablation_skip_note"] = (
        "A1–A5 ran all mechanisms at b=0.05 plus all budgets on mnar and "
        "zero_overlap, seeds 0/1/2. Other ablation budget×mechanism cells "
        "were skipped to keep MacBook runtime feasible. One component changed per ablation."
    )
    return summary


def main() -> int:
    failed = _run(REPO_ROOT / "idea_A" / "experiments" / "run_synthetic.py")
    for script in ABLATIONS:
        failed = _run(script) or failed
    summary_path = RESULTS / "SMOKE_SUMMARY.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload = _attach_ablations(payload)
        summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[idea_A] refreshed {summary_path}", flush=True)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
