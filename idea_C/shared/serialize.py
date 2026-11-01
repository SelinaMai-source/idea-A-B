"""Serialize one Idea C v0 run under ``idea_C/results/v0/raw/<run_id>/``.

Writes ``config.json``, ``seed.json``, ``commit.json``, ``metrics.json``,
and ``metrics.csv``. Numeric metric cells that are missing or non-finite
are omitted so ``update_v0_results_md.py`` fills ``NOT_RUN`` — never 0.

Column names match ``v0实验版本+结果.md`` so a later agent can append
without inventing numbers.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

IDEA_C = Path(__file__).resolve().parents[1]
DEFAULT_RAW = IDEA_C / "results" / "v0" / "raw"

# Markdown table columns (excluding 版本/内容/Data/结果解释).
TABLE_METRIC_COLUMNS: dict[str, tuple[str, ...]] = {
    "aaai_robust_gaussian": ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓"),
    "aaai_heavy_tailed": ("TPR↑", "FDR↓", "Samples@Target TPR↓", "Runtime↓"),
    "eps_gaussian": ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓"),
    "eps_studentt": ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓"),
    "overlap_a_clean": ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓"),
    "overlap_a_contaminated": ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓"),
}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def git_commit(repo: Path | None = None) -> str | None:
    """Current HEAD hash, or None if git is unavailable."""
    cwd = Path(repo) if repo is not None else IDEA_C
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _finite_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (np.floating,)) and not np.isfinite(value):
            continue
        cleaned[str(key)] = value
    return cleaned


def write_run(
    run_id: str,
    *,
    table_id: str,
    version: str,
    content: str,
    data: str,
    config: dict[str, Any],
    seed: int,
    metrics: dict[str, Any] | None = None,
    status: str = "NOT_RUN",
    result_explanation: str = "自动追加，结果解释待人工填写",
    raw_dir: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write one raw run directory. Missing metric columns stay absent."""
    directory = (raw_dir or DEFAULT_RAW) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    commit = git_commit()
    metric_payload = _finite_metrics(metrics or {})
    record = {
        "run_id": run_id,
        "table_id": table_id,
        "version": version,
        "content": content,
        "data": data,
        "metrics": metric_payload,
        "status": status,
        "result_explanation": result_explanation,
        "seed": int(seed),
        "commit": commit,
        "code_status": "independent_reimplementation",
    }
    if extra:
        record.update(_to_jsonable(extra))

    config_payload = _to_jsonable(dict(config))
    (directory / "config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (directory / "seed.json").write_text(
        json.dumps({"seed": int(seed)}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "commit.json").write_text(
        json.dumps({"commit": commit}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(_to_jsonable(record), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    row = {
        "run_id": run_id,
        "table_id": table_id,
        "version": version,
        "content": content,
        "data": data,
        "status": status,
        "seed": int(seed),
        "commit": commit or "",
    }
    row.update({k: v for k, v in metric_payload.items()})
    fieldnames = list(row.keys())
    with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    return directory
