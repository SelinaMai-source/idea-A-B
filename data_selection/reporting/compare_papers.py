"""Compare local runs with extracted paper numbers. Does not invent either side."""

from __future__ import annotations

import json
from pathlib import Path


def compare(paper_rows: list[dict], local_rows: list[dict]) -> list[dict]:
    local_index = {
        (row.get("method"), row.get("benchmark"), row.get("table_id")): row for row in local_rows
    }
    compared = []
    for paper in paper_rows:
        key = (paper.get("method"), paper.get("benchmark"), paper.get("table_id"))
        local = local_index.get(key)
        published = paper.get("published_value", "NOT_EXTRACTED")
        local_value = "NOT_RUN" if local is None else local.get("value", "NOT_RUN")
        gap = "NOT_RUN"
        if isinstance(published, (int, float)) and isinstance(local_value, (int, float)):
            gap = local_value - published
        compared.append(
            {
                "method": paper.get("method"),
                "benchmark": paper.get("benchmark"),
                "table_id": paper.get("table_id"),
                "paper_metric": paper.get("metric", "NOT_EXTRACTED"),
                "published_value": published,
                "local_value": local_value,
                "absolute_difference": gap if not isinstance(gap, float) else abs(gap),
                "seeds": "NOT_RUN" if local is None else local.get("seeds", "NOT_RUN"),
                "fidelity_status": "NOT_RUN",
                "source_commit": paper.get("source_commit", "NOT_RUN"),
            }
        )
    return compared


def write_comparison(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
