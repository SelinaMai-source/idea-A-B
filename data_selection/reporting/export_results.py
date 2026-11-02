"""Write CSV and JSONL. Parquet is attempted locally and is gitignored upstream."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


def export_rows(rows: list[dict], output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        payload = {"rows": 0, "csv": "", "jsonl": "", "parquet": "NOT_GENERATED_EMPTY"}
        (output_dir / "export_status.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload
    fieldnames = list(rows[0].keys())
    csv_path = output_dir / "metrics.csv"
    jsonl_path = output_dir / "metrics.jsonl"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    parquet_path = output_dir / "metrics.parquet"
    parquet_status = "NOT_GENERATED"
    try:
        pd.DataFrame(rows).to_parquet(parquet_path)
        parquet_status = "WRITTEN_LOCAL_GITIGNORED"
    except Exception as exc:  # noqa: BLE001 - missing engine is a recorded status
        parquet_status = f"NOT_GENERATED: {type(exc).__name__}: {exc}"
    payload = {
        "rows": len(rows),
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "parquet": parquet_status,
    }
    (output_dir / "export_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
