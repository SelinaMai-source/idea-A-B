#!/usr/bin/env python3
"""Append-only updater for idea_C/v0实验版本+结果.md.

Reads results/v0/raw/<run_id>/{metrics.json,metrics.csv} and appends rows.
Supports P0 tables and P1-A Overlap A (overlap_a_clean / overlap_a_contaminated)
plus P1-B. Never overwrites an existing data row or the human 结果解释 cell.
Empty or malformed raw dirs must not crash. Missing numeric cells stay NOT_RUN.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

IDEA_C = Path(__file__).resolve().parents[1]
DEFAULT_MD = IDEA_C / "v0实验版本+结果.md"
DEFAULT_RAW = IDEA_C / "results" / "v0" / "raw"
DEFAULT_SEEN = IDEA_C / "results" / "v0" / "summaries" / "ingested_run_ids.json"

TABLE_MARK = "<!-- TABLE_ID:"


def load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(data, list):
        return {str(x) for x in data}
    if isinstance(data, dict) and isinstance(data.get("run_ids"), list):
        return {str(x) for x in data["run_ids"]}
    return set()


def save_seen(path: Path, seen: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_ids": sorted(seen)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_json_metrics(path: Path) -> dict | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        data = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def parse_csv_metrics(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            row = next(reader, None)
    except (OSError, csv.Error, StopIteration):
        return None
    if not row:
        return None
    return dict(row)


def collect_runs(raw_dir: Path) -> list[dict]:
    runs: list[dict] = []
    if not raw_dir.exists() or not raw_dir.is_dir():
        return runs
    try:
        children = sorted(raw_dir.iterdir())
    except OSError:
        return runs
    for child in children:
        if not child.is_dir():
            continue
        payload: dict = {"run_id": child.name, "table_id": None, "version": child.name, "content": "", "data": "", "metrics": {}, "status": "ok", "result_explanation": "自动追加，结果解释待人工填写"}
        json_path = child / "metrics.json"
        csv_path = child / "metrics.csv"
        parsed = None
        if json_path.exists():
            parsed = parse_json_metrics(json_path)
        if parsed is None and csv_path.exists():
            parsed = parse_csv_metrics(csv_path)
        if parsed is None:
            continue
        payload["table_id"] = parsed.get("table_id") or parsed.get("TABLE_ID")
        payload["version"] = str(parsed.get("version") or parsed.get("版本") or child.name)
        payload["content"] = str(parsed.get("content") or parsed.get("内容") or "")
        payload["data"] = str(parsed.get("data") or parsed.get("Data") or "")
        metrics = parsed.get("metrics")
        if isinstance(metrics, dict):
            payload["metrics"] = {str(k): v for k, v in metrics.items()}
        else:
            skip = {"table_id", "TABLE_ID", "version", "版本", "content", "内容", "data", "Data", "run_id", "status", "result_explanation", "结果解释"}
            payload["metrics"] = {k: v for k, v in parsed.items() if k not in skip}
        payload["status"] = str(parsed.get("status") or "ok")
        note = parsed.get("result_explanation") or parsed.get("结果解释")
        if note:
            payload["result_explanation"] = str(note)
        payload["run_id"] = str(parsed.get("run_id") or child.name)
        runs.append(payload)
    return runs


def find_tables(lines: list[str]) -> list[tuple[str, int, int]]:
    """Return (table_id, header_index, last_data_row_index) inclusive."""
    found: list[tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith(TABLE_MARK) and stripped.endswith("-->"):
            inner = stripped[len(TABLE_MARK) :].rstrip("-->").strip()
            table_id = inner
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith("|"):
                j += 1
            if j >= len(lines):
                i += 1
                continue
            header_idx = j
            k = j + 1
            while k < len(lines) and lines[k].lstrip().startswith("|"):
                k += 1
            last_row = k - 1
            found.append((table_id, header_idx, last_row))
            i = k
            continue
        i += 1
    return found


def split_row(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [cell.strip() for cell in raw.split("|")]


def join_row(cells: list[str]) -> str:
    return "|" + "|".join(cells) + "|"


def append_rows(md_text: str, runs: list[dict], seen: set[str]) -> tuple[str, int, list[str]]:
    lines = md_text.splitlines()
    tables = find_tables(lines)
    id_to_table = {tid: (h, last) for tid, h, last in tables}
    inserted = 0
    warnings: list[str] = []
    # Insert from the bottom so later indices stay valid.
    by_table: dict[str, list[dict]] = {}
    for run in runs:
        rid = run["run_id"]
        if rid in seen:
            continue
        tid = run.get("table_id")
        if not tid or tid not in id_to_table:
            warnings.append(f"skip {rid}: unknown table_id={tid!r}")
            continue
        by_table.setdefault(str(tid), []).append(run)

    for tid, (header_idx, last_idx) in sorted(id_to_table.items(), key=lambda kv: kv[1][1], reverse=True):
        batch = by_table.get(tid, [])
        if not batch:
            continue
        header_cells = split_row(lines[header_idx])
        new_lines: list[str] = []
        for run in batch:
            cells = []
            lookup = {
                "版本": run["version"],
                "内容": run["content"],
                "Data": run["data"],
                "结果解释": run.get("result_explanation") or "自动追加，结果解释待人工填写",
            }
            metrics = run.get("metrics") or {}
            for col in header_cells:
                if col in lookup:
                    cells.append(str(lookup[col]))
                elif col in metrics and metrics[col] not in (None, ""):
                    cells.append(str(metrics[col]))
                else:
                    # Missing numeric cells stay NOT_RUN, never 0.
                    cells.append("NOT_RUN")
            new_lines.append(join_row(cells))
            seen.add(run["run_id"])
            inserted += 1
        lines[last_idx + 1 : last_idx + 1] = new_lines
    return "\n".join(lines) + ("\n" if md_text.endswith("\n") else ""), inserted, warnings


def update_markdown(md_path: Path, raw_dir: Path, seen_path: Path) -> int:
    if not md_path.exists():
        print(f"markdown not found: {md_path}", file=sys.stderr)
        return 1
    text = md_path.read_text(encoding="utf-8")
    seen = load_seen(seen_path)
    runs = collect_runs(raw_dir)
    new_text, n_inserted, warnings = append_rows(text, runs, seen)
    for msg in warnings:
        print(msg, file=sys.stderr)
    if n_inserted:
        md_path.write_text(new_text, encoding="utf-8")
        save_seen(seen_path, seen)
    print(f"raw_runs_parsed={len(runs)} rows_appended={n_inserted} seen={len(seen)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Append raw metrics into v0实验版本+结果.md")
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--seen", type=Path, default=DEFAULT_SEEN)
    args = parser.parse_args(argv)
    return update_markdown(args.md, args.raw, args.seen)


if __name__ == "__main__":
    raise SystemExit(main())
