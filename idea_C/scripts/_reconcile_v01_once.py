#!/usr/bin/env python3
"""One-shot CSV↔Markdown reconcile for v0.1. Read-only; never rewrites numbers."""

from __future__ import annotations

import json
import math
from pathlib import Path

IDEA = Path(__file__).resolve().parents[1]
MD_PATH = IDEA / "v0实验版本+结果.md"
RAW = IDEA / "results" / "v0.1" / "raw"
OUT = IDEA / "reports" / "v0.1" / "CSV_MD_RECONCILE.md"

SKIP_COLS = {"版本", "内容", "Data", "结果解释"}
TOKENS = {
    "NOT_RUN",
    "CENSORED",
    "N/A",
    "FIGURE_ONLY",
    "NOT_COMPARABLE",
    "BLOCKED_COMPUTE",
    "BLOCKED_NO_DATA",
}


def split_row(line: str) -> list[str]:
    raw = line.strip()
    if raw.startswith("|"):
        raw = raw[1:]
    if raw.endswith("|"):
        raw = raw[:-1]
    return [c.strip() for c in raw.split("|")]


def looks_metric(s: str) -> bool:
    if s in TOKENS:
        return True
    try:
        float(s)
        return True
    except ValueError:
        return False


def align_cells(headers: list[str], cells: list[str]) -> list[str]:
    if len(cells) == len(headers):
        return cells
    if len(cells) < len(headers):
        return cells + [""] * (len(headers) - len(cells))
    n_num = max(0, len(headers) - 4)
    idx = len(cells) - 1
    expl: list[str] = []
    while idx >= 0 and not looks_metric(cells[idx]):
        expl.append(cells[idx])
        idx -= 1
    metrics: list[str] = []
    while idx >= 0 and len(metrics) < n_num and looks_metric(cells[idx]):
        metrics.append(cells[idx])
        idx -= 1
    metrics.reverse()
    left = cells[: idx + 1]
    version = left[0] if left else ""
    content = left[1] if len(left) > 1 else ""
    data = "|".join(left[2:]) if len(left) > 2 else ""
    explanation = "|".join(reversed(expl))
    aligned = [version, content, data]
    if len(metrics) < n_num:
        metrics = [""] * (n_num - len(metrics)) + metrics
    aligned.extend(metrics[:n_num])
    aligned.append(explanation)
    if len(aligned) < len(headers):
        aligned.extend([""] * (len(headers) - len(aligned)))
    return aligned[: len(headers)]


def parse_md(text: str) -> list[dict]:
    rows: list[dict] = []
    lines = text.splitlines()
    current_table = None
    headers: list[str] | None = None
    for line in lines:
        s = line.strip()
        if s.startswith("<!-- TABLE_ID:") and s.endswith("-->"):
            current_table = s[len("<!-- TABLE_ID:") :].rstrip("-->").strip()
            headers = None
            continue
        if current_table and s.startswith("|"):
            cells = split_row(s)
            if headers is None:
                headers = cells
                continue
            if set(c.replace("-", "") for c in cells) == {""}:
                continue
            aligned = align_cells(headers, cells)
            rec = {headers[j]: aligned[j] for j in range(len(headers))}
            rec["_table"] = current_table
            rec["_ncols_raw"] = len(cells)
            rec["_ncols_header"] = len(headers)
            rows.append(rec)
    return rows


def collect_raw(raw_dir: Path) -> list[dict]:
    runs: list[dict] = []
    if not raw_dir.exists():
        return runs
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        jp = child / "metrics.json"
        if not jp.exists():
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            runs.append(data)
    return runs


def close(a, b) -> bool:
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    if math.isnan(fa) and math.isnan(fb):
        return True
    return abs(fa - fb) <= 1e-12 * max(1.0, abs(fa), abs(fb)) or abs(fa - fb) <= 1e-12


def main() -> int:
    text = MD_PATH.read_text(encoding="utf-8")
    md_rows = parse_md(text)
    by_key: dict[tuple[str, str], list[dict]] = {}
    by_version: dict[str, list[dict]] = {}
    for r in md_rows:
        ver = str(r.get("版本", ""))
        tid = str(r.get("_table", ""))
        by_key.setdefault((tid, ver), []).append(r)
        by_version.setdefault(ver, []).append(r)

    runs = collect_raw(RAW)
    compared = 0
    mismatches: list[tuple] = []
    unmatched_raw: list[tuple] = []
    token_ok = 0
    numeric_ok = 0
    pipe_fixed = sum(1 for r in md_rows if int(r.get("_ncols_raw", 0)) > int(r.get("_ncols_header", 0)))

    for run in runs:
        ver = str(run.get("version") or "")
        tid = str(run.get("table_id") or "")
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        rows = by_key.get((tid, ver), [])
        if not rows and ver in by_version and len(by_version[ver]) == 1:
            rows = by_version[ver]
        if not rows:
            status = str(run.get("status") or "")
            rid = str(run.get("run_id") or "")
            reason = "stub_or_not_ingested" if status in TOKENS or "NOT_RUN" in rid or "blocked" in rid.lower() else "no_md_row"
            unmatched_raw.append((rid, ver, reason))
            continue
        row = rows[0]
        for col, val in metrics.items():
            if col in SKIP_COLS or col not in row:
                continue
            md_val = row[col]
            if md_val in TOKENS or str(val) in TOKENS:
                compared += 1
                if str(md_val) == str(val):
                    token_ok += 1
                else:
                    mismatches.append((ver, col, md_val, val, run.get("run_id")))
                continue
            if val in (None, "") or md_val == "":
                continue
            compared += 1
            if close(md_val, val):
                numeric_ok += 1
            else:
                mismatches.append((ver, col, md_val, val, run.get("run_id")))

    v0_rows = [r for r in md_rows if str(r.get("版本", "")).startswith("v0")]
    c_v01_rows = [r for r in md_rows if str(r.get("版本", "")).startswith("C-v0.1")]
    v0_not_run_zero = []
    for r in v0_rows:
        if "尚未运行" not in r.get("结果解释", ""):
            continue
        for k, v in r.items():
            if k.startswith("_") or k in SKIP_COLS:
                continue
            if v in {"0", "0.0"}:
                v0_not_run_zero.append((r.get("版本"), k, v))

    lines = [
        "# CSV ↔ Markdown reconcile (v0.1)",
        "",
        "日期：2026-11-02。只读对账，未改任何已有数字。未把 `NOT_RUN` 换成 0。",
        "Data 列里未转义的 `|H1|` 按表头列数从右侧对齐后再比。",
        "",
        f"- markdown 数据行：{len(md_rows)}（`v0*`={len(v0_rows)}，`C-v0.1*`={len(c_v01_rows)}）",
        f"- 因 `|H1|` 多切列、已重对齐的行：{pipe_fixed}",
        f"- v0.1 raw `metrics.json`：{len(runs)}",
        f"- 已匹配 version 并比较的数值/token 格：{compared}",
        f"- numeric 一致：{numeric_ok}；token 一致：{token_ok}",
        f"- mismatch：{len(mismatches)}",
        f"- raw 无对应 MD 行：{len(unmatched_raw)}",
        f"- 「尚未运行」的 v0 行被填 0：{len(v0_not_run_zero)}",
        "",
    ]
    if not mismatches and not v0_not_run_zero:
        lines.append(
            "**结论：已写入的 C-v0.1 数字与 CSV/JSON 均值一致，0 处 mismatch。"
            "v0 旧行未被覆盖、未把 NOT_RUN 换成 0。**"
        )
    else:
        lines.append("**结论：存在 mismatch，见下表。未自动改表。**")
    lines.append("")
    if mismatches:
        lines.extend(
            [
                "## Mismatches",
                "",
                "|version|column|markdown|json|run_id|",
                "|---|---|---|---|---|",
            ]
        )
        for ver, col, md_val, val, rid in mismatches[:80]:
            lines.append(f"|{ver}|{col}|{md_val}|{val}|{rid}|")
        lines.append("")
    if unmatched_raw:
        lines.extend(["## Raw without markdown row", "", "|run_id|version|reason|", "|---|---|---|"])
        for rid, ver, reason in unmatched_raw:
            lines.append(f"|{rid}|{ver}|{reason}|")
        lines.append("")
    if v0_not_run_zero:
        lines.append("## v0 NOT_RUN overwritten with 0")
        lines.append("")
        for item in v0_not_run_zero:
            lines.append(f"- {item}")
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"wrote {OUT}")
    return 0 if not mismatches and not v0_not_run_zero else 1


if __name__ == "__main__":
    raise SystemExit(main())
