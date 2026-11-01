#!/usr/bin/env python3
"""Compare extractable paper numbers against local v0.1 CSVs.

Figure-only paper numbers are marked FIGURE_ONLY and never given fake
precision. Missing local runs stay NOT_RUN, never 0.

Tolerances (frozen before results; see v0_1_registry.TOLERANCES):
- deterministic tables: relative gap > 5% => FAIL_MISMATCH until explained
- curves without official CSV => FIGURE_ONLY
- FDR must not exceed alpha + 2 Monte Carlo SE
- 30-seed cannot be full reproduction
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
IDEA_C = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from v0_1_registry import (  # noqa: E402
    OBJECTS,
    TOLERANCES,
    flatten_extractable_numbers,
)

DEFAULT_RAW = IDEA_C / "results" / "v0.1" / "raw"
DEFAULT_OUT = IDEA_C / "reports" / "v0.1" / "PAPER_VS_REPRODUCTION.md"


def _finite(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        token = value.strip()
        if token in {"NOT_RUN", "FIGURE_ONLY", "CENSORED", "N/A", "NOT_LOCATED_IN_PDF"}:
            return None
        try:
            number = float(token)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _rel_gap(local: float, paper: float) -> float:
    if paper == 0.0:
        return abs(local - paper)
    return abs(local - paper) / abs(paper)


def _load_local_rows(raw_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not raw_dir.exists():
        return rows
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        payload: dict[str, Any] = {"run_id": child.name, "path": str(child)}
        json_path = child / "metrics.json"
        csv_path = child / "metrics.csv"
        parsed: dict[str, Any] | None = None
        if json_path.exists():
            try:
                parsed = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                parsed = None
        if parsed is None and csv_path.exists():
            try:
                with csv_path.open(encoding="utf-8", newline="") as handle:
                    parsed = next(csv.DictReader(handle), None)
            except (OSError, csv.Error):
                parsed = None
        if not parsed:
            continue
        payload.update(parsed if isinstance(parsed, dict) else {})
        config_path = child / "config.json"
        if config_path.exists():
            try:
                payload["config"] = json.loads(config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload["config"] = {}
        rows.append(payload)
    return rows


def _local_metric(run: dict[str, Any], metric: str) -> Any:
    metrics = run.get("metrics")
    if isinstance(metrics, dict) and metric in metrics:
        return metrics[metric]
    aliases = {
        "FDR(T)": ["FDR↓", "FDR", "FDR(T)"],
        "Power(T)": ["Power↑", "Power", "TPR↑", "Power(T)"],
        "Discoveries": ["Rejections↑", "Discoveries", "n_rejections"],
        "discoveries coinciding with annotated anomalous periods": [
            "Rejections↑",
            "Discoveries",
        ],
        "TPR": ["TPR↑", "TPR", "TPP↑"],
        "FDR": ["FDR↓", "FDR"],
    }
    keys = aliases.get(metric, [metric, f"{metric}↓", f"{metric}↑"])
    if isinstance(metrics, dict):
        for key in keys:
            if key in metrics:
                return metrics[key]
    for key in keys:
        if key in run:
            return run[key]
    return "NOT_RUN"


OBJECT_ALIASES = {
    "aaai.fig1": ["aaai.fig1", "aaai_robust_gaussian"],
    "aaai.fig2": ["aaai.fig2"],
    "aaai.fig3": ["aaai.fig3", "aaai_heavy_tailed"],
    "aaai.fig4": ["aaai.fig4"],
    "aaai.app_tables": ["aaai.app_tables"],
    "score.fig1": ["score_gaussian_mixture", "score.fig1"],
    "score.fig2": ["score_gaussian_mixture", "score.fig2"],
    "score.fig5": ["score_gaussian_mixture", "score.fig5"],
    "score.fig6": ["score_gaussian_mixture", "score.fig6"],
    "score.tab2": ["score_gaussian_mixture", "score.tab2"],
    "uai.fig2_power": ["uai.fig2_power", "uai_local_dependence"],
    "uai.fig2_runtime": ["uai.fig2_runtime", "m3000"],
    "eps.fig2a": ["eps.fig2a", "eps_gaussian"],
    "eps.fig2c": ["eps.fig2c", "eps_studentt"],
}


def _run_blob(run: dict[str, Any]) -> str:
    cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
    return " ".join(
        [
            str(run.get("run_id") or ""),
            str(run.get("content") or ""),
            str(run.get("version") or ""),
            str(run.get("table_id") or ""),
            str(cfg.get("method") or ""),
            str(cfg.get("object_id") or ""),
            str(run.get("object_id") or ""),
            str(run.get("status") or ""),
        ]
    )


def _matches_object(paper_row: dict[str, Any], run: dict[str, Any]) -> bool:
    object_id = paper_row["object_id"]
    cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
    rid = str(run.get("run_id") or "")
    blob = _run_blob(run)
    if object_id in rid or object_id.replace(".", "-") in rid or object_id in blob:
        return True
    if str(cfg.get("object_id") or "") == object_id or str(run.get("object_id") or "") == object_id:
        return True
    for alias in OBJECT_ALIASES.get(object_id, []):
        if alias in rid or alias in blob:
            return True
    status = str(run.get("status") or "")
    if object_id.replace(".", "-") in rid and status.startswith("BLOCKED"):
        return True
    return False


def _method_in_run(method: str | None, run: dict[str, Any]) -> bool:
    if not method:
        return True
    blob = _run_blob(run)
    variants = {method, method.replace(" ", "-"), method.replace("+", "plus"), method.replace("+", "-")}
    return any(token in blob for token in variants if token)


def _match_runs(paper_row: dict[str, Any], local_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    method = paper_row.get("method")
    family = paper_row.get("family")
    for run in local_rows:
        if not _matches_object(paper_row, run):
            continue
        if method and not _method_in_run(str(method), run):
            continue
        cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
        if paper_row.get("alpha") is not None:
            alpha = cfg.get("alpha")
            if alpha is not None and abs(float(alpha) - float(paper_row["alpha"])) > 1e-9:
                continue
        if paper_row.get("pi1") is not None:
            pi1 = cfg.get("pi1")
            if pi1 is not None and abs(float(pi1) - float(paper_row["pi1"])) > 1e-9:
                continue
        if family:
            blob = _run_blob(run).upper()
            if str(family).upper() not in blob and str(method or "").upper() not in blob:
                continue
        hits.append(run)

    def _rank(run: dict[str, Any]) -> tuple[int, int]:
        n = _n_reps(run) or 0
        scale = str((run.get("config") or {}).get("scale") or run.get("scale") or "")
        paperish = 0 if (n >= 200 or scale == "paper") else 1
        return (paperish, -n)

    hits.sort(key=_rank)
    return hits


def _match_run(paper_row: dict[str, Any], local_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    hits = _match_runs(paper_row, local_rows)
    return hits[0] if hits else None


def _figure_local_summary(paper_row: dict[str, Any], local_rows: list[dict[str, Any]]) -> str:
    hits = _match_runs({**paper_row, "method": None, "alpha": None, "pi1": None}, local_rows)
    if not hits:
        stub = _match_run(paper_row, local_rows)
        if stub and str(stub.get("status") or "").startswith("BLOCKED"):
            return str(stub.get("status"))
        return "NOT_RUN"
    parts: list[str] = []
    seen: set[str] = set()
    for run in hits:
        cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
        method = str(cfg.get("method") or run.get("version") or run.get("run_id") or "run")
        scale = str(cfg.get("scale") or run.get("scale") or "")
        n = _n_reps(run)
        key = f"{method}|{scale}|{n}"
        if key in seen:
            continue
        seen.add(key)
        metrics = run.get("metrics") if isinstance(run.get("metrics"), dict) else {}
        bits = []
        for label, keys in (
            ("TPR", ["TPR↑", "TPR", "Power↑"]),
            ("FDR", ["FDR↓", "FDR"]),
            ("SAT", ["Samples@Target TPR↓"]),
            ("Power", ["Power↑"]),
        ):
            for k in keys:
                if k in metrics and metrics[k] not in (None, ""):
                    bits.append(f"{label}={metrics[k]}")
                    break
        status = str(run.get("status") or "")
        if status and status not in {"ok"}:
            bits.append(status)
        if n is not None:
            bits.append(f"n={n}")
        if scale:
            bits.append(scale)
        parts.append(f"{method}: " + ", ".join(bits) if bits else method)
        if len(parts) >= 6:
            break
    return " | ".join(parts) if parts else "NOT_RUN"


def _n_reps(run: dict[str, Any] | None) -> int | None:
    if not run:
        return None
    cfg = run.get("config") if isinstance(run.get("config"), dict) else {}
    extra = run
    for key in ("n_reps", "n_trials", "reps"):
        if key in extra and extra[key] not in (None, ""):
            try:
                return int(extra[key])
            except (TypeError, ValueError):
                pass
        if key in cfg and cfg[key] not in (None, ""):
            try:
                return int(cfg[key])
            except (TypeError, ValueError):
                pass
    seeds = extra.get("seeds") or cfg.get("seeds")
    if isinstance(seeds, list):
        return len(seeds)
    return None


def _mc_se(run: dict[str, Any] | None, metric: str) -> float | None:
    if not run:
        return None
    series = None
    if "FDR" in metric:
        series = run.get("fdps")
    elif "Power" in metric or metric == "TPR":
        series = run.get("powers") or run.get("tpps")
    if not isinstance(series, list) or len(series) < 2:
        return None
    values = [float(x) for x in series if _finite(x) is not None]
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var / len(values))


def _blocked_status(run: dict[str, Any] | None) -> str | None:
    if not run:
        return None
    status = str(run.get("status") or "")
    if status.startswith("BLOCKED") or status in {"NOT_RUN", "INVALID_PROTOCOL"}:
        return status
    return None


def _verdict(paper_row: dict[str, Any], local_value: Any, run: dict[str, Any] | None) -> str:
    kind = paper_row.get("kind")
    blocked = _blocked_status(run)
    if kind == "FIGURE_ONLY":
        return "FIGURE_ONLY"
    if kind == "NOT_LOCATED":
        return blocked or "NOT_RUN"
    if blocked and local_value in (None, "", "NOT_RUN"):
        return blocked
    if local_value in (None, "", "NOT_RUN") or (
        _finite(local_value) is None and local_value not in {"CENSORED", "N/A"}
    ):
        return blocked or "NOT_RUN"
    n_reps = _n_reps(run)
    if n_reps is not None and n_reps < 200:
        # 30-seed / smoke cannot be full reproduction.
        if paper_row.get("deterministic"):
            local = _finite(local_value)
            paper = _finite(paper_row["paper_value"])
            if local is not None and paper is not None:
                gap = _rel_gap(local, paper)
                if gap > float(TOLERANCES["deterministic_relative_gap_fail"]):
                    return "FAIL_MISMATCH"
            return "NOT_RUN"
        return "NOT_RUN"
    local = _finite(local_value)
    paper = _finite(paper_row["paper_value"])
    if local is None or paper is None:
        return blocked or "NOT_RUN"
    if paper_row.get("deterministic"):
        gap = _rel_gap(local, paper)
        if gap > float(TOLERANCES["deterministic_relative_gap_fail"]):
            return "FAIL_MISMATCH"
        return "INDEPENDENT_PARTIAL_MATCH"
    if "FDR" in str(paper_row.get("metric") or ""):
        se = _mc_se(run, str(paper_row.get("metric") or "FDR"))
        alpha = paper_row.get("alpha")
        if alpha is None:
            cfg = run.get("config") if run and isinstance(run.get("config"), dict) else {}
            alpha = cfg.get("alpha") or cfg.get("delta")
        if alpha is not None and se is not None:
            if local > float(alpha) + 2.0 * se:
                return "FAIL_MISMATCH"
        elif alpha is not None and local > float(alpha):
            # Missing per-trial FDP: cannot accept FDR above alpha.
            return "FAIL_MISMATCH"
    # MC table cells with paper-scale local numbers. No official repo, so never
    # OFFICIAL_* and never invent FULL_MATCH for a partial alpha grid.
    return "INDEPENDENT_PARTIAL_MATCH"


def _local_power_ratio(
    paper_row: dict[str, Any], local_rows: list[dict[str, Any]]
) -> tuple[Any, dict[str, Any] | None]:
    method = str(paper_row.get("method") or "")
    family = str(paper_row.get("family") or "")
    baseline = "e-LORD" if family == "LORD" else "e-SAFFRON"
    method_run = _match_run({**paper_row, "metric": "Power(T)"}, local_rows)
    baseline_run = _match_run(
        {**paper_row, "method": baseline, "metric": "Power(T)"},
        local_rows,
    )
    method_p = _finite(_local_metric(method_run, "Power(T)")) if method_run else None
    base_p = _finite(_local_metric(baseline_run, "Power(T)")) if baseline_run else None
    if method_p is None or base_p is None or base_p == 0.0:
        return "NOT_RUN", method_run
    return method_p / base_p, method_run


def compare(raw_dir: Path) -> list[dict[str, Any]]:
    local_rows = _load_local_rows(raw_dir)
    out: list[dict[str, Any]] = []
    for paper_row in flatten_extractable_numbers():
        run = _match_run(paper_row, local_rows)
        local_value: Any = "NOT_RUN"
        if paper_row["kind"] in {"FIGURE_ONLY", "NOT_LOCATED"}:
            local_value = _figure_local_summary(paper_row, local_rows)
        elif paper_row.get("metric") == "power_ratio":
            local_value, run = _local_power_ratio(paper_row, local_rows)
        elif run is not None:
            local_value = _local_metric(run, str(paper_row["metric"]))
            if local_value in (None, ""):
                local_value = "NOT_RUN"
        if local_value == "NOT_RUN":
            stub = next(
                (
                    row
                    for row in local_rows
                    if paper_row["object_id"] in _run_blob(row)
                    and str(row.get("status") or "").startswith("BLOCKED")
                ),
                None,
            )
            if stub is not None:
                run = stub
                if paper_row["kind"] in {"FIGURE_ONLY", "NOT_LOCATED"}:
                    local_value = str(stub.get("status"))
        local_num = _finite(local_value)
        paper_num = _finite(paper_row["paper_value"])
        if paper_row["kind"] == "FIGURE_ONLY":
            gap: Any = "FIGURE_ONLY"
        elif paper_row["kind"] == "NOT_LOCATED":
            gap = "NOT_LOCATED_IN_PDF"
        elif local_num is None or paper_num is None:
            gap = "NOT_RUN" if local_value == "NOT_RUN" else str(local_value)
        else:
            gap = _rel_gap(local_num, paper_num)
        verdict = _verdict(paper_row, local_value, run)
        # Never emit OFFICIAL_* — no official repos.
        if str(verdict).startswith("OFFICIAL"):
            verdict = "NOT_RUN"
        out.append(
            {
                **paper_row,
                "local_value": local_value,
                "relative_gap": gap,
                "n_reps_local": _n_reps(run),
                "run_id": None if run is None else run.get("run_id"),
                "status": verdict,
            }
        )
    return out


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Paper vs reproduction (v0.1)",
        "",
        "Generated by `idea_C/scripts/verify_paper_results.py`.",
        "No fabricated matches. Missing local values are `NOT_RUN`, not 0.",
        "Official repos: none. Modes are independent only. UAI is not CCF-A. e-PS is preprint.",
        "",
        "## Frozen tolerances (written before results)",
        "",
        f"- Deterministic tables: relative gap > {100 * float(TOLERANCES['deterministic_relative_gap_fail']):.0f}% = `FAIL_MISMATCH` until explained.",
        f"- {TOLERANCES['curves_rule']}",
        f"- {TOLERANCES['fdr_rule']}",
        f"- {TOLERANCES['thirty_seed_rule']}",
        f"- {TOLERANCES['official_rule']}",
        "",
        "## Gap table",
        "",
        "|object_id|object|method|metric|paper|local|relative_gap|n_reps_local|status|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        paper = row["paper_value"]
        local = row["local_value"]
        gap = row["relative_gap"]
        if isinstance(gap, float):
            gap_s = f"{gap:.4f}"
        else:
            gap_s = str(gap)
        lines.append(
            "|"
            + "|".join(
                [
                    str(row["object_id"]),
                    str(row["object"]).replace("|", "/"),
                    str(row.get("method") or ""),
                    str(row.get("metric") or ""),
                    str(paper),
                    str(local),
                    gap_s,
                    str(row.get("n_reps_local") or "NOT_RUN"),
                    str(row["status"]),
                ]
            )
            + "|"
        )
    n_fig = sum(1 for r in rows if r.get("kind") == "FIGURE_ONLY")
    n_tab = sum(1 for r in rows if r.get("kind") == "table")
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    count_bits = ", ".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- checklist objects: {len(OBJECTS)}",
            f"- gap rows: {len(rows)} (table cells {n_tab}, figure-only {n_fig})",
            f"- status mix: {count_bits}",
            "- no OFFICIAL_* rows: official code was not found",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare extractable paper numbers vs local v0.1 CSV"
    )
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=None, help="optional markdown path")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="always print the gap table to stdout",
    )
    args = parser.parse_args(argv)
    rows = compare(args.raw)
    text = render_markdown(rows)
    out_path = args.out if args.out is not None else DEFAULT_OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Overwrite only the v0.1 skeleton/report, never v0 raw CSVs.
    out_path.write_text(text, encoding="utf-8")
    if args.stdout or args.out is None:
        sys.stdout.write(text)
    print(f"wrote {out_path} rows={len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
