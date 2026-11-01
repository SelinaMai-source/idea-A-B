#!/usr/bin/env python3
"""Idea C v0.1 unified reproduction CLI.

Wires existing paper runners (run_aaai / run_eps / run_score / run_uai)
instead of reimplementing algorithms. Writes checkpoints under
``idea_C/results/v0.1/raw/`` and does **not** touch v0 raw CSVs.

Default: list objects and refuse Monte Carlo.
``--scale paper`` is refused unless ``--paper-scale`` is also given.
``--mode official`` always errors: there are no official repos.
UAI is not CCF-A Main. e-PS is a preprint. 30-seed is pilot, never
full reproduction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
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
    PAPERS,
    TOLERANCES,
    counts_by_paper,
    objects_for,
)

RAW_DIR = IDEA_C / "results" / "v0.1" / "raw"
SUMMARIES = IDEA_C / "results" / "v0.1" / "summaries"
FIGURES = IDEA_C / "results" / "v0.1" / "figures"
OFFICIAL_AUDIT = IDEA_C / "reports" / "v0.1" / "OFFICIAL_CODE_AUDIT.md"

SCALES = ("smoke", "pilot", "paper")
PAPER_REPS = {"aaai": 500, "score": 500, "uai": 200, "eps": 500}

OFFICIAL_ERROR = (
    "Official mode is unavailable: no official repositories for any of the "
    "four papers (see idea_C/reports/v0.1/OFFICIAL_CODE_AUDIT.md). "
    "Modes are independent only. UAI is not CCF-A Main. e-PS is a preprint."
)

PAPER_SCALE_ERROR = (
    "Refusing --scale paper. Re-run with explicit --paper-scale if you really "
    "want the paper Monte Carlo budget. 30-seed is pilot, never paper-scale."
)


def _load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rss_mb() -> float | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return None


def _ensure_dirs() -> None:
    for path in (RAW_DIR, SUMMARIES, FIGURES):
        path.mkdir(parents=True, exist_ok=True)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in text).strip("-")


def _v01_versions(mapping: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in mapping.items():
        if value.startswith("v0_"):
            out[key] = "C-v0.1_" + value[3:]
        elif value.startswith("C-v0.1_"):
            out[key] = value
        else:
            out[key] = f"C-v0.1_{value}"
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _checkpoint_done(run_id: str) -> bool:
    metrics = RAW_DIR / run_id / "metrics.json"
    if not metrics.exists():
        return False
    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    status = str(data.get("status") or "")
    return status not in {"failed", ""}


def _write_stub(
    obj: dict[str, Any],
    *,
    scale: str,
    seed: int,
    status: str,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> str:
    from idea_C.shared.serialize import write_run

    run_id = f"v01_{_slug(obj['id'])}_{scale}_seed{seed}"
    t0 = time.perf_counter()
    rss0 = _rss_mb()
    directory = write_run(
        run_id,
        table_id=str(obj.get("benchmark") or obj["id"]),
        version=f"C-v0.1_INDEPENDENT_{_slug(obj['id'])}_{scale}",
        content=f"{obj['object']} {scale} stub independent_reimplementation",
        data=str(obj.get("dgp") or obj["id"]),
        config={
            "object_id": obj["id"],
            "scale": scale,
            "seed": seed,
            "wired": bool(obj.get("wired")),
            "reason": reason,
            "venue": obj.get("venue"),
        },
        seed=seed,
        metrics={},
        status=status,
        result_explanation=reason,
        raw_dir=RAW_DIR,
        extra=extra or {},
    )
    runtime = time.perf_counter() - t0
    log = [
        f"run_id={run_id}",
        f"object_id={obj['id']}",
        f"status={status}",
        f"scale={scale}",
        f"seed={seed}",
        f"wired={obj.get('wired')}",
        f"wall_s={runtime:.6f}",
        f"rss_mb={_rss_mb()}",
        f"rss_mb_before={rss0}",
        reason,
    ]
    (directory / "run.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    _write_json(
        directory / "runtime.json",
        {"wall_s": runtime, "rss_mb": _rss_mb(), "status": status},
    )
    return run_id


def _patch_raw_dir():
    import idea_C.shared.serialize as serialize

    previous = serialize.DEFAULT_RAW
    serialize.DEFAULT_RAW = RAW_DIR
    return serialize, previous


def _call_aaai(obj: dict[str, Any], args: argparse.Namespace) -> list[str]:
    run_aaai = _load_script("run_aaai")
    cfg_name = str(obj.get("config") or "aaai_robust_gaussian.json")
    cfg = run_aaai.load_config(cfg_name)
    methods = list(obj.get("methods") or ["eprocess_based"])
    scale = args.scale
    written: list[str] = []
    if not args.run:
        print(
            f"AAAI plan object={obj['id']} config={cfg_name} methods={methods} "
            f"scale={scale} trials={cfg['n_trials'][scale]} "
            f"horizon={cfg['horizon'][scale]} code_status=independent_reimplementation"
        )
        print("No --run: refusing to start Monte Carlo.")
        return written
    from idea_C.shared.serialize import write_run

    for method in methods:
        if method not in {"uniform", "mean_based", "eprocess_based", "algo3_heavytail", "nonrobust_stress"}:
            written.append(
                _write_stub(
                    obj,
                    scale=scale,
                    seed=args.seed,
                    status="NOT_RUN",
                    reason=f"method {method!r} is not in run_aaai.py choices",
                )
            )
            continue
        run_id = f"v01_{_slug(obj['id'])}_{_slug(method)}_{scale}_seed{args.seed}"
        if args.resume and _checkpoint_done(run_id):
            print(f"resume skip {run_id}")
            written.append(run_id)
            continue
        t0 = time.perf_counter()
        rss0 = _rss_mb()
        status = "ok"
        metrics: dict[str, Any] = {}
        note = (
            f"C-v0.1_INDEPENDENT {obj['id']} {method} {scale} "
            "INDEPENDENT_REIMPLEMENTATION FIGURE_ONLY"
        )
        log_lines = [f"run_id={run_id}", f"method={method}", f"object={obj['id']}"]
        try:
            metrics = run_aaai.run_trials(
                cfg, method=method, scale=scale, seed0=args.seed, table_id=str(cfg["table_id"])
            )
            if metrics.get("protocol_status") == "INVALID_PROTOCOL":
                status = "INVALID_PROTOCOL"
                note = (
                    f"INVALID_PROTOCOL: {method} never left Algorithm 2 uniform burn-in "
                    f"at horizon={metrics.get('horizon')}. Do not rank this row. "
                    "INDEPENDENT_REIMPLEMENTATION"
                )
            else:
                sat = metrics.get("samples_at_target") or {}
                note = (
                    f"C-v0.1_INDEPENDENT {obj['id']} {method} {scale} "
                    f"n_reps={metrics.get('n_trials')} horizon={metrics.get('horizon')} "
                    f"post_burnin={metrics.get('entered_adaptive_phase')} "
                    f"Samples@Target={metrics.get('Samples@Target TPR↓')} "
                    f"success_rate={sat.get('success_rate')} "
                    f"right_censoring_rate={sat.get('right_censoring_rate')} "
                    "Paper=FIGURE_ONLY Our value in metrics; gap=FIGURE_ONLY. "
                    "INDEPENDENT_REIMPLEMENTATION Lévy α/scale unspecified_in_paper=1.5/1.0."
                )
        except Exception as exc:
            status = "failed"
            note = f"失败：{type(exc).__name__}: {exc}"
            log_lines.append(traceback.format_exc())
            print(f"  {method}: FAILED {exc}")
        runtime = time.perf_counter() - t0
        directory = write_run(
            run_id,
            table_id=str(cfg["table_id"]),
            version=f"C-v0.1_INDEPENDENT_{_slug(method)}_{scale}",
            content=f"{method} {scale} independent_reimplementation",
            data=str(obj.get("dgp") or cfg["table_id"]),
            config={**cfg, "method": method, "scale": scale, "seed0": args.seed, "object_id": obj["id"]},
            seed=args.seed,
            metrics={k: v for k, v in metrics.items() if k in ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓")},
            status=status,
            result_explanation=note,
            raw_dir=RAW_DIR,
            extra={
                "wall_s": runtime,
                "rss_mb": _rss_mb(),
                "rss_mb_before": rss0,
                "object_id": obj["id"],
                "protocol_status": metrics.get("protocol_status"),
                "samples_at_target": metrics.get("samples_at_target"),
                "post_burnin": metrics.get("post_burnin"),
                "entered_adaptive_phase": metrics.get("entered_adaptive_phase"),
            },
        )
        log_lines.extend(
            [
                f"status={status}",
                f"wall_s={runtime:.6f}",
                f"rss_mb={_rss_mb()}",
                f"metrics={metrics}",
            ]
        )
        (directory / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        _write_json(directory / "runtime.json", {"wall_s": runtime, "rss_mb": _rss_mb(), "status": status})
        written.append(run_id)
        print(f"wrote {directory}")
    return written


def _call_eps(obj: dict[str, Any], args: argparse.Namespace) -> list[str]:
    run_eps = _load_script("run_eps")
    cfg_name = str(obj.get("config") or "eps_gaussian.json")
    cfg = run_eps.load_config(cfg_name)
    family = "student_t" if "student" in str(cfg.get("table_id", "")) else "gaussian"
    samplers = list(obj.get("methods") or ["eps"])
    eprocesses = list(obj.get("eprocess") or ["oracle", "adaptive"])
    scale = args.scale
    written: list[str] = []
    if not args.run:
        print(
            f"e-PS plan object={obj['id']} config={cfg_name} samplers={samplers} "
            f"eprocess={eprocesses} scale={scale} venue=preprint_not_CCF-A "
            f"code_status=independent_reimplementation"
        )
        print("No --run: refusing to start Monte Carlo.")
        return written
    from idea_C.shared.serialize import write_run

    for sampler in samplers:
        if sampler not in {"uniform", "greedy", "eps"}:
            written.append(
                _write_stub(
                    obj,
                    scale=scale,
                    seed=args.seed,
                    status="NOT_RUN",
                    reason=f"sampler {sampler!r} is not in run_eps.py",
                )
            )
            continue
        for eprocess in eprocesses:
            run_id = (
                f"v01_{_slug(obj['id'])}_{sampler}_{eprocess}_{scale}_seed{args.seed}"
            )
            if args.resume and _checkpoint_done(run_id):
                print(f"resume skip {run_id}")
                written.append(run_id)
                continue
            t0 = time.perf_counter()
            rss0 = _rss_mb()
            status = "ok"
            metrics: dict[str, Any] = {}
            note = (
                f"C-v0.1_INDEPENDENT {obj['id']} {sampler}/{eprocess} {scale} "
                "INDEPENDENT_REIMPLEMENTATION preprint_not_CCF-A FIGURE_ONLY"
            )
            log_lines = [f"run_id={run_id}", f"sampler={sampler}", f"eprocess={eprocess}"]
            try:
                metrics = run_eps.run_trials(
                    cfg,
                    sampler=sampler,
                    eprocess=eprocess,
                    scale=scale,
                    seed0=args.seed,
                    family=family,
                )
                sat = metrics.get("samples_at_target") or {}
                note = (
                    f"C-v0.1_INDEPENDENT {obj['id']} {sampler}/{eprocess} {scale} "
                    f"n_reps={metrics.get('n_trials')} horizon={metrics.get('horizon')} "
                    f"θ_k=0.02k·B_k (NOT μ=1) Samples@Target={metrics.get('Samples@Target TPR↓')} "
                    f"success_rate={sat.get('success_rate')} "
                    f"right_censoring_rate={sat.get('right_censoring_rate')} "
                    "Paper=FIGURE_ONLY; e-PS is a preprint, not published CCF-A. "
                    "INDEPENDENT_REIMPLEMENTATION"
                )
            except Exception as exc:
                status = "failed"
                note = f"失败：{type(exc).__name__}: {exc}"
                log_lines.append(traceback.format_exc())
                print(f"  {sampler}/{eprocess}: FAILED {exc}")
            runtime = time.perf_counter() - t0
            directory = write_run(
                run_id,
                table_id=str(cfg["table_id"]),
                version=f"C-v0.1_INDEPENDENT_{sampler}_{eprocess}_{scale}",
                content=f"{sampler} + {eprocess} {scale} independent_reimplementation",
                data=str(obj.get("dgp") or cfg["table_id"]),
                config={
                    **cfg,
                    "sampler": sampler,
                    "eprocess": eprocess,
                    "scale": scale,
                    "seed0": args.seed,
                    "object_id": obj["id"],
                },
                seed=args.seed,
                metrics={
                    k: v
                    for k, v in metrics.items()
                    if k in ("FDR↓", "TPR↑", "TPP↑", "Samples@Target TPR↓", "Runtime↓")
                },
                status=status,
                result_explanation=note,
                raw_dir=RAW_DIR,
                extra={
                    "wall_s": runtime,
                    "rss_mb": _rss_mb(),
                    "rss_mb_before": rss0,
                    "samples_at_target": metrics.get("samples_at_target"),
                    "object_id": obj["id"],
                },
            )
            log_lines.extend(
                [f"status={status}", f"wall_s={runtime:.6f}", f"rss_mb={_rss_mb()}"]
            )
            (directory / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            _write_json(
                directory / "runtime.json",
                {"wall_s": runtime, "rss_mb": _rss_mb(), "status": status},
            )
            written.append(run_id)
            print(f"wrote {directory}")
    return written


def _call_score(obj: dict[str, Any], args: argparse.Namespace) -> list[str]:
    run_score = _load_script("run_score")
    run_score.METHOD_VERSION = _v01_versions(run_score.METHOD_VERSION)
    cfg = run_score.load_config()
    v01_cfg = IDEA_C / "configs" / "score_gaussian_mixture_v0_1.json"
    if v01_cfg.exists():
        cfg = json.loads(v01_cfg.read_text(encoding="utf-8"))
    methods = list(obj.get("methods") or cfg.get("methods") or [])
    original_load = run_score.load_config

    def _filtered_load() -> dict:
        loaded = json.loads(v01_cfg.read_text(encoding="utf-8")) if v01_cfg.exists() else original_load()
        loaded["methods"] = methods
        return loaded

    run_score.load_config = _filtered_load
    scale = args.scale
    reps = args.reps
    if reps is None:
        reps = int(cfg["n_reps_paper"] if scale == "paper" else (30 if scale == "pilot" else cfg.get("n_reps_default", 3)))
    T = int(cfg["dgp"]["T"])
    pi1_list = list(obj.get("pi1") or cfg["dgp"]["pi1"])
    written: list[str] = []
    if not args.run:
        print(
            f"SCORE plan object={obj['id']} methods={methods} scale={scale} "
            f"reps={reps} T={T} pi1={pi1_list} "
            f"code_status=independent_reimplementation"
        )
        print("No --run: refusing to start Monte Carlo.")
        return written
    serialize, previous = _patch_raw_dir()
    try:
        for pi1 in pi1_list:
            run_ids = run_score.run_and_write(
                reps=int(reps),
                T=T,
                pi1=float(pi1),
                seed0=args.seed,
                assume_cpqd=bool(cfg.get("assume_cpqd_for_score_plus", True)),
                scale=scale,
                reduced_horizon=False,
            )
            written.extend(run_ids)
    finally:
        serialize.DEFAULT_RAW = previous
        run_score.load_config = original_load
    return written


def _call_uai(obj: dict[str, Any], args: argparse.Namespace) -> list[str]:
    run_uai = _load_script("run_uai")
    run_uai.METHOD_VERSION = _v01_versions(run_uai.METHOD_VERSION)
    cfg = run_uai.load_config()
    methods = list(obj.get("methods") or cfg.get("methods") or [])
    original_load = run_uai.load_config

    def _filtered_load() -> dict:
        loaded = original_load()
        loaded["methods"] = [m for m in methods if m in (loaded.get("methods") or methods)]
        if not loaded["methods"]:
            loaded["methods"] = list(cfg.get("methods") or [])
        return loaded

    run_uai.load_config = _filtered_load
    scale = args.scale
    reps = args.reps
    if reps is None:
        reps = int(cfg["n_trials_paper"] if scale == "paper" else (30 if scale == "pilot" else cfg.get("n_reps_default", 3)))
    m = int(cfg["dgp"]["m"])
    if obj["id"] == "uai.fig2_runtime" and scale == "paper":
        m = 3000
        reps = 80
    written: list[str] = []
    if not args.run:
        print(
            f"UAI plan object={obj['id']} methods={methods} scale={scale} "
            f"reps={reps} m={m} venue=UAI_NOT_CCF-A "
            f"code_status=independent_reimplementation"
        )
        print("No --run: refusing to start Monte Carlo.")
        return written
    serialize, previous = _patch_raw_dir()
    try:
        run_ids = run_uai.run_and_write(
            reps=int(reps),
            m=m,
            pi1=0.3,
            seed0=args.seed,
            scale=scale,
            reduced=m != int(cfg["dgp"]["m"]),
        )
        written.extend(run_ids)
    finally:
        serialize.DEFAULT_RAW = previous
        run_uai.load_config = original_load
    return written


def _dispatch(obj: dict[str, Any], args: argparse.Namespace) -> list[str]:
    if obj.get("block_on_run") and args.run:
        return [
            _write_stub(
                obj,
                scale=args.scale,
                seed=args.seed,
                status=str(obj["block_on_run"]),
                reason=str(obj.get("note") or obj["block_on_run"]),
            )
        ]
    if obj.get("numeric_kind") == "NOT_LOCATED":
        return [
            _write_stub(
                obj,
                scale=args.scale,
                seed=args.seed,
                status="NOT_RUN",
                reason="NOT_LOCATED_IN_PDF: AAAI appendix tables were not readable.",
            )
        ]
    if not obj.get("wired"):
        if not args.run:
            print(
                f"STUB plan object={obj['id']} wired=False script={obj.get('script')} "
                f"status=NOT_RUN (no Monte Carlo)"
            )
            return []
        return [
            _write_stub(
                obj,
                scale=args.scale,
                seed=args.seed,
                status="NOT_RUN",
                reason=(
                    obj.get("note")
                    or "No dedicated runner wired yet; checkpoint kept as NOT_RUN."
                ),
            )
        ]
    runner = obj.get("runner")
    if runner == "aaai":
        return _call_aaai(obj, args)
    if runner == "eps":
        return _call_eps(obj, args)
    if runner == "score":
        return _call_score(obj, args)
    if runner == "uai":
        return _call_uai(obj, args)
    return [
        _write_stub(
            obj,
            scale=args.scale,
            seed=args.seed,
            status="NOT_RUN",
            reason=f"unknown runner {runner!r}",
        )
    ]


def _call_updater() -> int:
    updater = _load_script("update_v0_results_md")
    seen = SUMMARIES / "ingested_run_ids.json"
    md = IDEA_C / "v0实验版本+结果.md"
    print(f"calling updater --raw {RAW_DIR} --seen {seen}")
    return updater.update_markdown(md, RAW_DIR, seen)


def list_objects(selected: list[dict[str, Any]]) -> int:
    counts = counts_by_paper()
    print("Idea C v0.1 FULL_REPRODUCTION_CHECKLIST")
    print("official_code=none; mode=independent only; UAI=NOT CCF-A; e-PS=preprint")
    print(
        "status vocabulary:",
        ", ".join(
            [
                "OFFICIAL_FULL_MATCH",
                "OFFICIAL_PARTIAL_MATCH",
                "INDEPENDENT_FULL_MATCH",
                "INDEPENDENT_PARTIAL_MATCH",
                "FAIL_MISMATCH",
                "BLOCKED_NO_CODE",
                "BLOCKED_NO_DATA",
                "BLOCKED_COMPUTE",
                "NOT_RUN",
                "INVALID_PROTOCOL",
            ]
        ),
    )
    print("tolerances (frozen, before results):")
    print(f"  deterministic relative gap > {100 * TOLERANCES['deterministic_relative_gap_fail']:.0f}% => FAIL_MISMATCH")
    print("  curves without official CSV => FIGURE_ONLY")
    print("  FDR must not exceed alpha + 2 MC SE")
    print("  30-seed cannot be full reproduction")
    print(f"raw dir: {RAW_DIR} exists={RAW_DIR.exists()}")
    print(
        "counts: "
        + " ".join(f"{paper}={counts[paper]}" for paper in PAPERS)
        + f" total={len(OBJECTS)}"
    )
    print(f"{'ID':<28} {'OBJECT':<36} {'WIRED':<5} {'REPS':<8} {'STATUS'}")
    for obj in selected:
        reps = obj.get("mc_reps")
        print(
            f"{obj['id']:<28} {str(obj['object'])[:36]:<36} "
            f"{str(bool(obj.get('wired'))):<5} {str(reps):<8} {obj.get('status', 'NOT_RUN')}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_v0_1.py",
        description=(
            "Idea C v0.1 unified runner. Independent reimplementation only. "
            "No official repos. UAI is not CCF-A. e-PS is a preprint. "
            "Default refuses Monte Carlo and refuses --scale paper."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python idea_C/scripts/run_v0_1.py --help
  python idea_C/scripts/run_v0_1.py --list
  python idea_C/scripts/run_v0_1.py --paper score --figure 1 --scale smoke
  python idea_C/scripts/run_v0_1.py --paper aaai --benchmark aaai_robust_gaussian --scale smoke --run
  python idea_C/scripts/run_v0_1.py --scale paper
      # refused unless also --paper-scale
  python idea_C/scripts/run_v0_1.py --mode official
      # always errors: no official code
""",
    )
    parser.add_argument("--list", action="store_true", help="list checklist objects and exit")
    parser.add_argument("--paper", choices=list(PAPERS), default=None)
    parser.add_argument("--figure", default=None, help="figure number, e.g. 1 or 2a")
    parser.add_argument("--table", default=None, help="table number, e.g. 1")
    parser.add_argument("--benchmark", default=None, help="benchmark id from the checklist")
    parser.add_argument("--object", dest="object_id", default=None, help="checklist object id")
    parser.add_argument("--scale", choices=list(SCALES), default="smoke")
    parser.add_argument(
        "--paper-scale",
        action="store_true",
        help="required together with --scale paper; 30-seed is never paper-scale",
    )
    parser.add_argument("--reps", type=int, default=None, help="override Monte Carlo repeats")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("independent", "official"),
        default="independent",
        help="official always refuses: no official repos exist",
    )
    parser.add_argument("--resume", action="store_true", help="skip completed checkpoints under results/v0.1/raw")
    parser.add_argument("--run", action="store_true", help="actually execute (default is list/plan only)")
    parser.add_argument(
        "--no-updater",
        action="store_true",
        help="do not call update_v0_results_md.py after a --run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _ensure_dirs()

    if args.mode == "official":
        print(OFFICIAL_ERROR, file=sys.stderr)
        print(f"audit: {OFFICIAL_AUDIT}", file=sys.stderr)
        return 2

    if args.scale == "paper" and not args.paper_scale:
        print(PAPER_SCALE_ERROR, file=sys.stderr)
        return 2

    if args.paper_scale and args.scale != "paper":
        args.scale = "paper"

    if args.reps is not None and args.scale != "paper":
        paper_budget = max(PAPER_REPS.values())
        if args.reps >= 200 and not args.paper_scale:
            print(
                f"Refusing reps={args.reps} without --paper-scale "
                f"(paper budgets are 200/500, max listed {paper_budget}).",
                file=sys.stderr,
            )
            return 2

    selected = objects_for(
        paper=args.paper,
        figure=args.figure,
        table=args.table,
        benchmark=args.benchmark,
        object_id=args.object_id,
    )
    if args.figure or args.table or args.benchmark or args.object_id or args.paper:
        if not selected:
            print("No checklist objects matched the filters.", file=sys.stderr)
            return 2

    if args.list or not args.run:
        rc = list_objects(selected if selected else OBJECTS)
        if not args.run:
            return rc

    written: list[str] = []
    failed = False
    targets = selected if selected else OBJECTS
    print(f"dispatching {len(targets)} object(s) scale={args.scale} seed={args.seed}")
    for obj in targets:
        try:
            written.extend(_dispatch(obj, args))
        except Exception as exc:
            failed = True
            rid = _write_stub(
                obj,
                scale=args.scale,
                seed=args.seed,
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
                extra={"traceback": traceback.format_exc()},
            )
            written.append(rid)
            print(f"kept failed run {rid}", file=sys.stderr)

    print("wrote run_ids:")
    for rid in written:
        print(f"  {RAW_DIR / rid}")

    if not args.no_updater:
        updater_rc = _call_updater()
        if updater_rc != 0:
            return updater_rc
    return 1 if failed else 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    raise SystemExit(main())
