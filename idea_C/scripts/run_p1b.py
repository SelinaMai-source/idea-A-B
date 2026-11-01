#!/usr/bin/env python3
"""P1-B streaming overlap runner. Same e-value stream; FDR and SupFDR separate.

Default: list and refuse. Pass --run. Never paper-scale 200/500.
Skip NYC Taxi / Jester / watermark / multi-agent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDEA_C = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = IDEA_C / "experiments" / "configs" / "overlap_streaming.json"
SCORE_CFG = IDEA_C / "configs" / "score_gaussian_mixture.json"
V01_RAW = IDEA_C / "results" / "v0.1" / "raw"

METHOD_VERSION = {
    "e-LOND": "v0_ob_elond",
    "Donation e-LOND": "v0_ob_donation_elond",
    "Closed e-LOND": "v0_ob_closed_elond",
    "e-LORD": "v0_ob_elord",
    "SCORE-LORD": "v0_ob_score_lord",
    "SCORE+-LORD": "v0_ob_scoreplus_lord",
}

# Methods whose papers primarily report FDR; SupFDR column may be N/A.
SUPFDR_NA = {"e-LORD", "SCORE-LORD", "SCORE+-LORD"}


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def list_config() -> int:
    cfg = load_config()
    print("P1-B streaming overlap (same e-value stream)")
    print(f"config: {CONFIG}")
    print(f"DGP borrow: {cfg['dgp']['borrow']} T={cfg['dgp']['T']} pi1={cfg['dgp']['pi1']}")
    print(f"methods: {cfg['methods']}")
    print("FDR and SupFDR are distinct columns; never mixed")
    print("Skipped: NYC Taxi / Jester / watermark / multi-agent")
    return 0


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-" else "-" for ch in text).strip("-")


def _finalize(directory: Path, *, seeds: list[int], log_text: str) -> None:
    (directory / "seeds.json").write_text(
        json.dumps({"seeds": seeds, "seed0": seeds[0] if seeds else None}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "run.log").write_text(log_text, encoding="utf-8")


def write_skip(version: str, content: str, data: str, reason: str) -> str:
    from idea_C.shared.serialize import write_run

    run_id = f"overlap_b_streaming_{_slug(version)}_NOT_RUN"
    directory = write_run(
        run_id,
        table_id="overlap_b_streaming",
        version=version,
        content=content,
        data=data,
        config={"skipped": True, "reason": reason},
        seed=0,
        metrics={},
        status="NOT_RUN",
        result_explanation=reason,
    )
    _finalize(directory, seeds=[], log_text=reason + "\n")
    return run_id


def run_and_write(
    *,
    reps: int,
    T: int,
    pi1: float,
    seed0: int,
    scale: str,
    assume_cpqd: bool,
    raw_dir: Path | None = None,
    version_prefix: str = "",
    run_id_prefix: str = "",
    skip_methods: set[str] | None = None,
) -> list[str]:
    from idea_C.baselines.eclosure import (
        closed_e_lond,
        donation_e_lond,
        e_lond,
        fdr_terminal,
        sup_fdp,
        average_power_terminal,
    )
    from idea_C.baselines.score import (
        e_lord,
        independent_gaussian_mixture,
        score_lord,
        score_plus_lord,
    )
    from idea_C.shared.serialize import write_run
    from idea_C.shared.stats import mean_std_ci

    cfg = load_config()
    score_cfg = json.loads(SCORE_CFG.read_text())
    seeds = [seed0 + r for r in range(reps)]
    alpha = float(cfg["dgp"].get("alpha", 0.05))
    data = f"P1-B streaming SCORE-like mix T={T} $\\pi_1={pi1}$ ADAPTED_PROTOCOL"
    skip_methods = skip_methods or set()
    print(
        f"Running P1-B streams reps={reps} T={T} pi1={pi1} seed0={seed0} "
        f"scale={scale} skip={sorted(skip_methods)}"
    )
    streams = [independent_gaussian_mixture(T=T, pi1=pi1, seed=s) for s in seeds]

    dispatch = {
        "e-LOND": lambda ev: e_lond(ev, delta=alpha),
        "Donation e-LOND": lambda ev: donation_e_lond(ev, delta=alpha),
        "Closed e-LOND": lambda ev: closed_e_lond(ev, delta=alpha),
        "e-LORD": lambda ev: e_lord(ev, alpha=alpha, omega=0.05),
        "SCORE-LORD": lambda ev: score_lord(ev, alpha=alpha, omega=0.05),
        "SCORE+-LORD": lambda ev: score_plus_lord(ev, alpha=alpha, omega=0.05, assume_cpqd=assume_cpqd),
    }
    run_ids: list[str] = []
    for name in list(cfg.get("methods") or dispatch):
        if name in skip_methods:
            print(f"  skip {name}: {skip_methods}")
            continue
        version = f"{METHOD_VERSION.get(name, 'v0_ob_' + _slug(name))}_{scale}"
        if version_prefix:
            if version.startswith("v0_"):
                version = f"{version_prefix}_{version[3:]}"
            elif not version.startswith(version_prefix):
                version = f"{version_prefix}_{version}"
        run_id = f"{run_id_prefix}overlap_b_streaming_{_slug(name)}_{scale}_T{T}_seed{seed0}"
        fdps: list[float] = []
        sups: list[float] = []
        powers: list[float] = []
        nrejs: list[float] = []
        log_lines = [
            f"run_id={run_id}",
            f"method={name}",
            f"scale={scale}",
            f"reps={reps}",
            f"seeds={seeds}",
            f"T={T}",
            f"pi1={pi1}",
            f"assume_cpqd={assume_cpqd}",
            "code_status=independent_reimplementation",
        ]
        t0 = time.perf_counter()
        status = "ok"
        note = (
            f"C-v0.1_RERUN P1-B {scale} {reps}-seed ADAPTED_PROTOCOL "
            f"independent_reimplementation same SCORE §6.1 stream T={T} π1={pi1} "
            f"assume_cpqd={assume_cpqd}. FDR and SupFDR are separate. "
            "Not paper-scale. Closed 30-seed skipped when tens of minutes."
        )
        metrics: dict = {}
        runtime = 0.0
        try:
            first = None
            for i, stream in enumerate(streams):
                result = dispatch[name](stream.evalues)
                first = result
                if getattr(result, "status", "ok") == "N/A":
                    status = "N/A"
                    note = getattr(result, "na_reason", None) or "SCORE+ N/A without CPQD"
                    metrics = {
                        "FDR↓": "N/A",
                        "SupFDR↓": "N/A",
                        "Power↑": "N/A",
                        "Rejections↑": "N/A",
                        "Runtime↓": "N/A",
                    }
                    break
                fdps.append(fdr_terminal(result.rejected, stream.is_null))
                if name not in SUPFDR_NA:
                    sups.append(sup_fdp(result.rejected, stream.is_null))
                powers.append(average_power_terminal(result.rejected, stream.is_null))
                nrejs.append(float(result.n_rejections))
                print(f"    {name} seed={seeds[i]} done")
            runtime = time.perf_counter() - t0
            if status == "ok":
                metrics = {
                    "FDR↓": mean_std_ci(fdps)["mean"],
                    "Power↑": mean_std_ci(powers)["mean"],
                    "Rejections↑": mean_std_ci(nrejs)["mean"],
                    "Runtime↓": runtime,
                }
                if name in SUPFDR_NA:
                    metrics["SupFDR↓"] = "N/A"
                else:
                    metrics["SupFDR↓"] = mean_std_ci(sups)["mean"]
            log_lines.append(f"status={status} wall_s={runtime:.6f} metrics={metrics}")
            print(f"  {name}: status={status} metrics={metrics} runtime={runtime:.3f}s")
        except Exception as exc:
            runtime = time.perf_counter() - t0
            status = "failed"
            note = f"失败：{type(exc).__name__}: {exc}"
            log_lines.append(traceback.format_exc())
            print(f"  {name}: FAILED {exc}")
            metrics = {}

        directory = write_run(
            run_id,
            table_id="overlap_b_streaming",
            version=version,
            content=f"{name} P1-B {scale} {reps}-seed ADAPTED_PROTOCOL C-v0.1_RERUN independent_reimplementation",
            data=data,
            config={
                **cfg,
                "method": name,
                "scale": scale,
                "reps": reps,
                "T": T,
                "pi1": pi1,
                "seed0": seed0,
                "seeds": seeds,
                "assume_cpqd": assume_cpqd,
                "score_cfg_assume_cpqd": score_cfg.get("assume_cpqd_for_score_plus"),
                "protocol": "ADAPTED_PROTOCOL",
                "label": "C-v0.1_RERUN",
            },
            seed=seed0,
            metrics=metrics,
            status=status,
            result_explanation=note,
            raw_dir=raw_dir,
            extra={
                "seeds": seeds,
                "wall_s": runtime,
                "fdps": fdps,
                "sup_fdps": sups,
                "protocol": "ADAPTED_PROTOCOL",
            },
        )
        _finalize(directory, seeds=seeds, log_text="\n".join(log_lines) + "\n")
        run_ids.append(run_id)
    return run_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1-B streaming overlap runner")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--scale", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--pi1", type=float, default=0.3)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--assume-cpqd", action="store_true")
    parser.add_argument("--no-assume-cpqd", action="store_true")
    parser.add_argument(
        "--v01",
        action="store_true",
        help="write C-v0.1_RERUN rows under results/v0.1/raw (never overwrite v0)",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--skip-closed", action="store_true")
    args = parser.parse_args(argv)

    if args.list or not args.run:
        return list_config()
    if args.paper_scale:
        print("P1-B refuses --paper-scale (not a paper table).", file=sys.stderr)
        return 2

    cfg = load_config()
    score_cfg = json.loads(SCORE_CFG.read_text())
    if args.reps is None:
        args.reps = 30 if args.scale == "pilot" else 3
    if args.reps >= 200:
        print("Refusing paper-scale-sized P1-B.", file=sys.stderr)
        return 2
    T = args.T if args.T is not None else int(cfg["dgp"]["T"])
    assume = bool(score_cfg.get("assume_cpqd_for_score_plus", False))
    if args.no_assume_cpqd:
        assume = False
    if args.assume_cpqd:
        assume = True
    raw_dir = args.raw_dir or (V01_RAW if args.v01 else None)
    version_prefix = "C-v0.1_RERUN" if args.v01 else ""
    run_id_prefix = "v01_p1b_" if args.v01 else ""
    skip_methods = {"Closed e-LOND"} if args.skip_closed else set()
    written = run_and_write(
        reps=args.reps,
        T=T,
        pi1=args.pi1,
        seed0=args.seed0,
        scale=args.scale,
        assume_cpqd=assume,
        raw_dir=raw_dir,
        version_prefix=version_prefix,
        run_id_prefix=run_id_prefix,
        skip_methods=skip_methods,
    )
    print("wrote run_ids:")
    for rid in written:
        print(f"  {rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
