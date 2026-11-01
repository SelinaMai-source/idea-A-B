#!/usr/bin/env python3
"""SCORE §6.1 runner. Does not start paper-scale Monte Carlo unless asked.

Default: list config and refuse. Pass --run for a real simulation.
500-rep paper-scale is refused without --paper-scale.
30-seed is labeled pilot, never paper-scale.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "score_gaussian_mixture.json"
PAPER_REPS = 500

METHOD_VERSION = {
    "e-LOND": "v0_elond",
    "SCORE-LOND": "v0_score_lond",
    "e-LORD": "v0_elord",
    "SCORE-LORD": "v0_score_lord",
    "e-SAFFRON": "v0_esaffron",
    "SCORE-SAFFRON": "v0_score_saffron",
    "SCORE+-LORD": "v0_scoreplus_lord",
    "SCORE+-SAFFRON": "v0_scoreplus_saffron",
}


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def list_config() -> int:
    cfg = load_config()
    print("SCORE ICML 2026 independent_reimplementation (no official code)")
    print(f"config: {CONFIG}")
    print(f"DGP: {cfg['dgp']['name']} T={cfg['dgp']['T']} pi1={cfg['dgp']['pi1']}")
    print(f"paper reps: {cfg['n_reps_paper']} (refused unless --paper-scale)")
    print(f"methods: {cfg['methods']}")
    print(f"SCORE+ CPQD flag in config: {cfg['assume_cpqd_for_score_plus']}")
    print("Power: AP(t)=expected fraction of arrived non-nulls rejected; ratio vs family baseline.")
    return 0


def refuse_paper_scale(reps: int, paper_scale: bool) -> str | None:
    if reps >= PAPER_REPS and not paper_scale:
        return (
            f"Refusing paper-scale SCORE run (reps={reps} ≥ {PAPER_REPS}). "
            "Pass --paper-scale only for a true 500-rep experiment; "
            "do not use 3/30-seed pilots as paper-scale."
        )
    return None


def _slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch)
        elif ch in {".", "-"}:
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-")


def _finalize(directory: Path, *, seeds: list[int], log_text: str) -> None:
    (directory / "seeds.json").write_text(
        json.dumps({"seeds": seeds, "seed0": seeds[0] if seeds else None}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "run.log").write_text(log_text, encoding="utf-8")


def run_and_write(
    *,
    reps: int,
    T: int,
    pi1: float,
    seed0: int,
    assume_cpqd: bool,
    scale: str,
    reduced_horizon: bool,
) -> list[str]:
    from idea_C.baselines.score import (
        DISPATCH,
        average_power_terminal,
        fdr_terminal,
        independent_gaussian_mixture,
        run_named,
    )
    from idea_C.shared.serialize import write_run
    from idea_C.shared.stats import mean_std_ci

    cfg = load_config()
    seeds = [seed0 + r for r in range(reps)]
    protocol = "reduced-horizon / NOT paper protocol" if reduced_horizon else "paper T=1000 protocol"
    data = (
        f"SCORE §6.1 $\\pi_1={pi1}$ $T={T}$"
        + (" reduced-horizon (NOT paper protocol)" if reduced_horizon else "")
    )
    print(
        f"Running SCORE streams reps={reps} T={T} pi1={pi1} seed0={seed0} "
        f"scale={scale} assume_cpqd={assume_cpqd} protocol={protocol}"
    )
    t_gen = time.perf_counter()
    streams = [independent_gaussian_mixture(T=T, pi1=pi1, seed=s) for s in seeds]
    gen_s = time.perf_counter() - t_gen
    print(f"  DGP generated in {gen_s:.3f}s")

    run_ids: list[str] = []
    methods = list(cfg.get("methods") or DISPATCH)
    for name in methods:
        version = f"{METHOD_VERSION.get(name, 'v0_' + _slug(name))}_{scale}_pi{_slug(str(pi1))}"
        if reduced_horizon:
            version = f"{version}_T{T}"
        run_id = f"score_gaussian_mixture_{_slug(name)}_{scale}_pi{_slug(str(pi1))}_T{T}_seed{seed0}"
        kwargs: dict = {"alpha": float(cfg.get("alpha", 0.05))}
        if name in {"e-LORD", "SCORE-LORD", "SCORE+-LORD"}:
            kwargs["omega"] = float(cfg.get("lord_omega_t", 0.05))
        if name in {"e-SAFFRON", "SCORE-SAFFRON", "SCORE+-SAFFRON"}:
            kwargs["omega"] = float(cfg.get("lord_omega_t", 0.05))
            kwargs["lam"] = float(cfg.get("saffron_lambda_t", 0.5))
        if name.startswith("SCORE+"):
            kwargs["assume_cpqd"] = assume_cpqd
        fdps: list[float] = []
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
            f"code_status=independent_reimplementation",
            f"protocol={protocol}",
        ]
        t0 = time.perf_counter()
        status = "ok"
        note = (
            f"自动追加，{scale} {reps}-seed，independent_reimplementation，"
            f"{protocol}，结果解释待人工填写"
        )
        metrics: dict = {}
        try:
            first = None
            for stream in streams:
                result = run_named(name, stream.evalues, **kwargs)
                first = result
                if result.status == "N/A":
                    status = "N/A"
                    note = (
                        f"SCORE+ N/A：assume_cpqd={assume_cpqd}。"
                        f"{result.na_reason or 'CPQD required'}"
                    )
                    metrics = {
                        "FDR↓": "N/A",
                        "Power↑": "N/A",
                        "Rejections↑": "N/A",
                        "Runtime↓": "N/A",
                    }
                    break
                fdps.append(fdr_terminal(result.rejected, stream.is_null))
                powers.append(average_power_terminal(result.rejected, stream.is_null))
                nrejs.append(float(result.n_rejections))
            runtime = time.perf_counter() - t0
            if status != "N/A":
                metrics = {
                    "FDR↓": mean_std_ci(fdps)["mean"],
                    "Power↑": mean_std_ci(powers)["mean"],
                    "Rejections↑": mean_std_ci(nrejs)["mean"],
                    "Runtime↓": runtime,
                }
            log_lines.append(f"status={status}")
            log_lines.append(f"wall_s={runtime:.6f}")
            log_lines.append(f"fdps={fdps}")
            log_lines.append(f"powers={powers}")
            log_lines.append(f"nrejs={nrejs}")
            log_lines.append(f"metrics={metrics}")
            if first is not None:
                log_lines.append(f"method_status={first.status}")
            print(
                f"  {name}: status={status} FDR={metrics.get('FDR↓')} "
                f"Power={metrics.get('Power↑')} Rej={metrics.get('Rejections↑')} "
                f"runtime={runtime:.3f}s"
            )
        except Exception as exc:
            runtime = time.perf_counter() - t0
            status = "failed"
            note = f"失败：{type(exc).__name__}: {exc}"
            log_lines.append("FAILED")
            log_lines.append(traceback.format_exc())
            print(f"  {name}: FAILED {exc}")
            metrics = {}

        directory = write_run(
            run_id,
            table_id="score_gaussian_mixture",
            version=version,
            content=f"{name} {scale} {reps}-seed independent_reimplementation",
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
                "reduced_horizon": reduced_horizon,
                "protocol": protocol,
            },
            seed=seed0,
            metrics=metrics,
            status=status,
            result_explanation=note,
            extra={
                "seeds": seeds,
                "n_reps": reps,
                "scale": scale,
                "wall_s": runtime,
                "assume_cpqd": assume_cpqd,
                "reduced_horizon": reduced_horizon,
                "fdps": fdps,
                "powers": powers,
                "n_rejections": nrejs,
            },
        )
        _finalize(directory, seeds=seeds, log_text="\n".join(log_lines) + "\n")
        run_ids.append(run_id)
    return run_ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SCORE v0 runner")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true", help="actually simulate (default is list-only)")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--scale", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--T", type=int, default=None, help="override horizon; default from config (1000)")
    parser.add_argument("--pi1", type=float, action="append", default=None)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--paper-scale", action="store_true")
    parser.add_argument("--assume-cpqd", action="store_true", default=False)
    parser.add_argument("--no-assume-cpqd", action="store_true")
    args = parser.parse_args(argv)

    if args.list or not args.run:
        return list_config()

    cfg = load_config()
    if args.reps is None:
        args.reps = 30 if args.scale == "pilot" else int(cfg.get("n_reps_default", 3))
    if args.reps >= 30 and args.scale == "smoke":
        args.scale = "pilot"

    msg = refuse_paper_scale(args.reps, args.paper_scale)
    if msg:
        print(msg, file=sys.stderr)
        return 2

    T = args.T if args.T is not None else int(cfg["dgp"]["T"])
    assume = bool(cfg.get("assume_cpqd_for_score_plus", False))
    if args.no_assume_cpqd:
        assume = False
    if args.assume_cpqd:
        assume = True
    pi1_list = args.pi1 if args.pi1 else list(cfg["dgp"]["pi1"])
    reduced = T != int(cfg["dgp"]["T"])
    written: list[str] = []
    for pi1 in pi1_list:
        written.extend(
            run_and_write(
                reps=args.reps,
                T=T,
                pi1=float(pi1),
                seed0=args.seed0,
                assume_cpqd=assume,
                scale=args.scale,
                reduced_horizon=reduced,
            )
        )
    print("wrote run_ids:")
    for rid in written:
        print(f"  {rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
