#!/usr/bin/env python3
"""UAI 2026 local-dependence runner. Venue is NOT CCF-A Main.

Default: list config and refuse. Pass --run for a real simulation.
200-trial paper-scale is refused without --paper-scale.
Do not mix FDR and SupFDR.
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

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "uai_local_dependence.json"
PAPER_TRIALS = 200

METHOD_VERSION = {
    "e-LOND": "v0_elond_uai",
    "Closed e-LOND": "v0_closed_elond",
    "Donation e-LOND": "v0_donation_elond",
    "r-LOND": "v0_rlond_uai",
    "Closed r-LOND": "v0_closed_rlond",
    "Donation r-LOND": "v0_donation_rlond",
}


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def list_config() -> int:
    cfg = load_config()
    print("UAI 2026 online e-closure / donation (**NOT CCF-A Main**)")
    print(f"config: {CONFIG}")
    print(f"DGP: {cfg['dgp']['name']} m={cfg['dgp']['m']} L={cfg['dgp']['lag_L']}")
    print(f"related official exp name: {cfg['related_official_experiment']['name']} (AISTATS 2024, not this DGP)")
    print(f"paper trials: {cfg['n_trials_paper']} (refused unless --paper-scale)")
    print(f"methods: {cfg['methods']}")
    print("Metrics: FDR=E[FDP_T], SupFDR=E[sup_t FDP_t] — distinct")
    print(f"online code status: {cfg['code_status']['online_uai']}")
    return 0


def refuse_paper_scale(reps: int, paper_scale: bool) -> str | None:
    if reps >= PAPER_TRIALS and not paper_scale:
        return (
            f"Refusing paper-scale UAI run (reps={reps} ≥ {PAPER_TRIALS}). "
            "Pass --paper-scale only for a true 200-trial experiment."
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


def _cebh_note() -> str:
    from idea_C.baselines.eclosure import try_offline_closedeBH

    probe = try_offline_closedeBH([1.0, 1.0], alpha=0.05)
    return (
        f"C++ _cebh / offline closedeBH status={probe.status} "
        f"reason={probe.reason or 'n/a'} (does not block online UAI methods)"
    )


def run_and_write(
    *,
    reps: int,
    m: int,
    pi1: float,
    seed0: int,
    scale: str,
    reduced: bool,
) -> list[str]:
    from idea_C.baselines.eclosure import (
        average_power_terminal,
        closed_e_lond,
        closed_r_lond,
        donation_e_lond,
        donation_r_lond,
        e_lond,
        fdr_terminal,
        local_dependent_gaussian,
        r_lond,
        sup_fdp,
    )
    from idea_C.shared.serialize import write_run
    from idea_C.shared.stats import mean_std_ci

    cfg = load_config()
    seeds = [seed0 + r for r in range(reps)]
    delta = float(cfg.get("delta", 0.1))
    lag = int(cfg["dgp"]["lag_L"])
    protocol = (
        "UAI §4 local-dependent Gaussian (NOT AISTATS lag_comp)"
        + (" reduced-m / NOT paper protocol" if reduced else "")
    )
    data = (
        f"UAI Fig2 local-dep $m={m},L={lag},\\delta={delta}$"
        + (" reduced-m (NOT paper protocol)" if reduced else "")
    )
    cebh = _cebh_note()
    print(
        f"Running UAI streams reps={reps} m={m} pi1={pi1} seed0={seed0} "
        f"scale={scale} protocol={protocol}"
    )
    print("FDR and SupFDR are computed separately.")
    print(cebh)
    t_gen = time.perf_counter()
    streams = [
        local_dependent_gaussian(m=m, pi1=pi1, seed=s, lag=lag) for s in seeds
    ]
    gen_s = time.perf_counter() - t_gen
    print(f"  DGP generated in {gen_s:.3f}s (UAI §4 banded Σ, not lag_comp)")

    dispatch = {
        "e-LOND": lambda stream: e_lond(stream.evalues, delta=delta),
        "Closed e-LOND": lambda stream: closed_e_lond(stream.evalues, delta=delta),
        "Donation e-LOND": lambda stream: donation_e_lond(stream.evalues, delta=delta),
        "r-LOND": lambda stream: r_lond(stream.pvalues, delta=delta),
        "Closed r-LOND": lambda stream: closed_r_lond(stream.pvalues, delta=delta),
        "Donation r-LOND": lambda stream: donation_r_lond(stream.pvalues, delta=delta),
    }
    run_ids: list[str] = []
    for name in list(cfg.get("methods") or dispatch):
        version = f"{METHOD_VERSION.get(name, 'v0_' + _slug(name))}_{scale}_pi{_slug(str(pi1))}"
        if reduced:
            version = f"{version}_m{m}"
        run_id = f"uai_local_dependence_{_slug(name)}_{scale}_pi{_slug(str(pi1))}_m{m}_seed{seed0}"
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
            f"m={m}",
            f"L={lag}",
            f"pi1={pi1}",
            f"delta={delta}",
            "code_status=independent_reimplementation",
            "DGP=UAI §4 local-dependent Gaussian (NOT AISTATS lag_comp)",
            cebh,
            "FDR=E[FDP_T]; SupFDR=E[sup_t FDP_t]; never mixed",
        ]
        t0 = time.perf_counter()
        status = "ok"
        note = (
            f"自动追加，{scale} {reps}-seed，independent_reimplementation，"
            f"{protocol}；{cebh}；结果解释待人工填写"
        )
        metrics: dict = {}
        runtime = 0.0
        try:
            fn = dispatch[name]
            for i, stream in enumerate(streams):
                result = fn(stream)
                if result.status != "ok":
                    status = result.status
                    note = result.reason or note
                    break
                fdps.append(fdr_terminal(result.rejected, stream.is_null))
                sups.append(sup_fdp(result.rejected, stream.is_null))
                powers.append(average_power_terminal(result.rejected, stream.is_null))
                nrejs.append(float(result.n_rejections))
                print(f"    {name} seed={seeds[i]} done")
            runtime = time.perf_counter() - t0
            if status == "ok":
                metrics = {
                    "FDR↓": mean_std_ci(fdps)["mean"],
                    "SupFDR↓": mean_std_ci(sups)["mean"],
                    "Power↑": mean_std_ci(powers)["mean"],
                    "Rejections↑": mean_std_ci(nrejs)["mean"],
                    "Runtime↓": runtime,
                }
            log_lines.append(f"status={status}")
            log_lines.append(f"wall_s={runtime:.6f}")
            log_lines.append(f"fdps={fdps}")
            log_lines.append(f"sups={sups}")
            log_lines.append(f"powers={powers}")
            log_lines.append(f"nrejs={nrejs}")
            log_lines.append(f"metrics={metrics}")
            print(
                f"  {name}: status={status} FDR={metrics.get('FDR↓')} "
                f"SupFDR={metrics.get('SupFDR↓')} Power={metrics.get('Power↑')} "
                f"Rej={metrics.get('Rejections↑')} runtime={runtime:.3f}s"
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
            table_id="uai_local_dependence",
            version=version,
            content=(
                f"{name} {scale} {reps}-seed independent_reimplementation "
                "(NOT CCF-A Main; not lag_comp)"
            ),
            data=data,
            config={
                **cfg,
                "method": name,
                "scale": scale,
                "reps": reps,
                "m": m,
                "pi1": pi1,
                "seed0": seed0,
                "seeds": seeds,
                "reduced_m": reduced,
                "protocol": protocol,
                "cebh_note": cebh,
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
                "fdps": fdps,
                "sup_fdps": sups,
                "powers": powers,
                "n_rejections": nrejs,
                "cebh_note": cebh,
                "dgp_note": "UAI 2026 §4 / Appendix C.3 local-dependent Gaussian; NOT AISTATS 2024 lag_comp",
            },
        )
        _finalize(directory, seeds=seeds, log_text="\n".join(log_lines) + "\n")
        run_ids.append(run_id)
    return run_ids


def write_not_run_skip(*, reason: str, version: str, content: str, data: str) -> str:
    from idea_C.shared.serialize import write_run

    run_id = f"uai_local_dependence_{_slug(version)}_NOT_RUN"
    directory = write_run(
        run_id,
        table_id="uai_local_dependence",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="UAI 2026 v0 runner (NOT CCF-A Main)")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--scale", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--m", type=int, default=None)
    parser.add_argument("--pi1", type=float, default=0.3)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--paper-scale", action="store_true")
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

    m = args.m if args.m is not None else int(cfg["dgp"]["m"])
    reduced = m != int(cfg["dgp"]["m"])
    written = run_and_write(
        reps=args.reps,
        m=m,
        pi1=args.pi1,
        seed0=args.seed0,
        scale=args.scale,
        reduced=reduced,
    )
    print("wrote run_ids:")
    for rid in written:
        print(f"  {rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
