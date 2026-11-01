#!/usr/bin/env python3
"""P1-A common multi-arm runner (Clean then Contaminated).

Same arm distribution, same budget, same evaluation time. Reuses AAAI
robust-bandit and e-PS implementations plus shared e-BH. No C-v1 method.

Default: print the plan and refuse. Pass ``--run``. Never paper-scale
200/500. 30-seed is always labeled ``pilot``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
IDEA_C = Path(__file__).resolve().parents[1]
CONFIG_DIR = IDEA_C / "configs"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCALES = ("smoke", "pilot")

# AAAI Algorithm 2 burn-in B(δ,ε)=4 ε^{-1} log(4K/δ²); KB at K=50,δ=0.05,ε=0.1.
DEFAULT_HORIZON = 22600

METHOD_SPECS = {
    "uniform": {
        "family": "aaai",
        "aaai_method": "uniform",
        "display_clean": "AAAI Uniform + robust e-process",
        "display_contaminated": "AAAI Uniform + robust e-process",
        "version_clean": "v0_oa_uniform",
        "version_contaminated": "v0_oa_c_uniform",
        "stress_under_contamination": False,
    },
    "greedy": {
        "family": "eps",
        "sampler": "greedy",
        "eprocess": "adaptive",
        "display_clean": "Greedy + Adaptive plug-in",
        "display_contaminated": "Greedy + Adaptive plug-in（stress test）",
        "version_clean": "v0_oa_greedy",
        "version_contaminated": "v0_oa_c_greedy_stress",
        "stress_under_contamination": True,
    },
    "eps_oracle": {
        "family": "eps",
        "sampler": "eps",
        "eprocess": "oracle",
        "display_clean": "e-PS Oracle LR",
        "display_contaminated": "e-PS Oracle LR 非鲁棒 increment（stress test）",
        "version_clean": "v0_oa_eps_oracle",
        "version_contaminated": "v0_oa_c_eps_oracle_stress",
        "stress_under_contamination": True,
    },
    "eps_adaptive": {
        "family": "eps",
        "sampler": "eps",
        "eprocess": "adaptive",
        "display_clean": "e-PS Adaptive plug-in",
        "display_contaminated": "e-PS Adaptive plug-in 非鲁棒 increment（stress test）",
        "version_clean": "v0_oa_eps_adaptive",
        "version_contaminated": "v0_oa_c_eps_adaptive_stress",
        "stress_under_contamination": True,
    },
    "mean_based": {
        "family": "aaai",
        "aaai_method": "mean_based",
        "display_clean": "AAAI Mean-based",
        "display_contaminated": "AAAI Mean-based",
        "version_clean": "v0_oa_mean",
        "version_contaminated": "v0_oa_c_mean",
        "stress_under_contamination": False,
    },
    "eprocess_based": {
        "family": "aaai",
        "aaai_method": "eprocess_based",
        "display_clean": "AAAI E-process-based",
        "display_contaminated": "AAAI E-process-based",
        "version_clean": "v0_oa_eprocess",
        "version_contaminated": "v0_oa_c_eprocess",
        "stress_under_contamination": False,
    },
}


class AlgoEpsOverlay:
    """Expose paper ε to AAAI algorithms without changing DGP contamination."""

    def __init__(self, env, algo_eps: float) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "eps", float(algo_eps))

    def __getattr__(self, name: str):
        return getattr(self._env, name)


@dataclass
class CommonProtocolEPSEnv:
    """Duck-typed e-PS env on the shared AAAI Gaussian arms.

    Adaptation vs e-PS paper: Oracle alternatives are μ=1 for every arm
    (common-protocol H1), not θ_alt = 0.02 k.
    """

    inner: object
    mu_nonnull: float = 1.0

    @property
    def k(self) -> int:
        return int(self.inner.k)

    @property
    def family(self) -> str:
        return "gaussian"

    @property
    def student_df(self) -> float:
        return 5.0

    @property
    def oracle_alternatives(self) -> np.ndarray:
        return np.full(self.k, float(self.mu_nonnull), dtype=float)

    @property
    def is_null(self) -> np.ndarray:
        return self.inner.is_null

    def observe(self, arm: int) -> float:
        return self.inner.observe(arm)

    def assert_no_extra_generation(self, pulls) -> None:
        self.inner.assert_no_extra_generation(pulls)


def load_config(name: str) -> dict:
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / name
    if not path.exists():
        path = CONFIG_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def algo2_burnin_total(k: int, delta: float, eps: float) -> int:
    from idea_C.baselines.robust_bandit.sampling import eprocess_burnin_length

    return int(k) * int(eprocess_burnin_length(k, delta, eps))


def make_env(cfg: dict, rng: np.random.Generator, *, contaminated: bool):
    from idea_C.baselines.robust_bandit.dgp import make_gaussian_instance

    k = int(cfg["K"])
    n1 = int(cfg.get("n_nonnull", cfg.get("dgp", {}).get("n_nonnull", 2)))
    mu0 = float(cfg.get("dgp", {}).get("mu_null", 0.0))
    mu1 = float(cfg.get("dgp", {}).get("mu_nonnull", 1.0))
    dgp_eps = 0.1 if contaminated else 0.0
    return make_gaussian_instance(k, n1, rng, mu_null=mu0, mu_nonnull=mu1, eps=dgp_eps)


def explanation(
    *,
    setting: str,
    spec: dict,
    scale: str,
    n_trials: int,
    horizon: int,
    burnin_total: int,
    seeds: list[int],
    contaminated: bool,
    stress: bool,
    status: str,
    error: str | None = None,
) -> str:
    parts = [
        f"P1-A {setting} {scale} {n_trials}-seed（标 {scale}，不是 paper-scale / 500）。",
        "common protocol：同一 arm 分布 N(μ,1) μ∈{0,1}，K=50，|H1|=2，同一预算，同一评估时点。",
        f"horizon={horizon}，Algorithm 2 KB≈{burnin_total}（ε_algo=0.1，δ=0.05）。",
        f"seeds={seeds}。code_status=independent_reimplementation / adapted。",
        "Lévy-stable 稳定性指数/尺度原文未给，代码用 alpha=1.5 scale=1.0。",
        "e-PS Oracle 适应：假设备择 μ=1（共同协议），不是原文 θ_alt=0.02k。",
        "Greedy 配对 Adaptive plug-in（已有 e-PS 实现，不是新方法）。",
        "AAAI 算法侧 ε=0.1（论文稳健参数）；Clean 数据 ε=0，Contaminated 数据 ε=1/10。",
        "Mean-based 用 Algorithm 1 的 t(s) 日程；t(1)≈3240/臂，22600 预算不足以完成全臂初始化。",
    ]
    if horizon <= burnin_total:
        parts.append(
            f"horizon={horizon} ≤ KB={burnin_total}：代码对 t≤KB 做均匀抽样，"
            "本预算内 E-process-based 从未进入 post-burn-in，与 Uniform 抽样规则相同"
            "（e-process 仍每步更新）。不得据此宣称 E-process / Mean-based 相对 Uniform 胜出。"
        )
    else:
        parts.append(
            f"horizon={horizon} > KB={burnin_total}，E-process-based 在 burn-in 后按 e-value 选臂。"
        )
    if stress:
        parts.append(
            "INVALIDITY STRESS TEST / NOT_COMPARABLE：污染下非鲁棒 e-process / e-PS increment "
            "不得用于反驳原文定理。"
        )
    if contaminated and not stress:
        parts.append("该行用 AAAI 稳健实现，可在污染设定下对照；仍标 independent_reimplementation。")
    if error:
        parts.append(f"失败：{error}")
    elif status != "ok":
        parts.append(f"status={status}")
    return "".join(parts)


def _finalize(directory: Path, *, seeds: list[int], log_text: str) -> None:
    (directory / "seeds.json").write_text(
        json.dumps({"seeds": seeds, "seed0": seeds[0] if seeds else None}, indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "run.log").write_text(log_text, encoding="utf-8")


def run_one_method(
    cfg: dict,
    *,
    method_key: str,
    setting: str,
    scale: str,
    seed0: int,
    n_trials: int,
    horizon: int,
) -> str:
    from idea_C.baselines.eps.algorithms import run_eps
    from idea_C.baselines.robust_bandit.algorithms import run_bandit_mt
    from idea_C.shared.metrics import fdr, samples_at_target_tpr, tpp
    from idea_C.shared.seeds import seed_everything
    from idea_C.shared.serialize import write_run
    from idea_C.shared.stats import mean_std_ci

    spec = METHOD_SPECS[method_key]
    contaminated = setting == "contaminated"
    table_id = str(cfg["table_id"])
    alpha = float(cfg.get("alpha", cfg.get("delta", 0.05)))
    delta = float(cfg.get("delta", alpha))
    algo_eps = float(cfg.get("algo_eps", 0.1))
    k = int(cfg["K"])
    burnin_total = algo2_burnin_total(k, delta, algo_eps)
    seeds = [seed0 + i for i in range(n_trials)]
    version = spec["version_contaminated"] if contaminated else spec["version_clean"]
    version = f"{version}_{scale}"
    display = spec["display_contaminated"] if contaminated else spec["display_clean"]
    stress = bool(contaminated and spec["stress_under_contamination"])
    data = (
        "P1-A2 contaminated Gaussian"
        if contaminated
        else "P1-A1 clean Gaussian"
    )
    run_id = f"{table_id}_{method_key}_{scale}_seed{seed0}"
    log_lines = [
        f"run_id={run_id}",
        f"method={method_key}",
        f"setting={setting}",
        f"scale={scale}",
        f"n_trials={n_trials}",
        f"seeds={seeds}",
        f"horizon={horizon}",
        f"algo_eps={algo_eps}",
        f"burnin_KB={burnin_total}",
        f"stress_test={stress}",
        "code_status=independent_reimplementation",
        "single_process=1",
    ]
    fdps: list[float] = []
    tpps: list[float] = []
    samples: list[float] = []
    status = "ok"
    error = None
    t0 = time.perf_counter()
    try:
        for i, seed in enumerate(seeds):
            rng = seed_everything(seed)
            env = make_env(cfg, rng, contaminated=contaminated)
            if spec["family"] == "aaai":
                wrapped = AlgoEpsOverlay(env, algo_eps)
                result = run_bandit_mt(
                    wrapped,
                    method=spec["aaai_method"],
                    delta=delta,
                    budget=horizon,
                )
                is_null = env.is_null
            else:
                eps_env = CommonProtocolEPSEnv(
                    env,
                    mu_nonnull=float(cfg.get("dgp", {}).get("mu_nonnull", 1.0)),
                )
                result = run_eps(
                    eps_env,
                    rng,
                    sampler=spec["sampler"],
                    eprocess=spec["eprocess"],
                    alpha=alpha,
                    budget=horizon,
                )
                is_null = env.is_null
            fdps.append(fdr(result.final_rejected, is_null))
            tpp_path = [tpp(row, is_null) for row in result.rejected_path]
            tpps.append(tpp_path[-1] if tpp_path else 0.0)
            samples.append(samples_at_target_tpr(tpp_path, delta=delta))
            print(f"    {method_key} seed={seed} fdp={fdps[-1]:.4f} tpp={tpps[-1]:.4f}", flush=True)
        runtime = time.perf_counter() - t0
        metrics = {
            "FDR↓": mean_std_ci(fdps)["mean"],
            "TPR↑": mean_std_ci(tpps)["mean"],
            "TPP↑": mean_std_ci(tpps)["mean"],
            "Runtime↓": runtime,
        }
        samp_mean = mean_std_ci(samples)["mean"]
        if samp_mean is not None and isinstance(samp_mean, float) and math.isfinite(samp_mean):
            metrics["Samples@Target TPR↓"] = samp_mean
        content = f"{display} {scale} {n_trials}-seed independent_reimplementation"
        if stress:
            content += " NOT_COMPARABLE stress test"
    except Exception as exc:
        runtime = time.perf_counter() - t0
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        metrics = {"Runtime↓": runtime}
        content = f"{display} {scale} {n_trials}-seed FAILED"
        log_lines.append(traceback.format_exc())
        print(f"  {method_key}: FAILED {exc}", flush=True)

    note = explanation(
        setting=setting,
        spec=spec,
        scale=scale,
        n_trials=n_trials,
        horizon=horizon,
        burnin_total=burnin_total,
        seeds=seeds,
        contaminated=contaminated,
        stress=stress,
        status=status,
        error=error,
    )
    log_lines.append(f"status={status} wall_s={runtime:.6f} metrics={metrics}")
    directory = write_run(
        run_id,
        table_id=table_id,
        version=version,
        content=content,
        data=data,
        config={
            **cfg,
            "method": method_key,
            "setting": setting,
            "scale": scale,
            "seed0": seed0,
            "seeds": seeds,
            "horizon_used": horizon,
            "n_trials_used": n_trials,
            "burnin_KB": burnin_total,
            "stress_test": stress,
            "validity_label": "NOT_COMPARABLE" if stress else "comparable_on_this_DGP",
        },
        seed=seed0,
        metrics=metrics,
        status=status,
        result_explanation=note,
        extra={
            "seeds": seeds,
            "wall_s": runtime,
            "fdps": fdps,
            "tpps": tpps,
            "stress_test": stress,
        },
    )
    _finalize(directory, seeds=seeds, log_text="\n".join(log_lines) + "\n")
    print(f"  wrote results/v0/raw/{run_id} status={status} wall={runtime:.3f}s", flush=True)
    return run_id


def list_plan(cfg: dict, setting: str) -> int:
    k = int(cfg["K"])
    delta = float(cfg.get("delta", 0.05))
    algo_eps = float(cfg.get("algo_eps", 0.1))
    kb = algo2_burnin_total(k, delta, algo_eps)
    print(f"P1-A {setting} table_id={cfg['table_id']} protocol={cfg['protocol']}")
    print(f"  K={k} |H1|={cfg.get('n_nonnull', 2)} alpha={cfg.get('alpha')} algo_eps={algo_eps}")
    print(f"  Algorithm 2 KB≈{kb}")
    print(f"  horizon={cfg.get('horizon')} n_trials={cfg.get('n_trials')}")
    print(f"  methods={cfg.get('methods')}")
    print("  e-PS Oracle/Adaptive are separate rows; no C-v1 method")
    print("  contaminated non-robust increment = stress test / NOT_COMPARABLE")
    print("No --run: refusing to start Monte Carlo.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P1-A common multi-arm runner")
    parser.add_argument("--setting", choices=("clean", "contaminated"), default="clean")
    parser.add_argument("--config", default=None, help="override config path")
    parser.add_argument("--method", default=None, choices=list(METHOD_SPECS))
    parser.add_argument("--scale", default="smoke", choices=list(SCALES))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--allow-paper-scale", action="store_true")
    args = parser.parse_args(argv)

    default_name = "p1a_clean.json" if args.setting == "clean" else "p1a_contaminated.json"
    cfg = load_config(args.config or default_name)
    if args.scale == "pilot":
        label = "pilot"
    else:
        label = "smoke"
    n_trials = args.n_trials if args.n_trials is not None else int(cfg["n_trials"][args.scale])
    horizon = args.horizon if args.horizon is not None else int(cfg["horizon"][args.scale])
    if n_trials >= 200 and not args.allow_paper_scale:
        print("Refusing paper-scale (>=200 trials). 30-seed is pilot, never paper-scale.", file=sys.stderr)
        return 2
    if not args.run:
        return list_plan(cfg, args.setting)

    methods = [args.method] if args.method else list(cfg.get("methods") or METHOD_SPECS)
    # Config may use display names; map back to keys.
    alias = {
        "Uniform": "uniform",
        "Greedy": "greedy",
        "Greedy (non-robust increment, stress test)": "greedy",
        "e-PS": "eps_adaptive",
        "e-PS Oracle LR": "eps_oracle",
        "e-PS Adaptive plug-in": "eps_adaptive",
        "e-PS Oracle LR (non-robust increment, stress test)": "eps_oracle",
        "e-PS Adaptive plug-in (non-robust increment, stress test)": "eps_adaptive",
        "AAAI Mean-based": "mean_based",
        "AAAI E-process-based": "eprocess_based",
        "e-PS (non-robust increment, stress test)": "eps_adaptive",
    }
    keys: list[str] = []
    for name in methods:
        key = name if name in METHOD_SPECS else alias.get(str(name))
        if key is None:
            print(f"skip unknown method {name!r}", file=sys.stderr)
            continue
        if key not in keys:
            keys.append(key)

    print(
        f"Running P1-A {args.setting} scale={args.scale} label={label} "
        f"trials={n_trials} horizon={horizon} methods={keys} single_process",
        flush=True,
    )
    written: list[str] = []
    for key in keys:
        written.append(
            run_one_method(
                cfg,
                method_key=key,
                setting=args.setting,
                scale=args.scale,
                seed0=args.seed,
                n_trials=n_trials,
                horizon=horizon,
            )
        )
    print("wrote run_ids:")
    for rid in written:
        print(f"  {rid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
