"""AAAI 2026 robust bandit unit tests. No Monte Carlo ranking."""

from __future__ import annotations

import math

import numpy as np
import pytest

from idea_C.baselines.robust_bandit.algorithms import run_bandit_mt
from idea_C.baselines.robust_bandit.dgp import assign_nonnulls, make_gaussian_instance, make_heavytail_instance
from idea_C.baselines.robust_bandit.eprocess import (
    HeavyTailEProcess,
    NonRobustEProcess,
    RobustEProcess,
    catoni_phi,
    recommended_lambda,
)
from idea_C.baselines.robust_bandit.sampling import mean_based_schedule_t, trimmed_mean
from idea_C.shared.ebh import ebh_rejections
from idea_C.shared.metrics import tpp
from idea_C.shared.seeds import seed_everything


def test_code_status_and_levy_deviation_are_documented() -> None:
    from idea_C.baselines.robust_bandit import CODE_STATUS
    from idea_C.baselines.robust_bandit.dgp import LEVY_ALPHA, LEVY_SCALE

    assert CODE_STATUS == "independent_reimplementation"
    assert LEVY_ALPHA == 1.5
    assert LEVY_SCALE == 1.0


def test_catoni_phi_bounds_and_log1p() -> None:
    assert catoni_phi(-2.0) == pytest.approx(-math.log(2.0))
    assert catoni_phi(2.0) == pytest.approx(math.log(2.0))
    assert catoni_phi(0.0) == pytest.approx(0.0)
    assert abs(catoni_phi(0.5)) <= math.log(2.0) + 1e-12


def test_recommended_lambda_matches_paper() -> None:
    assert recommended_lambda(0.1, 1.0) == pytest.approx(math.sqrt(0.1) / 2.0)


def test_robust_eprocess_stays_finite_under_levy_like_outliers() -> None:
    proc = RobustEProcess(3, eps=0.1, sigma=1.0)
    for x in (0.0, 1.0, 1000.0, -1000.0, 1e9, -1e9):
        proc.update(0, x)
    assert np.all(np.isfinite(proc.values))
    assert np.all(proc.values > 0.0)


def test_heavytail_eprocess_initializes_at_one() -> None:
    proc = HeavyTailEProcess(2, p=1.85, kappa=50.0, eps=0.1)
    np.testing.assert_allclose(proc.values, 1.0)
    proc.update(0, 5.0)
    assert np.isfinite(proc.values[0])


def test_nonrobust_is_marked_stress_test() -> None:
    assert NonRobustEProcess.STRESS_TEST_ONLY is True
    assert NonRobustEProcess.LABEL == "NOT_COMPARABLE"


def test_trimmed_mean_drops_tails() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 1000.0])
    assert trimmed_mean(x, 0.2) == pytest.approx(float(np.mean([1.0, 2.0, 3.0])))


def test_mean_based_schedule_formula() -> None:
    t1 = mean_based_schedule_t(1, k=4, eps=0.5, delta=0.05)
    inner = (4 * (math.pi**2) * 1) / (3 * 0.05)
    expect = math.ceil((4.0 * 1 / 0.25) * math.log(inner))
    assert t1 == max(int(expect), 1)


def test_same_seed_reproducible_uniform() -> None:
    def once(seed: int):
        rng = seed_everything(seed)
        env = make_gaussian_instance(6, 2, rng, eps=0.1)
        return run_bandit_mt(env, method="uniform", delta=0.05, budget=40)

    a = once(3)
    b = once(3)
    np.testing.assert_array_equal(a.arms, b.arms)
    np.testing.assert_allclose(a.rewards, b.rewards)
    np.testing.assert_array_equal(a.rejected_path, b.rejected_path)


def test_budget_exactly_spent_and_no_peeking() -> None:
    rng = seed_everything(1)
    env = make_gaussian_instance(5, 2, rng, eps=0.0)
    result = run_bandit_mt(env, method="eprocess_based", delta=0.05, budget=17)
    assert result.budget_spent == 17
    assert int(result.n_pulls.sum()) == 17
    env.assert_no_extra_generation(result.n_pulls)


def test_no_label_leakage_into_eprocess() -> None:
    rng = seed_everything(2)
    env = make_gaussian_instance(5, 2, rng, eps=0.0)
    truth = env.is_null.copy()
    result = run_bandit_mt(env, method="uniform", delta=0.05, budget=20)
    # Recompute e-process from rewards only; must match stored path.
    proc = RobustEProcess(5, eps=0.0, sigma=1.0)
    for t, (arm, y) in enumerate(zip(result.arms, result.rewards)):
        proc.update(int(arm), float(y))
        np.testing.assert_allclose(proc.values, result.evalues_path[t])
    # Labels were not required to reconstruct the path.
    assert truth.shape == (5,)


def test_nested_ebh_along_path_when_skipping_rejected() -> None:
    rng = seed_everything(4)
    env = make_gaussian_instance(6, 3, rng, mu_nonnull=3.0, eps=0.0)
    result = run_bandit_mt(env, method="eprocess_based", delta=0.2, budget=60)
    prev = np.zeros(6, dtype=bool)
    for row in result.rejected_path:
        assert np.all(prev <= row)
        prev = row


def test_ebh_level_nested_on_final_evalues() -> None:
    rng = seed_everything(5)
    env = make_gaussian_instance(6, 2, rng, mu_nonnull=2.0, eps=0.0)
    result = run_bandit_mt(env, method="uniform", delta=0.05, budget=40)
    e = result.evalues_path[-1]
    assert np.all(ebh_rejections(e, 0.05) <= ebh_rejections(e, 0.3))


def test_null_only_gaussian_does_not_reject_everything() -> None:
    rng = seed_everything(6)
    env = make_gaussian_instance(8, 0, rng, eps=0.0)
    result = run_bandit_mt(env, method="uniform", delta=0.05, budget=30)
    assert result.final_rejected.sum() <= 3


def test_strong_signal_beats_weak_signal() -> None:
    def tpp_at(mu: float, seed: int) -> float:
        rng = seed_everything(seed)
        env = make_gaussian_instance(6, 2, rng, mu_nonnull=mu, eps=0.0)
        result = run_bandit_mt(env, method="eprocess_based", delta=0.2, budget=80)
        return tpp(result.final_rejected, env.is_null)

    strong = np.mean([tpp_at(3.0, 20 + i) for i in range(4)])
    weak = np.mean([tpp_at(0.2, 20 + i) for i in range(4)])
    assert strong >= weak


def test_mean_based_unit_batches_respects_budget() -> None:
    rng = seed_everything(8)
    env = make_gaussian_instance(4, 1, rng, eps=0.1)
    result = run_bandit_mt(env, method="mean_based", delta=0.05, budget=12, unit_batches=True)
    assert result.budget_spent == 12
    env.assert_no_extra_generation(result.n_pulls)


def test_heavytail_sampler_runs() -> None:
    rng = seed_everything(9)
    env = make_heavytail_instance(k=6, n_nonnull=2, rng=rng)
    result = run_bandit_mt(env, method="algo3_heavytail", delta=0.05, budget=24)
    assert result.budget_spent == 24
    assert np.all(np.isfinite(result.evalues_path))
    env.assert_no_extra_generation(result.n_pulls)


def test_n_nonnull_grid_helpers() -> None:
    rng = seed_everything(0)
    mask = assign_nonnulls(50, 2, rng)
    assert int(mask.sum()) == 2
    from idea_C.baselines.robust_bandit.dgp import n_nonnull_from_spec

    assert n_nonnull_from_spec("floor(log K)", 50) == int(np.floor(np.log(50)))
    assert n_nonnull_from_spec("floor(sqrt K)", 50) == int(np.floor(np.sqrt(50)))


def test_runner_refuses_paper_scale_without_flag() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_aaai.py"
    spec = importlib.util.spec_from_file_location("run_aaai", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--scale", "paper"]) == 2
    assert mod.main(["--scale", "smoke"]) == 0
