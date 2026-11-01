"""e-PS (arXiv:2609.26651) unit tests. No Monte Carlo ranking / watermark."""

from __future__ import annotations

import numpy as np
import pytest

from idea_C.baselines.eps.algorithms import run_eps
from idea_C.baselines.eps.dgp import alternative_mean, draw_theta, make_eps_instance
from idea_C.baselines.eps.eprocess import AdaptivePluginLR, OracleLR, gaussian_log_lr, student_t_log_lr
from idea_C.baselines.eps.sampling import variance_proxy
from idea_C.shared.ebh import ebh_rejections
from idea_C.shared.metrics import tpp
from idea_C.shared.seeds import seed_everything


def test_code_status_preprint_label() -> None:
    from idea_C.baselines.eps import CODE_STATUS

    assert CODE_STATUS == "independent_reimplementation"


def test_theta_formula_one_based() -> None:
    rng = seed_everything(0)
    # Force all non-null to check 0.02*k.
    theta = 0.02 * np.arange(1, 6)
    np.testing.assert_allclose(theta[0], 0.02)
    np.testing.assert_allclose(alternative_mean(3), 0.06)
    drawn = draw_theta(20, rng, pi1=1.0)
    np.testing.assert_allclose(drawn, 0.02 * np.arange(1, 21))


def test_oracle_uses_hypothesized_alternative_not_truth() -> None:
    oracle = OracleLR(np.array([0.02, 0.04]), family="gaussian")
    # Truth could be 0; increment still uses 0.02.
    log_e = oracle.log_increment(0, 1.0)
    assert log_e == pytest.approx(gaussian_log_lr(1.0, 0.02))
    assert log_e != pytest.approx(gaussian_log_lr(1.0, 0.0))


def test_adaptive_first_increment_is_one_then_uses_past_only() -> None:
    ad = AdaptivePluginLR(1, family="gaussian")
    ad.update(0, 4.0)
    assert ad.values[0] == pytest.approx(1.0)
    # Second increment uses θ̂ = 4 from the first observation only.
    before = float(ad.values[0])
    ad.update(0, 0.0)
    expected = before * np.exp(gaussian_log_lr(0.0, 4.0))
    assert ad.values[0] == pytest.approx(expected)


def test_oracle_and_adaptive_are_separate_classes() -> None:
    assert OracleLR.name != AdaptivePluginLR.name
    assert OracleLR is not AdaptivePluginLR


def test_student_t_log_lr_finite() -> None:
    val = student_t_log_lr(1.5, 0.4, df=5.0)
    assert np.isfinite(val)


def test_variance_proxy_1_1_and_n_lt_2() -> None:
    assert variance_proxy([1.0]) == pytest.approx(1.1)
    xs = [1.0, 3.0, 5.0]
    assert variance_proxy(xs) == pytest.approx(1.1 * float(np.var(xs, ddof=1)))


def test_same_seed_reproducible() -> None:
    def once(seed: int):
        rng = seed_everything(seed)
        theta = np.array([0.0, 0.0, 0.4, 0.0])
        env = make_eps_instance(4, rng, family="gaussian", theta=theta)
        return run_eps(env, rng, sampler="eps", eprocess="oracle", alpha=0.05, budget=24)

    a = once(15)
    b = once(15)
    np.testing.assert_array_equal(a.arms, b.arms)
    np.testing.assert_allclose(a.rewards, b.rewards)
    np.testing.assert_array_equal(a.rejected_path, b.rejected_path)


def test_budget_and_no_peeking() -> None:
    rng = seed_everything(1)
    env = make_eps_instance(5, rng, family="gaussian", theta=np.array([0.0, 0.0, 0.5, 0.0, 0.0]))
    result = run_eps(env, rng, sampler="uniform", eprocess="oracle", alpha=0.05, budget=18)
    assert result.budget_spent == 18
    assert int(result.n_pulls.sum()) == 18
    env.assert_no_extra_generation(result.n_pulls)


def test_nested_rejections_along_path() -> None:
    rng = seed_everything(2)
    theta = np.array([0.0, 1.2, 0.0, 1.5, 0.0])
    env = make_eps_instance(5, rng, family="gaussian", theta=theta)
    result = run_eps(env, rng, sampler="eps", eprocess="oracle", alpha=0.2, budget=40)
    prev = np.zeros(5, dtype=bool)
    for row in result.rejected_path:
        assert np.all(prev <= row)
        prev = row


def test_ebh_level_nested_on_final_evalues() -> None:
    rng = seed_everything(3)
    env = make_eps_instance(5, rng, family="gaussian", theta=np.array([0.0, 0.8, 0.0, 0.0, 1.0]))
    result = run_eps(env, rng, sampler="greedy", eprocess="oracle", alpha=0.05, budget=30)
    e = result.evalues_path[-1]
    assert np.all(ebh_rejections(e, 0.05) <= ebh_rejections(e, 0.4))


def test_null_only_oracle_fdr_sanity() -> None:
    rng = seed_everything(4)
    env = make_eps_instance(6, rng, family="gaussian", theta=np.zeros(6))
    result = run_eps(env, rng, sampler="uniform", eprocess="oracle", alpha=0.05, budget=24)
    assert result.final_rejected.sum() <= 2


def test_strong_vs_weak_oracle() -> None:
    def tpp_at(mu: float, seed: int) -> float:
        rng = seed_everything(seed)
        theta = np.array([0.0, 0.0, mu, 0.0])
        env = make_eps_instance(4, rng, family="gaussian", theta=theta)
        # Oracle alternatives stay 0.02k; inject a matching alternative by
        # overwriting after construct would leak. Instead compare Adaptive
        # plug-in power, which uses data only.
        result = run_eps(env, rng, sampler="eps", eprocess="adaptive", alpha=0.2, budget=60)
        return tpp(result.final_rejected, env.is_null)

    strong = np.mean([tpp_at(2.5, 30 + i) for i in range(3)])
    weak = np.mean([tpp_at(0.05, 30 + i) for i in range(3)])
    assert strong >= weak


def test_student_t_path_finite() -> None:
    rng = seed_everything(5)
    env = make_eps_instance(4, rng, family="student_t", theta=np.array([0.0, 0.8, 0.0, 0.0]))
    result = run_eps(env, rng, sampler="eps", eprocess="oracle", alpha=0.05, budget=20)
    assert np.all(np.isfinite(result.evalues_path))
    assert np.all(result.evalues_path > 0.0)
    env.assert_no_extra_generation(result.n_pulls)


def test_no_future_in_adaptive_plugin() -> None:
    rng = seed_everything(6)
    env = make_eps_instance(3, rng, family="gaussian", theta=np.array([0.0, 1.0, 0.0]))
    result = run_eps(env, rng, sampler="uniform", eprocess="adaptive", alpha=0.2, budget=15)
    ad = AdaptivePluginLR(3, family="gaussian")
    for t, (arm, y) in enumerate(zip(result.arms, result.rewards)):
        ad.update(int(arm), float(y))
        np.testing.assert_allclose(ad.values, result.evalues_path[t])


def test_runner_refuses_paper_scale() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_eps.py"
    spec = importlib.util.spec_from_file_location("run_eps", script)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--scale", "paper"]) == 2
    assert mod.main(["--scale", "smoke"]) == 0
