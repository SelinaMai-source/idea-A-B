"""Env2 identification: explicit likelihood / Fisher, not a plot."""

import numpy as np
import pytest

from idea_B.benchmarks.B_Env2_Observational_Equivalence import make_environment as make_env2
from idea_B.methods.v1_1 import (
    audit_restores_identifiability,
    fisher_information,
    likelihoods_equal,
    minimax_identification_lower_bound,
    observational_equivalence_report,
    observed_log_likelihood,
    structural_unrecoverable,
    two_latent_states,
)
from shared.data_utils.seeds import seed_everything


def _one_hot_batch(skills: np.ndarray, theta: np.ndarray, noise_std: float, rng) -> dict:
    n = skills.shape[0]
    d = theta.shape[0]
    phi = np.zeros((n, d), dtype=float)
    phi[np.arange(n), skills] = 1.0
    y = phi @ theta + noise_std * rng.normal(size=n)
    return {"phi": phi, "y": y, "skills": skills}


def test_two_latent_states_agree_off_the_weak_coordinate():
    theta_a, theta_b = two_latent_states()
    np.testing.assert_allclose(theta_a[:2], theta_b[:2])
    assert theta_a[2] != theta_b[2]
    bound = minimax_identification_lower_bound(theta_a, theta_b)
    assert bound == pytest.approx(0.5 * abs(theta_a[2] - theta_b[2]))
    assert bound > 0.0


def test_natural_support_makes_likelihoods_equal_and_fisher_kernel_on_weak_skill():
    rng = seed_everything(0)
    theta_a, theta_b = two_latent_states()
    skills = np.array([0] * 20 + [1] * 16, dtype=int)
    batch = _one_hot_batch(skills, theta_a, 0.15, rng)
    natural = np.ones(skills.shape[0], dtype=bool)
    report = observational_equivalence_report(
        theta_a,
        theta_b,
        batch["phi"],
        batch["y"],
        natural,
        weak_skill=2,
        noise_std=0.15,
    )
    assert report["likelihoods_equal"] is True
    assert report["weak_coordinate_unidentified"] is True
    assert report["positive_lower_bound"] is True
    assert report["proof"] == "gaussian_observed_likelihood_and_fisher"
    fisher = fisher_information(batch["phi"], natural, 0.15)
    assert fisher[2, 2] == pytest.approx(0.0, abs=1e-12)
    ll_a = observed_log_likelihood(theta_a, batch["phi"], batch["y"], natural, 0.15)
    ll_b = observed_log_likelihood(theta_b, batch["phi"], batch["y"], natural, 0.15)
    assert ll_a == pytest.approx(ll_b, abs=1e-10)


def test_randomized_audit_of_missing_skill_restores_identifiability():
    rng = seed_everything(1)
    theta_a, theta_b = two_latent_states()
    skills = np.array([0] * 16 + [1] * 16 + [2] * 8, dtype=int)
    batch = _one_hot_batch(skills, theta_a, 0.15, rng)
    natural = skills != 2
    audit = skills == 2
    result = audit_restores_identifiability(
        batch["phi"], natural, audit, weak_skill=2, noise_std=0.15
    )
    assert result["structural_unrecoverable"] is False
    assert result["weak_in_natural_kernel"] is True
    assert result["restored_by_audit"] is True
    assert result["fisher_weak_natural"] == pytest.approx(0.0, abs=1e-12)
    assert result["fisher_weak_observed"] > 1.0
    observed = natural | audit
    assert likelihoods_equal(
        theta_a, theta_b, batch["phi"], batch["y"], natural, 0.15
    )
    assert not likelihoods_equal(
        theta_a, theta_b, batch["phi"], batch["y"], observed, 0.15, atol=1e-6
    )


def test_structural_unrecoverable_stays_unidentified_even_with_full_audit():
    rng = seed_everything(2)
    env = make_env2(unrecoverable=True, population_per_round=40, feature_noise=0.0)
    batch = env.sample_batch(rng, t=0)
    assert env.unrecoverable is True
    assert structural_unrecoverable(batch["phi"], env.weak_skill)
    audit = np.ones(batch["phi"].shape[0], dtype=bool)
    natural = np.zeros(batch["phi"].shape[0], dtype=bool)
    result = audit_restores_identifiability(
        batch["phi"], natural, audit, env.weak_skill, env.noise_std
    )
    assert result["structural_unrecoverable"] is True
    assert result["restored_by_audit"] is False
    effective = env.apply_audit_identifiability(batch, audit)
    weak = batch["skills"] == env.weak_skill
    if np.any(weak):
        assert not np.any(effective[weak])


def test_env2_recoverable_natural_channel_excludes_weak_skill():
    rng = seed_everything(3)
    env = make_env2(unrecoverable=False, population_per_round=40, feature_noise=0.0)
    batch = env.sample_batch(rng, t=0)
    theta0 = env.default_theta0()
    q_nat = env.natural_observation_prob(batch["phi"], batch["skills"], theta0, t=0)
    weak = batch["skills"] == env.weak_skill
    if np.any(weak):
        np.testing.assert_allclose(q_nat[weak], 0.0)
    assert env.extra["identification_error_lower_bound"] > 0.0
    assert env.audits_identify_weak_skill is True
