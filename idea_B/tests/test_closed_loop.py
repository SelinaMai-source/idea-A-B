"""Closed-loop transitions vs exogenous missingness."""

import numpy as np

from idea_B.benchmarks.Synthetic_Closed_Loop import make_environment
from shared.data_utils.seeds import seed_everything


def test_b2_observation_channel_depends_on_policy():
    rng = seed_everything(0)
    env = make_environment("B2", population_per_round=30)
    batch = env.sample_batch(rng, t=3)
    phi = batch["phi"]
    skills = batch["skills"]
    q_zero = env.natural_observation_prob(phi, skills, np.zeros(3), t=3)
    q_bias = env.natural_observation_prob(
        phi,
        skills,
        np.array([2.0, 2.0, -3.0]),
        t=3,
    )
    assert env.policy_affects_feedback is True
    assert not np.allclose(q_zero, q_bias)
    weak = skills == env.weak_skill
    if np.any(weak):
        assert float(np.mean(q_bias[weak])) <= float(np.mean(q_bias[~weak])) + 1e-12


def test_b1_observation_channel_ignores_policy():
    rng = seed_everything(1)
    env = make_environment("B1", population_per_round=30)
    batch = env.sample_batch(rng, t=5)
    phi = batch["phi"]
    skills = batch["skills"]
    q_a = env.natural_observation_prob(phi, skills, np.zeros(3), t=5)
    q_b = env.natural_observation_prob(phi, skills, np.array([5.0, -5.0, 5.0]), t=5)
    assert env.policy_affects_feedback is False
    np.testing.assert_allclose(q_a, q_b)
    q_later = env.natural_observation_prob(phi, skills, np.zeros(3), t=20)
    weak = skills == env.weak_skill
    if np.any(weak):
        assert float(np.mean(q_later[weak])) <= float(np.mean(q_a[weak])) + 1e-12


def test_b3_unrecoverable_projects_out_weak_coordinate():
    rng = seed_everything(2)
    env = make_environment("B3_unrecoverable", population_per_round=40)
    batch = env.sample_batch(rng, t=0)
    weak = batch["skills"] == env.weak_skill
    assert env.audits_identify_weak_skill is False
    if np.any(weak):
        np.testing.assert_allclose(batch["phi"][weak, env.weak_skill], 0.0)
        audit = np.ones(batch["phi"].shape[0], dtype=bool)
        effective = env.apply_audit_identifiability(batch, audit)
        assert not np.any(effective[weak])
