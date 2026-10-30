"""v1.1 allocation rules must split when skill frequencies are unequal."""

import numpy as np
import pytest

from idea_B.methods import allocate_audit as allocate_audit_v10
from idea_B.methods import make_controller_state
from idea_B.methods.v1_1 import (
    allocate_equal_expected_count,
    allocate_proportional_frequency,
    allocate_uniform,
    expected_skill_audit_counts,
)


def _unbalanced_skills() -> np.ndarray:
    # 24 / 12 / 4: frequencies 0.6, 0.3, 0.1. Distinct under all three rules.
    return np.array([0] * 24 + [1] * 12 + [2] * 4, dtype=int)


def test_uniform_is_constant_per_sample():
    skills = _unbalanced_skills()
    q = allocate_uniform(skills.shape[0], 0.20)
    np.testing.assert_allclose(q, np.full(skills.shape[0], 0.20))
    assert float(np.mean(q)) == 0.20


def test_equal_skill_uniform_proportional_differ_when_frequencies_unequal():
    skills = _unbalanced_skills()
    budget = 0.20
    q_u = allocate_uniform(skills.shape[0], budget)
    q_e = allocate_equal_expected_count(skills, budget, 3)
    q_p = allocate_proportional_frequency(skills, budget, 3)
    assert not np.allclose(q_e, q_u)
    assert not np.allclose(q_p, q_u)
    assert not np.allclose(q_e, q_p)
    # Rare skill: higher per-sample q under equal expected counts, lower under proportional.
    assert float(np.mean(q_e[skills == 2])) > float(np.mean(q_e[skills == 0])) + 1e-8
    assert float(np.mean(q_p[skills == 2])) < float(np.mean(q_p[skills == 0])) - 1e-8
    assert float(np.mean(q_u[skills == 2])) == pytest.approx(float(np.mean(q_u[skills == 0])))


def test_equal_skill_expected_counts_are_equal_when_clipping_does_not_bind():
    skills = _unbalanced_skills()
    budget = 0.15
    q = allocate_equal_expected_count(skills, budget, 3)
    counts = expected_skill_audit_counts(q, skills, 3)
    present = counts > 0
    np.testing.assert_allclose(counts[present], counts[present][0], rtol=0, atol=1e-8)
    np.testing.assert_allclose(float(np.mean(q)), budget, atol=1e-8)


def test_proportional_expected_counts_increase_with_frequency():
    skills = _unbalanced_skills()
    q = allocate_proportional_frequency(skills, 0.20, 3)
    counts = expected_skill_audit_counts(q, skills, 3)
    assert counts[0] > counts[1] > counts[2]
    np.testing.assert_allclose(float(np.mean(q)), 0.20, atol=1e-8)


def test_v10_equal_skill_still_matches_uniform_on_unbalanced_batch():
    skills = _unbalanced_skills()
    n = skills.shape[0]
    phi = np.eye(3)[skills]
    theta = np.zeros(3)
    gramian = np.zeros((3, 3))
    rng = np.random.default_rng(0)
    q_v10, _state = allocate_audit_v10(
        "EqualSkillAllocation",
        phi,
        skills,
        theta,
        gramian,
        0.20,
        make_controller_state(3),
        rng,
        n_skills=3,
    )
    q_uniform = np.full(n, 0.20)
    np.testing.assert_allclose(q_v10, q_uniform)
    q_v11 = allocate_equal_expected_count(skills, 0.20, 3)
    assert not np.allclose(q_v11, q_v10)


def test_equal_frequencies_make_all_three_rules_constant():
    skills = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=int)
    budget = 0.30
    q_u = allocate_uniform(skills.shape[0], budget)
    q_e = allocate_equal_expected_count(skills, budget, 3)
    q_p = allocate_proportional_frequency(skills, budget, 3)
    np.testing.assert_allclose(q_e, q_u, atol=1e-10)
    np.testing.assert_allclose(q_p, q_u, atol=1e-10)
