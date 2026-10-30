"""Observed design Gramian construction."""

import numpy as np
import pytest

from idea_B.methods import (
    accumulate_conditional_gramian,
    accumulate_observed_gramian,
    condition_number,
    lambda_min,
    observed_design_gramian,
    skill_information,
)


def test_observed_design_gramian_matches_sum_of_outer_products():
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    mask = np.array([1.0, 0.0, 1.0, 0.0])
    gramian = observed_design_gramian(phi, mask)
    expected = np.outer(phi[0], phi[0]) + np.outer(phi[2], phi[2])
    np.testing.assert_allclose(gramian, expected)
    running = np.zeros((2, 2))
    running = accumulate_observed_gramian(running, phi[:2], mask[:2])
    running = accumulate_observed_gramian(running, phi[2:], mask[2:])
    np.testing.assert_allclose(running, expected)


def test_conditional_gramian_uses_observation_probabilities():
    phi = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([0.25, 0.5])
    gramian = accumulate_conditional_gramian(np.zeros((2, 2)), phi, q)
    expected = 0.25 * np.outer(phi[0], phi[0]) + 0.5 * np.outer(phi[1], phi[1])
    np.testing.assert_allclose(gramian, expected)


def test_lambda_min_and_condition_number_of_known_matrix():
    gramian = np.diag([4.0, 1.0, 0.25])
    assert lambda_min(gramian) == pytest.approx(0.25)
    assert condition_number(gramian) == pytest.approx(16.0)
    info = skill_information(gramian)
    np.testing.assert_allclose(info, [4.0, 1.0, 0.25])


def test_unobserved_direction_has_zero_eigenvalue():
    phi = np.array([[1.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
    mask = np.ones(3)
    gramian = observed_design_gramian(phi, mask)
    assert lambda_min(gramian) == pytest.approx(0.0, abs=1e-12)
    assert skill_information(gramian)[1] == pytest.approx(0.0)
