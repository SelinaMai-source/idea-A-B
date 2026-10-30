"""Gramian PSD / rank / eigenvalue stability, and the three information types."""

import numpy as np
import pytest

from idea_B.methods.v1_1 import (
    SlidingWindowGramian,
    all_candidates_gramian,
    accumulate_conditional_gramian,
    accumulate_observed_gramian,
    condition_number,
    eigenspectrum,
    is_psd,
    lambda_min,
    numerical_rank,
    observed_design_gramian,
    skill_information,
)


def test_realized_gramian_is_psd_and_matches_outer_products():
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    mask = np.array([1.0, 0.0, 1.0, 0.0])
    gramian = observed_design_gramian(phi, mask)
    expected = np.outer(phi[0], phi[0]) + np.outer(phi[2], phi[2])
    np.testing.assert_allclose(gramian, expected)
    assert is_psd(gramian)
    values, _vectors = eigenspectrum(gramian)
    assert float(np.min(values)) >= -1e-12


def test_candidates_gramian_dominates_realized_in_loewner_order():
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    mask = np.array([1.0, 0.0, 1.0])
    realized = observed_design_gramian(phi, mask)
    candidates = all_candidates_gramian(phi)
    gap = 0.5 * ((candidates - realized) + (candidates - realized).T)
    assert float(np.min(np.linalg.eigvalsh(gap))) >= -1e-10
    assert numerical_rank(realized) <= numerical_rank(candidates)


def test_unobserved_direction_has_rank_deficiency():
    phi = np.array([[1.0, 0.0], [0.5, 0.0], [2.0, 0.0]])
    mask = np.ones(3)
    gramian = observed_design_gramian(phi, mask)
    assert lambda_min(gramian) == pytest.approx(0.0, abs=1e-12)
    assert numerical_rank(gramian) == 1
    assert skill_information(gramian)[1] == pytest.approx(0.0)


def test_conditional_gramian_uses_probabilities_not_realized_mask():
    phi = np.array([[1.0, 0.0], [0.0, 1.0]])
    q = np.array([0.25, 0.5])
    gramian = accumulate_conditional_gramian(np.zeros((2, 2)), phi, q)
    expected = 0.25 * np.outer(phi[0], phi[0]) + 0.5 * np.outer(phi[1], phi[1])
    np.testing.assert_allclose(gramian, expected)
    assert is_psd(gramian)


def test_eigenvalue_stability_on_known_psd_matrix():
    gramian = np.diag([4.0, 1.0, 0.25])
    values_a, _ = eigenspectrum(gramian)
    values_b = np.linalg.eigvalsh(0.5 * (gramian + gramian.T))
    np.testing.assert_allclose(values_a, np.sort(values_b))
    assert lambda_min(gramian) == pytest.approx(0.25)
    assert condition_number(gramian) == pytest.approx(16.0)
    perturbed = gramian + 1e-10 * np.eye(3)
    np.testing.assert_allclose(eigenspectrum(perturbed)[0], values_a, rtol=0, atol=1e-8)


def test_sliding_window_matches_last_increments_and_stays_psd():
    window = SlidingWindowGramian(2, window=2)
    a = np.array([[1.0, 0.0], [0.0, 0.0]])
    b = np.array([[0.0, 0.0], [0.0, 2.0]])
    c = np.array([[3.0, 0.0], [0.0, 0.0]])
    window.push(a)
    window.push(b)
    np.testing.assert_allclose(window.gramian(), a + b)
    window.push(c)
    np.testing.assert_allclose(window.gramian(), b + c)
    assert is_psd(window.gramian())
    stats = window.summaries()
    assert stats["rank"] == 2
    assert stats["n_increments"] == 2


def test_accumulate_observed_matches_sum_of_batches():
    phi = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]])
    mask = np.array([1.0, 0.0, 1.0, 0.0])
    expected = observed_design_gramian(phi, mask)
    running = np.zeros((2, 2))
    running = accumulate_observed_gramian(running, phi[:2], mask[:2])
    running = accumulate_observed_gramian(running, phi[2:], mask[2:])
    np.testing.assert_allclose(running, expected)
