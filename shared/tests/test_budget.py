"""Audit fraction accounting."""

import numpy as np
import pytest

from shared.evaluation.budget import (
    assert_budget,
    expected_audit_fraction,
    realized_audit_fraction,
)


def test_expected_and_realized_audit_fractions():
    q = np.array([0.1, 0.2, 0.3, 0.4])
    a = np.array([1, 0, 1, 0])
    assert expected_audit_fraction(q) == pytest.approx(0.25)
    assert expected_audit_fraction(0.1) == pytest.approx(0.1)
    assert realized_audit_fraction(a) == pytest.approx(0.5)
    assert realized_audit_fraction(np.array([True, False, False, False])) == pytest.approx(0.25)


def test_assert_budget_uses_inclusive_tolerance():
    assert_budget(0.5, 0.5, 0.0)
    assert_budget(0.3, 0.1, 0.2)
    with pytest.raises(AssertionError):
        assert_budget(0.5, 0.25, 0.1)
    realized = realized_audit_fraction(np.array([1, 0, 1, 0]))
    assert_budget(realized, expected_audit_fraction(np.array([0.5, 0.5, 0.5, 0.5])), 0.0)
