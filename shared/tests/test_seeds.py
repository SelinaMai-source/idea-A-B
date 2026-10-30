"""Deterministic seeding."""

import random

import numpy as np
import pytest

from shared.data_utils.seeds import seed_everything


def test_seed_everything_repeats_and_changes_with_the_seed():
    generator_a = seed_everything(0)
    draws_a = generator_a.random(6)
    python_a = random.random()
    numpy_a = np.random.random(6)

    generator_b = seed_everything(0)
    draws_b = generator_b.random(6)
    python_b = random.random()
    numpy_b = np.random.random(6)

    np.testing.assert_array_equal(draws_a, draws_b)
    assert python_a == python_b
    np.testing.assert_array_equal(numpy_a, numpy_b)

    generator_c = seed_everything(1)
    draws_c = generator_c.random(6)
    assert not np.array_equal(draws_a, draws_c)


def test_seed_rejects_bool_and_negative():
    with pytest.raises(TypeError):
        seed_everything(True)
    with pytest.raises(ValueError):
        seed_everything(-1)
