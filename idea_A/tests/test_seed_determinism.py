"""Same seed reproduces a world; a different seed changes it."""

import numpy as np

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import simulate
from idea_A.experiments.protocol import MethodSpec, run_method


def test_simulate_is_deterministic_and_seed_sensitive():
    a = simulate("mnar", 0, n=400)
    b = simulate("mnar", 0, n=400)
    c = simulate("mnar", 1, n=400)
    np.testing.assert_array_equal(a.y, b.y)
    np.testing.assert_array_equal(a.s, b.s)
    np.testing.assert_array_equal(a.stratum, b.stratum)
    assert not np.array_equal(a.y, c.y)


def test_audit_method_repeats_with_the_same_rng_seed():
    world = simulate("mar", 0, n=400)
    spec = MethodSpec("uniform_ipw", "uniform", "ipw")
    config = {"n": world.n, "exploration_mix": 0.15, "q_floor": 1e-6, "ridge_lambda": 1e-3}
    row_a = run_method(world, spec, 0.10, config, rng=np.random.default_rng(123))
    row_b = run_method(world, spec, 0.10, config, rng=np.random.default_rng(123))
    row_c = run_method(world, spec, 0.10, config, rng=np.random.default_rng(124))
    assert row_a["population_mean_reward"]["estimate"] == row_b["population_mean_reward"]["estimate"]
    assert row_a["n_audited"] == row_b["n_audited"]
    assert row_a["n_audited"] != row_c["n_audited"] or (
        row_a["population_mean_reward"]["estimate"] != row_c["population_mean_reward"]["estimate"]
    )
