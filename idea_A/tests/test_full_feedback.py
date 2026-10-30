"""Full feedback (q=1) recovers finite-population truths."""

import numpy as np
import pytest

from idea_A.benchmarks.Synthetic_Selective_Feedback.simulator import MECHANISMS, simulate
from idea_A.experiments.protocol import MethodSpec, run_method
from idea_A.methods.v1_load import load_v1_module
from shared.metrics.estimators import dr_mean, ipw_mean, observed_only_mean, snips_mean

_intervention = load_v1_module("intervention")


@pytest.mark.parametrize("mechanism", MECHANISMS)
def test_full_feedback_recovers_population_mean(mechanism):
    world = simulate(mechanism, 0, n=500)
    q, a = _intervention.full_feedback(world.n)
    y = world.observe(a)
    truth = world.truths["population_mean_reward"]
    assert not np.any(np.isnan(y))
    assert observed_only_mean(y, a) == pytest.approx(truth)
    assert ipw_mean(y, a, q) == pytest.approx(truth)
    assert snips_mean(y, a, q) == pytest.approx(truth)
    assert dr_mean(y, a, q, np.zeros(world.n)) == pytest.approx(truth)


def test_oracle_method_spec_recovers_truths():
    world = simulate("zero_overlap", 1, n=500)
    row = run_method(
        world,
        MethodSpec("full_feedback_oracle", "full_oracle", "ipw", uses_budget=False),
        budget=0.05,
        config={"n": world.n, "exploration_mix": 0.15, "q_floor": 1e-6, "ridge_lambda": 1e-3},
        rng=np.random.default_rng(0),
    )
    assert row["population_mean_reward"]["estimate"] == pytest.approx(
        world.truths["population_mean_reward"]
    )
    assert row["preference_rate"]["estimate"] == pytest.approx(world.truths["preference_rate"])
    assert row["policy_value_target_distribution"]["estimate"] == pytest.approx(
        world.truths["policy_value_target_distribution"]
    )
    assert row["realized_audit_fraction"] == pytest.approx(1.0)
