"""Fixed policy plus fixed sampling must not be labeled endogenous collapse."""

import numpy as np

from idea_B.benchmarks.Synthetic_Closed_Loop import make_environment
from idea_B.methods import classify_collapse, run_closed_loop


def _series(n: int, start: float, end: float) -> np.ndarray:
    return np.linspace(start, end, n)


def test_classify_rejects_fixed_policy_fixed_sampling_even_if_numbers_look_like_collapse():
    n = 20
    label = classify_collapse(
        policy_affects_feedback=True,
        theta_is_fixed=True,
        sampling_is_fixed=True,
        observed_performance=_series(n, 0.9, 0.89),
        lambda_min_over_t=_series(n, 0.4, 0.02),
        weak_skill_error=_series(n, 0.1, 1.2),
    )
    assert label["endogenous_collapse"] is False
    assert "fixed policy" in label["reason"]


def test_classify_rejects_exogenous_missingness():
    n = 20
    label = classify_collapse(
        policy_affects_feedback=False,
        theta_is_fixed=False,
        sampling_is_fixed=False,
        observed_performance=_series(n, 0.9, 0.89),
        lambda_min_over_t=_series(n, 0.4, 0.02),
        weak_skill_error=_series(n, 0.1, 1.2),
    )
    assert label["endogenous_collapse"] is False
    assert "does not depend on the policy" in label["reason"]


def test_classify_accepts_policy_loop_with_stable_observed_and_worsening_information():
    n = 24
    label = classify_collapse(
        policy_affects_feedback=True,
        theta_is_fixed=False,
        sampling_is_fixed=False,
        observed_performance=_series(n, 0.88, 0.90),
        lambda_min_over_t=_series(n, 0.5, 0.05),
        weak_skill_error=_series(n, 0.2, 1.0),
    )
    assert label["endogenous_collapse"] is True


def test_frozen_runner_is_not_labeled_endogenous_collapse():
    env = make_environment("fixed_policy_fixed_sampling", population_per_round=20)
    result = run_closed_loop(
        env,
        "FrozenPolicyUniformSampling",
        seed=0,
        budget=0.05,
        horizon=10,
    )
    assert result["method_meta"]["theta_is_fixed"] is True
    assert result["method_meta"]["sampling_is_fixed"] is True
    assert result["collapse"]["endogenous_collapse"] is False
    assert env.policy_affects_feedback is False
