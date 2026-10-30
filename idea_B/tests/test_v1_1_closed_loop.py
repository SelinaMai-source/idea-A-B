"""Closed-loop dependence: fixed policy plus fixed sampling is not endogenous collapse."""

import numpy as np

from idea_B.benchmarks.B_Env1_Endogenous_Selection import make_environment as make_env1
from idea_B.benchmarks.Synthetic_Closed_Loop import make_environment as make_v10_env
from idea_B.methods.v1_1 import classify_collapse, run_closed_loop
from shared.data_utils.seeds import seed_everything


def _series(n: int, start: float, end: float) -> np.ndarray:
    return np.linspace(start, end, n)


def test_env1_observation_channel_depends_on_policy():
    rng = seed_everything(0)
    env = make_env1(population_per_round=30)
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


def test_frozen_runner_is_not_labeled_endogenous_collapse():
    env = make_v10_env("fixed_policy_fixed_sampling", population_per_round=20)
    result = run_closed_loop(
        env,
        "FrozenPolicyUniformSampling",
        seed=0,
        budget=0.05,
        horizon=10,
        theta0=np.zeros(3),
    )
    assert result["method_meta"]["theta_is_fixed"] is True
    assert result["method_meta"]["sampling_is_fixed"] is True
    assert result["collapse"]["endogenous_collapse"] is False


def test_env1_default_theta0_leaves_room_for_weak_skill_error_to_worsen():
    env = make_env1()
    theta0 = env.default_theta0()
    truth = env.theta_star_array
    weak = int(env.weak_skill)
    majority = [k for k in range(env.d) if k != weak]
    for k in majority:
        assert abs(theta0[k] - truth[k]) <= 1e-12
    assert abs(theta0[weak] - truth[weak]) > 0.2
    assert abs(theta0[weak] - truth[weak]) < abs(0.0 - truth[weak]) - 1e-12


def test_env1_observed_mse_can_stay_low_while_weak_skill_error_rises():
    env = make_env1(population_per_round=40, feature_noise=0.0)
    theta0 = env.default_theta0()
    weak = int(env.weak_skill)
    start_err = abs(float(theta0[weak] - env.theta_star_array[weak]))
    result = run_closed_loop(
        env,
        "NoFeedbackController",
        seed=0,
        budget=0.05,
        horizon=12,
        theta0=theta0,
    )
    traj = result["trajectory"]
    final_err = abs(float(result["theta_final"][weak] - env.theta_star_array[weak]))
    late_obs = float(np.nanmean(traj["observed_mse"][-3:]))
    late_weak = float(np.nanmean(traj[f"skill_mse_{weak}"][-3:]))
    assert np.isfinite(late_obs) and np.isfinite(late_weak)
    # Majority remains well fit; observed loss stays near the noise floor.
    assert late_obs < 0.15
    assert late_obs < late_weak
    # Without weak-skill labels, ridge pulls that coordinate toward 0 and error rises.
    assert final_err > start_err + 0.1
    assert result["summary"]["noise_floor"] == float(env.noise_std**2)
