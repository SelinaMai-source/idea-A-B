"""Oracle information dominates a budgeted baseline on the same feature stream."""

import numpy as np

from idea_B.benchmarks.Synthetic_Closed_Loop import make_environment
from idea_B.methods import lambda_min, run_closed_loop


def test_oracle_gramian_dominates_uniform_baseline():
    env_oracle = make_environment("B2", population_per_round=20)
    env_uniform = make_environment("B2", population_per_round=20)
    oracle = run_closed_loop(
        env_oracle,
        "FullFeedbackOracle",
        seed=0,
        budget=0.05,
        horizon=8,
    )
    uniform = run_closed_loop(
        env_uniform,
        "UniformExploration",
        seed=0,
        budget=0.05,
        horizon=8,
    )
    w_oracle = np.asarray(oracle["W_T"], dtype=float)
    w_uniform = np.asarray(uniform["W_T"], dtype=float)
    gap = np.linalg.eigvalsh(0.5 * ((w_oracle - w_uniform) + (w_oracle - w_uniform).T))
    assert float(np.min(gap)) >= -1e-8
    assert lambda_min(w_oracle) >= lambda_min(w_uniform) - 1e-8
    assert float(np.trace(w_oracle)) >= float(np.trace(w_uniform)) - 1e-8
