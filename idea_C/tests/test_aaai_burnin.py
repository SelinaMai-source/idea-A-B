"""AAAI Algorithm 2 burn-in protocol. Do not shorten B; horizon must exceed KB."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from idea_C.baselines.robust_bandit.algorithms import run_bandit_mt
from idea_C.baselines.robust_bandit.dgp import LEVY_UNSPECIFIED_IN_PAPER, make_gaussian_instance
from idea_C.baselines.robust_bandit.sampling import algo2_burnin_budget, eprocess_burnin_length
from idea_C.shared.seeds import seed_everything

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
V01_CONFIGS = (
    "aaai_robust_gaussian_v0_1.json",
    "aaai_heavy_tailed_v0_1.json",
    "p1a_clean_post_burnin.json",
    "p1a_contaminated_post_burnin.json",
)


def test_paper_B_is_452_and_not_shortened() -> None:
    b, kb = algo2_burnin_budget(50, 0.05, 0.1)
    expect = int(math.ceil(4.0 / 0.1 * math.log(4.0 * 50 / (0.05**2))))
    assert expect == 452
    assert b == 452
    assert kb == 22600
    assert eprocess_burnin_length(50, 0.05, 0.1) == 452


def test_new_configs_horizon_strictly_exceeds_KB() -> None:
    for name in V01_CONFIGS:
        cfg = json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))
        burn = cfg["burnin"]
        assert burn["B_per_arm"] == 452
        assert burn["KB"] == 22600
        assert burn["do_not_shorten_B"] is True
        for scale, horizon in cfg["horizon"].items():
            assert int(horizon) > 22600, f"{name} {scale}={horizon} is not > KB"
        levy = (cfg.get("dgp") or {}).get("contamination")
        if isinstance(levy, dict) and levy.get("family") == "levy_stable":
            assert levy.get("unspecified_in_paper") is True


def test_v0_configs_keep_historical_horizons() -> None:
    p0 = json.loads((CONFIG_DIR / "aaai_robust_gaussian.json").read_text(encoding="utf-8"))
    p1 = json.loads((CONFIG_DIR / "p1a_clean.json").read_text(encoding="utf-8"))
    assert p0["horizon"]["pilot"] == 4000
    assert p1["horizon"]["pilot"] == 22600


def test_short_horizon_is_invalid_protocol() -> None:
    rng = seed_everything(0)
    env = make_gaussian_instance(4, 1, rng, eps=0.1)
    b, kb = algo2_burnin_budget(4, 0.05, 0.1)
    assert kb > 20
    result = run_bandit_mt(env, method="eprocess_based", delta=0.05, budget=20)
    assert result.burnin_per_arm == b
    assert result.burnin_total == kb
    assert result.n_post_burnin_pulls == 0
    assert result.entered_adaptive_phase is False
    assert result.protocol_status == "INVALID_PROTOCOL"
    assert any("INVALID_PROTOCOL" in note for note in result.notes)


def test_horizon_past_KB_records_post_burnin_frequencies() -> None:
    rng = seed_everything(1)
    env = make_gaussian_instance(3, 1, rng, mu_nonnull=3.0, eps=0.5)
    b, kb = algo2_burnin_budget(3, 0.05, 0.5)
    budget = kb + 9
    result = run_bandit_mt(env, method="eprocess_based", delta=0.05, budget=budget)
    assert result.burnin_per_arm == b
    assert result.protocol_status == "ok"
    assert result.entered_adaptive_phase is True
    assert result.n_post_burnin_pulls == 9
    assert int(result.post_burnin_counts.sum()) == 9
    np.testing.assert_allclose(result.post_burnin_frequencies.sum(), 1.0)
    assert result.post_burnin_counts.shape == (3,)


def test_uniform_is_not_invalidated_by_algo2_burnin() -> None:
    rng = seed_everything(2)
    env = make_gaussian_instance(4, 1, rng, eps=0.1)
    result = run_bandit_mt(env, method="uniform", delta=0.05, budget=12)
    assert result.protocol_status == "ok"
    assert result.n_post_burnin_pulls == 0


def test_levy_flag_remains_unspecified() -> None:
    assert LEVY_UNSPECIFIED_IN_PAPER is True
