"""B-Env3: recovery after a pre-specified midpoint latent change.

The change round is ``horizon // 2`` (16 when horizon is 32). The post-change
parameter is a function of the experiment seed only. It is drawn before any
method runs and does not depend on algorithm performance.

Recoverable: the weak skill stays in the feature support, so audits can
identify the new value. Unrecoverable: after the change the weak coordinate
is projected out, so neither natural feedback nor audits identify it.

Metrics (computed by the v1.1 runner): detection delay, recovery time,
excess MSE, worst-skill MSE, regret, cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from idea_B.benchmarks.B_Env1_Endogenous_Selection.environment import protocol_skill_defaults
from idea_B.benchmarks.Synthetic_Closed_Loop.environments import B2PolicyInducedCollapse

CHANGE_SEED_OFFSET = 20261031
WEAK_SHIFT = 1.4
WEAK_INIT_VALUE = 0.55


def predraw_theta_after(
    seed: int,
    theta_before: Any,
    weak_skill: int,
    *,
    shift: float = WEAK_SHIFT,
) -> np.ndarray:
    """Seed-determined post-change ``theta_star``. Independent of any method."""
    rng = np.random.default_rng(int(CHANGE_SEED_OFFSET) + int(seed) * 17)
    theta = np.asarray(theta_before, dtype=float).reshape(-1).copy()
    k = int(weak_skill)
    sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
    theta[k] = float(theta[k]) + sign * float(shift)
    return theta


class Env3RecoveryAfterLatentChange(B2PolicyInducedCollapse):
    """Midpoint change in the weak-skill reward, recoverable or not."""

    def __init__(self, **kwargs: Any) -> None:
        unrecoverable = bool(kwargs.pop("unrecoverable", False))
        seed = int(kwargs.pop("seed", 0))
        horizon = int(kwargs.pop("horizon", 32))
        d = int(kwargs.get("d", 3))
        defaults = protocol_skill_defaults(d)
        kwargs.setdefault("name", "B-Env3-unrecoverable" if unrecoverable else "B-Env3-recoverable")
        kwargs.setdefault("d", d)
        kwargs.setdefault("mix", defaults["mix"])
        kwargs.setdefault("theta_star", defaults["theta_star"])
        kwargs.setdefault("weak_skill", defaults["weak_skill"])
        kwargs.setdefault("noise_std", 0.15)
        kwargs.setdefault("feature_noise", 0.05)
        kwargs.setdefault("population_per_round", 40)
        kwargs.setdefault("policy_affects_feedback", True)
        kwargs.setdefault("audits_identify_weak_skill", not unrecoverable)
        kwargs.setdefault(
            "collapse_type",
            "unrecoverable_change" if unrecoverable else "recoverable_change",
        )
        kwargs.setdefault(
            "description",
            "Latent change at a pre-specified midpoint independent of algorithm "
            "performance. "
            + (
                "After the change the weak coordinate is structurally missing."
                if unrecoverable
                else "The weak skill remains identifiable, so audits can restore it."
            ),
        )
        super().__init__(**kwargs)
        extra = dict(self.extra)
        extra.setdefault("serve_fraction", 0.35)
        extra.setdefault("logit_temperature", 0.0)
        extra.setdefault("weak_init_value", WEAK_INIT_VALUE)
        extra["unrecoverable"] = unrecoverable
        extra["seed"] = seed
        extra["horizon"] = horizon
        extra["change_round"] = horizon // 2
        extra["noise_floor"] = float(self.noise_std**2)
        self.extra = extra
        self.unrecoverable = unrecoverable
        self.env_seed = seed
        self.horizon = horizon
        self.change_round = int(horizon // 2)
        self.theta_star_before = np.array(self.theta_star_array, dtype=float, copy=True)
        self.theta_star_after = predraw_theta_after(
            seed, self.theta_star_before, self.weak_skill
        )
        extra["theta_star_before"] = self.theta_star_before.tolist()
        extra["theta_star_after"] = self.theta_star_after.tolist()
        self.theta_star_array = np.array(self.theta_star_before, dtype=float, copy=True)
        self._project_weak = False

    def reset(self, rng: np.random.Generator) -> None:
        _ = rng
        self.theta_star_array = np.array(self.theta_star_before, dtype=float, copy=True)
        self._project_weak = False

    def default_theta0(self) -> np.ndarray:
        theta = np.array(self.theta_star_before, dtype=float, copy=True)
        theta[int(self.weak_skill)] = float(self.extra["weak_init_value"])
        return theta

    def _set_phase(self, t: int) -> None:
        if int(t) >= int(self.change_round):
            self.theta_star_array = np.array(self.theta_star_after, dtype=float, copy=True)
            self._project_weak = bool(self.unrecoverable)
        else:
            self.theta_star_array = np.array(self.theta_star_before, dtype=float, copy=True)
            self._project_weak = False

    def sample_batch(self, rng: np.random.Generator, t: int) -> dict[str, np.ndarray]:
        self._set_phase(t)
        batch = super().sample_batch(rng, t)
        if not self._project_weak:
            return batch
        phi = np.array(batch["phi"], dtype=float, copy=True)
        skills = np.asarray(batch["skills"], dtype=int)
        weak = skills == self.weak_skill
        phi[:, self.weak_skill] = 0.0
        batch["phi"] = phi
        batch["identifiable"] = ~weak
        return batch

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["default_theta0"] = self.default_theta0().tolist()
        payload["change_round"] = int(self.change_round)
        payload["theta_star_before"] = self.theta_star_before.tolist()
        payload["theta_star_after"] = self.theta_star_after.tolist()
        payload["unrecoverable"] = self.unrecoverable
        payload["change_depends_on_algorithm"] = False
        payload["noise_floor"] = float(self.noise_std**2)
        return payload


def make_environment(
    *,
    unrecoverable: bool = False,
    seed: int = 0,
    horizon: int = 32,
    **kwargs: Any,
) -> Env3RecoveryAfterLatentChange:
    return Env3RecoveryAfterLatentChange(
        unrecoverable=unrecoverable,
        seed=seed,
        horizon=horizon,
        **kwargs,
    )
