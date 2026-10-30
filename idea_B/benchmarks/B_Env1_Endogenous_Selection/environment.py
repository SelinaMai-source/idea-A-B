"""B-Env1: endogenous selection with a well-fit majority initialization.

True closed loop: policy scores the batch, only served queries produce
natural feedback, parameters update, and the new parameters become the next
policy. Majority skills start already close to truth so the weak-skill error
can still worsen. v1 started at 0, where early weak-skill error was already
near its maximum, so the joint collapse criterion could not fire.

Observed loss on the served majority can fall (or stay near the noise floor)
while weak-skill MSE rises, because ridge shrinkage plus missing weak-skill
labels pull that coordinate toward 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from idea_B.benchmarks.Synthetic_Closed_Loop.environments import B2PolicyInducedCollapse

WEAK_INIT_VALUE = 0.55


def protocol_skill_defaults(d: int) -> dict[str, Any]:
    dim = int(d)
    if dim == 3:
        return {
            "mix": (0.42, 0.42, 0.16),
            "theta_star": (1.2, 1.05, 1.0),
            "weak_skill": 2,
        }
    if dim == 2:
        return {
            "mix": (0.70, 0.30),
            "theta_star": (1.2, 1.0),
            "weak_skill": 1,
        }
    if dim == 4:
        return {
            "mix": (0.30, 0.30, 0.24, 0.16),
            "theta_star": (1.2, 1.1, 1.05, 1.0),
            "weak_skill": 3,
        }
    if dim == 5:
        return {
            "mix": (0.24, 0.22, 0.22, 0.16, 0.16),
            "theta_star": (1.2, 1.12, 1.08, 1.04, 1.0),
            "weak_skill": 4,
        }
    raise ValueError("skill dimension must be in 2..5")


class Env1EndogenousSelection(B2PolicyInducedCollapse):
    """Policy-induced serving with a majority-already-fit initialization."""

    def __init__(self, **kwargs: Any) -> None:
        d = int(kwargs.get("d", 3))
        defaults = protocol_skill_defaults(d)
        kwargs.setdefault("name", "B-Env1")
        kwargs.setdefault("d", d)
        kwargs.setdefault("mix", defaults["mix"])
        kwargs.setdefault("theta_star", defaults["theta_star"])
        kwargs.setdefault("weak_skill", defaults["weak_skill"])
        kwargs.setdefault("noise_std", 0.15)
        kwargs.setdefault("feature_noise", 0.05)
        kwargs.setdefault("population_per_round", 40)
        kwargs.setdefault("policy_affects_feedback", True)
        kwargs.setdefault("audits_identify_weak_skill", True)
        kwargs.setdefault("collapse_type", "endogenous_selection")
        kwargs.setdefault(
            "description",
            "Endogenous selection. Majority skills start near truth so the "
            "weak-skill error can worsen. Natural labels come only from queries "
            "the current parameters serve. Observed loss on the served set can "
            "fall while weak-skill MSE rises.",
        )
        super().__init__(**kwargs)
        extra = dict(self.extra)
        extra.setdefault("serve_fraction", 0.35)
        extra.setdefault("logit_temperature", 0.0)
        extra.setdefault("weak_init_value", WEAK_INIT_VALUE)
        extra.setdefault("majority_at_truth", True)
        extra["noise_floor"] = float(self.noise_std**2)
        self.extra = extra

    def default_theta0(self) -> np.ndarray:
        theta = np.array(self.theta_star_array, dtype=float, copy=True)
        theta[int(self.weak_skill)] = float(self.extra["weak_init_value"])
        return theta

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["default_theta0"] = self.default_theta0().tolist()
        payload["noise_floor"] = float(self.noise_std**2)
        payload["initialization"] = (
            "majority skills at theta_star; weak skill at weak_init_value so "
            "weak-skill error can increase"
        )
        return payload


def make_environment(**kwargs: Any) -> Env1EndogenousSelection:
    return Env1EndogenousSelection(**kwargs)
