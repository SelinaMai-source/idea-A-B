"""B-Env2: two latent states that are observationally equivalent under the policy.

The world is in state ``θ_A``. A second state ``θ_B`` agrees on every
coordinate in the current policy's observation support and differs on the
unaudited weak skill. Without an intervention the Gaussian observed-data
likelihood cannot separate them, so the minimax parameter-error lower bound
is ``||θ_A - θ_B|| / 2 > 0``. A randomized audit that reveals the missing
coordinate restores identifiability. If that coordinate is projected out of
every feature, including audits, the case is structural unrecoverable.

The identification argument is the observed Gaussian likelihood and Fisher
information in ``identification_test.py``, not a plot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from idea_B.benchmarks.B_Env1_Endogenous_Selection.environment import protocol_skill_defaults
from idea_B.benchmarks.Synthetic_Closed_Loop.environments import B2PolicyInducedCollapse

WEAK_INIT_VALUE = 0.0
LATENT_B_WEAK = -0.5


class Env2ObservationalEquivalence(B2PolicyInducedCollapse):
    """Policy support leaves one coordinate unidentified without audits."""

    def __init__(self, **kwargs: Any) -> None:
        unrecoverable = bool(kwargs.pop("unrecoverable", False))
        d = int(kwargs.get("d", 3))
        defaults = protocol_skill_defaults(d)
        weak = int(kwargs.get("weak_skill", defaults["weak_skill"]))
        theta_a = np.asarray(kwargs.get("theta_star", defaults["theta_star"]), dtype=float).reshape(-1)
        if theta_a.shape[0] != d:
            raise ValueError("theta_star length must equal d")
        theta_b = np.array(theta_a, dtype=float, copy=True)
        theta_b[weak] = float(LATENT_B_WEAK)
        kwargs.setdefault("name", "B-Env2-unrecoverable" if unrecoverable else "B-Env2")
        kwargs.setdefault("d", d)
        kwargs.setdefault("mix", defaults["mix"])
        kwargs.setdefault("theta_star", tuple(float(x) for x in theta_a))
        kwargs.setdefault("weak_skill", defaults["weak_skill"])
        kwargs.setdefault("noise_std", 0.15)
        # Exact observational equivalence uses one-hot features (no feature noise).
        # The protocol experiment default 0.05 is "near-identical" and can be set
        # by the caller. Identification unit tests use the exact case.
        kwargs.setdefault("feature_noise", 0.0)
        kwargs.setdefault("population_per_round", 40)
        kwargs.setdefault("policy_affects_feedback", True)
        kwargs.setdefault("audits_identify_weak_skill", not unrecoverable)
        kwargs.setdefault(
            "collapse_type",
            "structural_unrecoverable" if unrecoverable else "observational_equivalence",
        )
        kwargs.setdefault(
            "description",
            "Two latent states agree on the policy observation support and "
            "differ on the unaudited skill. Identification uses the observed "
            "Gaussian likelihood and Fisher information."
            + (
                " Structural unrecoverable: the weak coordinate is absent from every feature."
                if unrecoverable
                else " Randomized audit of the missing skill restores identifiability."
            ),
        )
        super().__init__(**kwargs)
        extra = dict(self.extra)
        extra.setdefault("serve_fraction", 0.35)
        extra.setdefault("logit_temperature", 0.0)
        extra.setdefault("weak_init_value", WEAK_INIT_VALUE)
        extra.setdefault("exclude_weak_from_natural", True)
        extra["theta_a"] = theta_a.tolist()
        extra["theta_b"] = theta_b.tolist()
        extra["unrecoverable"] = unrecoverable
        extra["identification_error_lower_bound"] = float(0.5 * np.linalg.norm(theta_a - theta_b))
        extra["noise_floor"] = float(self.noise_std**2)
        extra["proof"] = "gaussian_observed_likelihood_and_fisher"
        self.extra = extra
        self.theta_a = np.asarray(theta_a, dtype=float)
        self.theta_b = np.asarray(theta_b, dtype=float)
        self.unrecoverable = unrecoverable

    def default_theta0(self) -> np.ndarray:
        theta = np.array(self.theta_star_array, dtype=float, copy=True)
        theta[int(self.weak_skill)] = float(self.extra["weak_init_value"])
        return theta

    def sample_batch(self, rng: np.random.Generator, t: int) -> dict[str, np.ndarray]:
        batch = super().sample_batch(rng, t)
        if not self.unrecoverable:
            return batch
        phi = np.array(batch["phi"], dtype=float, copy=True)
        skills = np.asarray(batch["skills"], dtype=int)
        weak = skills == self.weak_skill
        phi[:, self.weak_skill] = 0.0
        identifiable = ~weak
        batch["phi"] = phi
        batch["identifiable"] = identifiable
        return batch

    def natural_observation_prob(
        self,
        phi: np.ndarray,
        skills: np.ndarray,
        theta: np.ndarray,
        t: int,
    ) -> np.ndarray:
        q = super().natural_observation_prob(phi, skills, theta, t)
        if bool(self.extra.get("exclude_weak_from_natural", True)):
            skills_i = np.asarray(skills, dtype=int).reshape(-1)
            q = np.array(q, dtype=float, copy=True)
            q[skills_i == self.weak_skill] = 0.0
        return q

    def metadata(self) -> dict[str, Any]:
        payload = super().metadata()
        payload["default_theta0"] = self.default_theta0().tolist()
        payload["theta_a"] = self.theta_a.tolist()
        payload["theta_b"] = self.theta_b.tolist()
        payload["identification_error_lower_bound"] = float(
            self.extra["identification_error_lower_bound"]
        )
        payload["unrecoverable"] = self.unrecoverable
        payload["proof"] = "gaussian_observed_likelihood_and_fisher"
        payload["noise_floor"] = float(self.noise_std**2)
        return payload


def make_environment(*, unrecoverable: bool = False, **kwargs: Any) -> Env2ObservationalEquivalence:
    return Env2ObservationalEquivalence(unrecoverable=unrecoverable, **kwargs)
