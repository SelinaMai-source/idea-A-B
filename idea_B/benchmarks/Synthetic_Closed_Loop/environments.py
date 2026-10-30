"""Synthetic environments for Feedback-Induced Observability Collapse.

B1 is an exogenous-missing diagnostic: observed accuracy can stay high while
weak-skill information vanishes, but the missingness schedule does not depend
on the policy. That is not endogenous collapse.

B2 is a true closed loop: the current parameters score queries, the policy
selects who is served, only served queries generate natural feedback, the
observed design Gramian and labels update the parameters, and the new
parameters become the next policy. Collapse is not simulated by externally
lowering a selection probability and then calling it policy-induced.

B3 splits into a recoverable loop (weak-skill queries remain in the
population and audits identify them) and an unrecoverable mechanism (the
weak direction is absent from the observed feature support, so extra budget
cannot restore it).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class LinearSkillEnvironment:
    """Shared linear skill population. Subclasses only change the observation channel."""

    name: str
    d: int = 3
    mix: tuple[float, ...] = (0.42, 0.42, 0.16)
    theta_star: tuple[float, ...] = (1.2, 1.05, 1.0)
    noise_std: float = 0.15
    feature_noise: float = 0.05
    population_per_round: int = 48
    weak_skill: int = 2
    policy_affects_feedback: bool = False
    audits_identify_weak_skill: bool = True
    collapse_type: str = "none"
    description: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mix = np.asarray(self.mix, dtype=float)
        if mix.shape[0] != self.d:
            raise ValueError("mix length must equal d")
        if mix.shape[0] < 2 or mix.shape[0] > 5:
            raise ValueError("skill dimension must be in 2..5")
        if np.any(mix < 0) or float(np.sum(mix)) <= 0.0:
            raise ValueError("mix must be a non-negative probability vector")
        self.mix_array = mix / float(np.sum(mix))
        self.theta_star_array = np.asarray(self.theta_star, dtype=float)
        if self.theta_star_array.shape[0] != self.d:
            raise ValueError("theta_star length must equal d")
        if not 0 <= self.weak_skill < self.d:
            raise ValueError("weak_skill must be a valid coordinate")

    def reset(self, rng: np.random.Generator) -> None:
        _ = rng

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "d": self.d,
            "mix": self.mix_array.tolist(),
            "theta_star": self.theta_star_array.tolist(),
            "noise_std": self.noise_std,
            "feature_noise": self.feature_noise,
            "population_per_round": self.population_per_round,
            "weak_skill": self.weak_skill,
            "policy_affects_feedback": self.policy_affects_feedback,
            "audits_identify_weak_skill": self.audits_identify_weak_skill,
            "collapse_type": self.collapse_type,
            "description": self.description,
            "extra": dict(self.extra),
        }

    def sample_batch(self, rng: np.random.Generator, t: int) -> dict[str, np.ndarray]:
        n = int(self.population_per_round)
        skills = rng.choice(self.d, size=n, p=self.mix_array)
        phi = np.zeros((n, self.d), dtype=float)
        phi[np.arange(n), skills] = 1.0
        if self.feature_noise > 0.0:
            phi = phi + self.feature_noise * rng.normal(size=(n, self.d))
        y = phi @ self.theta_star_array + self.noise_std * rng.normal(size=n)
        identifiable = np.ones(n, dtype=bool)
        return {
            "phi": phi,
            "y": y,
            "skills": skills,
            "identifiable": identifiable,
            "t": np.asarray(t),
        }

    def natural_observation_prob(
        self,
        phi: np.ndarray,
        skills: np.ndarray,
        theta: np.ndarray,
        t: int,
    ) -> np.ndarray:
        raise NotImplementedError

    def apply_audit_identifiability(
        self,
        batch: dict[str, np.ndarray],
        audit_indicator: np.ndarray,
    ) -> np.ndarray:
        audit = np.asarray(audit_indicator, dtype=bool).reshape(-1)
        if self.audits_identify_weak_skill:
            return audit
        identifiable = np.asarray(batch["identifiable"], dtype=bool).reshape(-1)
        return audit & identifiable


class B1ExogenousMissing(LinearSkillEnvironment):
    """Observed accuracy stays high while weak-skill information is externally starved."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "B1")
        kwargs.setdefault("policy_affects_feedback", False)
        kwargs.setdefault("audits_identify_weak_skill", True)
        kwargs.setdefault("collapse_type", "exogenous_missing")
        kwargs.setdefault(
            "description",
            "Exogenous missingness diagnostic: weak-skill observation rate decays "
            "on a calendar schedule that does not depend on the deployed parameters.",
        )
        super().__init__(**kwargs)
        extra = dict(self.extra)
        extra.setdefault("strong_rate", 0.45)
        extra.setdefault("weak_rate0", 0.20)
        extra.setdefault("weak_decay", 0.82)
        extra.setdefault("weak_floor", 0.0)
        self.extra = extra

    def natural_observation_prob(
        self,
        phi: np.ndarray,
        skills: np.ndarray,
        theta: np.ndarray,
        t: int,
    ) -> np.ndarray:
        _ = phi, theta
        skills_i = np.asarray(skills, dtype=int).reshape(-1)
        strong = float(self.extra["strong_rate"])
        weak0 = float(self.extra["weak_rate0"])
        decay = float(self.extra["weak_decay"])
        floor = float(self.extra["weak_floor"])
        weak_rate = max(floor, weak0 * (decay ** int(t)))
        q = np.full(skills_i.shape[0], strong, dtype=float)
        q[skills_i == self.weak_skill] = weak_rate
        return q


class B2PolicyInducedCollapse(LinearSkillEnvironment):
    """True endogenous loop: policy scores -> who is served -> labels -> update."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "B2")
        kwargs.setdefault("policy_affects_feedback", True)
        kwargs.setdefault("audits_identify_weak_skill", True)
        kwargs.setdefault("collapse_type", "endogenous_closed_loop")
        kwargs.setdefault(
            "description",
            "Policy-induced feedback collapse. Natural labels are collected only "
            "for queries the current parameters choose to serve (highest predicted "
            "quality). Parameters update from those labels and become the next policy.",
        )
        super().__init__(**kwargs)
        extra = dict(self.extra)
        extra.setdefault("serve_fraction", 0.35)
        extra.setdefault("logit_temperature", 0.0)
        self.extra = extra

    def natural_observation_prob(
        self,
        phi: np.ndarray,
        skills: np.ndarray,
        theta: np.ndarray,
        t: int,
    ) -> np.ndarray:
        _ = skills, t
        features = np.asarray(phi, dtype=float)
        n = features.shape[0]
        yhat = features @ np.asarray(theta, dtype=float).reshape(-1)
        temperature = float(self.extra.get("logit_temperature", 0.0))
        if temperature > 0.0:
            logits = yhat / temperature
            logits = logits - np.max(logits)
            weights = np.exp(logits)
            probs = weights / (float(np.sum(weights)) + 1e-12)
            scale = float(self.extra["serve_fraction"]) * n
            return np.clip(probs * scale, 0.0, 1.0)
        k = max(1, int(round(float(self.extra["serve_fraction"]) * n)))
        k = min(k, n)
        chosen = np.argsort(yhat)[-k:]
        q = np.zeros(n, dtype=float)
        q[chosen] = 1.0
        return q


class B3Recoverable(B2PolicyInducedCollapse):
    """Same loop as B2: the weak skill remains in the population and audits identify it."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "B3_recoverable")
        kwargs.setdefault("collapse_type", "recoverable")
        kwargs.setdefault(
            "description",
            "Recoverable information loss. The closed loop can starve the weak "
            "skill, but that skill remains in the population and audits restore "
            "the corresponding column of the observed design Gramian.",
        )
        super().__init__(**kwargs)


class B3Unrecoverable(B2PolicyInducedCollapse):
    """Weak-skill coordinate is structurally missing from observed features."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "B3_unrecoverable")
        kwargs.setdefault("audits_identify_weak_skill", False)
        kwargs.setdefault("collapse_type", "unrecoverable")
        kwargs.setdefault(
            "description",
            "Unrecoverable information loss. Weak-skill queries are projected off "
            "the missing coordinate, so neither natural feedback nor audits can "
            "identify that skill parameter.",
        )
        super().__init__(**kwargs)

    def sample_batch(self, rng: np.random.Generator, t: int) -> dict[str, np.ndarray]:
        batch = super().sample_batch(rng, t)
        phi = np.array(batch["phi"], dtype=float, copy=True)
        skills = np.asarray(batch["skills"], dtype=int)
        weak = skills == self.weak_skill
        phi[weak, self.weak_skill] = 0.0
        identifiable = ~weak
        batch["phi"] = phi
        batch["identifiable"] = identifiable
        return batch

    def apply_audit_identifiability(
        self,
        batch: dict[str, np.ndarray],
        audit_indicator: np.ndarray,
    ) -> np.ndarray:
        audit = np.asarray(audit_indicator, dtype=bool).reshape(-1)
        identifiable = np.asarray(batch["identifiable"], dtype=bool).reshape(-1)
        return audit & identifiable


class FixedPolicySamplingDiagnostic(B1ExogenousMissing):
    """Exogenous missingness with a frozen policy. Must not be labeled endogenous."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("name", "fixed_policy_fixed_sampling")
        kwargs.setdefault("collapse_type", "fixed_policy_fixed_sampling")
        kwargs.setdefault(
            "description",
            "Diagnostic: a frozen parameter vector and a calendar missingness "
            "schedule. Information can vanish, but this is not endogenous collapse.",
        )
        super().__init__(**kwargs)


ENVIRONMENTS = {
    "B1": B1ExogenousMissing,
    "B2": B2PolicyInducedCollapse,
    "B3_recoverable": B3Recoverable,
    "B3_unrecoverable": B3Unrecoverable,
    "fixed_policy_fixed_sampling": FixedPolicySamplingDiagnostic,
}


def make_environment(name: str, **kwargs: Any) -> LinearSkillEnvironment:
    if name not in ENVIRONMENTS:
        raise KeyError(f"unknown environment {name!r}")
    return ENVIRONMENTS[name](**kwargs)
