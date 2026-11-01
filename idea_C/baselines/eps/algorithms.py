"""e-PS Algorithm 1 + Uniform / Greedy (arXiv:2609.26651).

Label: ``independent_reimplementation``. Preprint, not published CCF-A.
Oracle LR and Adaptive plug-in are separate implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from idea_C.baselines.eps.dgp import EPSBandit
from idea_C.baselines.eps.eprocess import AdaptivePluginLR, OracleLR
from idea_C.baselines.eps.sampling import eps_posterior_sample, greedy_unrejected, uniform_unrejected
from idea_C.shared.budget import SampleBudget
from idea_C.shared.ebh import ebh_rejections

CODE_STATUS = "independent_reimplementation"

SamplerName = Literal["uniform", "greedy", "eps"]
EProcessName = Literal["oracle", "adaptive"]


@dataclass
class EPSResult:
    arms: np.ndarray
    rewards: np.ndarray
    evalues_path: np.ndarray
    rejected_path: np.ndarray
    n_pulls: np.ndarray
    budget_spent: int
    sampler: str
    eprocess: str
    code_status: str = CODE_STATUS
    notes: list[str] = field(default_factory=list)

    @property
    def final_rejected(self) -> np.ndarray:
        if self.rejected_path.size == 0:
            return np.zeros(self.n_pulls.size, dtype=bool)
        return self.rejected_path[-1]


def run_eps(
    env: EPSBandit,
    rng: np.random.Generator,
    *,
    sampler: SamplerName = "eps",
    eprocess: EProcessName = "oracle",
    alpha: float = 0.05,
    budget: int = 200,
) -> EPSResult:
    """One anytime path. ``env.theta`` / ``is_null`` are never read.

    Nested e-BH: only unrejected arms are sampled (Proposition 3.1).
    """
    k = env.k
    purse = SampleBudget(int(budget))
    if eprocess == "oracle":
        evidence = OracleLR(env.oracle_alternatives, family=env.family, student_df=env.student_df)
    elif eprocess == "adaptive":
        evidence = AdaptivePluginLR(k, family=env.family, student_df=env.student_df)
    else:
        raise ValueError(f"unknown eprocess {eprocess}")

    rejected = np.zeros(k, dtype=bool)
    n_pulls = np.zeros(k, dtype=int)
    histories: list[list[float]] = [[] for _ in range(k)]
    arms: list[int] = []
    rewards: list[float] = []
    e_path: list[np.ndarray] = []
    r_path: list[np.ndarray] = []
    notes = [
        f"e-process={eprocess} is a separate implementation from "
        f"{'adaptive' if eprocess == 'oracle' else 'oracle'}.",
        "v_{k,t}=1.1 * sample_variance(observations); n<2 uses variance=1.",
    ]

    t = 0
    while purse.remaining > 0 and not bool(np.all(rejected)):
        t += 1
        if sampler == "uniform":
            arm = uniform_unrejected(rejected, rng)
        elif sampler == "greedy":
            if t <= k and int(n_pulls.sum()) < k:
                # Give every arm one look so greedy e-values are defined.
                unseen = np.flatnonzero((n_pulls == 0) & ~rejected)
                arm = int(unseen[0]) if unseen.size else greedy_unrejected(evidence.values, rejected)
            else:
                arm = greedy_unrejected(evidence.values, rejected)
        elif sampler == "eps":
            if t <= k:
                arm = t - 1
                if rejected[arm]:
                    arm = uniform_unrejected(rejected, rng)
            else:
                arm = eps_posterior_sample(evidence.eproc.log_e, n_pulls, histories, rejected, rng)
        else:
            raise ValueError(f"unknown sampler {sampler}")

        if rejected[arm] or not purse.can_spend(1):
            break
        purse.spend(1)
        y = env.observe(arm)
        evidence.update(arm, y)
        histories[arm].append(y)
        n_pulls[arm] += 1
        rejected = ebh_rejections(evidence.values, alpha)
        arms.append(arm)
        rewards.append(y)
        e_path.append(evidence.values.copy())
        r_path.append(rejected.copy())

    purse.assert_not_overspent()
    env.assert_no_extra_generation(n_pulls)
    if not arms:
        return EPSResult(
            arms=np.zeros(0, dtype=int),
            rewards=np.zeros(0, dtype=float),
            evalues_path=np.zeros((0, k), dtype=float),
            rejected_path=np.zeros((0, k), dtype=bool),
            n_pulls=n_pulls,
            budget_spent=purse.spent,
            sampler=sampler,
            eprocess=eprocess,
            notes=notes,
        )
    return EPSResult(
        arms=np.asarray(arms, dtype=int),
        rewards=np.asarray(rewards, dtype=float),
        evalues_path=np.vstack(e_path),
        rejected_path=np.vstack(r_path),
        n_pulls=n_pulls,
        budget_spent=purse.spent,
        sampler=sampler,
        eprocess=eprocess,
        notes=notes,
    )
