"""AAAI 2026 robust bandit multiple testing algorithms.

Label: ``independent_reimplementation``. There is no official code.
``neilzxu/e_bmt`` is NeurIPS 2021 and is not this paper.

Methods
    Uniform, Mean-based (Algorithm 1), E-process-based (Algorithm 2),
    Algorithm 3 (heavy-tail e-process + Algorithm 2 sampling).

The contaminated non-robust e-process is an invalidity stress test only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from idea_C.baselines.robust_bandit.dgp import ContaminatedBandit
from idea_C.baselines.robust_bandit.eprocess import HeavyTailEProcess, NonRobustEProcess, RobustEProcess
from idea_C.baselines.robust_bandit.sampling import (
    argmax_eprocess,
    argmax_mean_lcb,
    eprocess_burnin_length,
    mean_based_schedule_t,
    next_uniform_unrejected,
)
from idea_C.shared.budget import SampleBudget
from idea_C.shared.ebh import ebh_rejections

CODE_STATUS = "independent_reimplementation"

MethodName = Literal[
    "uniform",
    "mean_based",
    "eprocess_based",
    "algo3_heavytail",
    "nonrobust_stress",
]


@dataclass
class BanditMTResult:
    arms: np.ndarray
    rewards: np.ndarray
    evalues_path: np.ndarray
    rejected_path: np.ndarray
    n_pulls: np.ndarray
    budget_spent: int
    method: str
    code_status: str = CODE_STATUS
    notes: list[str] = field(default_factory=list)

    @property
    def final_rejected(self) -> np.ndarray:
        if self.rejected_path.size == 0:
            return np.zeros(self.n_pulls.size, dtype=bool)
        return self.rejected_path[-1]


def _new_eprocess(
    method: MethodName,
    k: int,
    *,
    mu0: float,
    eps: float,
    sigma: float,
    p: float,
    kappa: float,
) -> Any:
    if method == "algo3_heavytail":
        return HeavyTailEProcess(k, mu0=mu0, eps=eps, p=p, kappa=kappa)
    if method == "nonrobust_stress":
        return NonRobustEProcess(k, mu0=mu0, sigma=sigma)
    return RobustEProcess(k, mu0=mu0, eps=eps, sigma=sigma)


def run_bandit_mt(
    env: ContaminatedBandit,
    *,
    method: MethodName = "eprocess_based",
    delta: float = 0.05,
    budget: int = 200,
    mu0: float = 0.0,
    sigma: float = 1.0,
    p: float = 1.85,
    kappa: float = 50.0,
    unit_batches: bool = False,
) -> BanditMTResult:
    """Run one anytime path. ``env.is_null`` is never read.

    ``unit_batches`` forces Algorithm 1 ``t(s) = s`` so unit tests can
    exercise the selection loop without the theoretical ``t(1)`` burn-in.
    Default is the paper schedule.
    """
    k = env.k
    eps = float(env.eps)
    purse = SampleBudget(int(budget))
    eproc = _new_eprocess(method, k, mu0=mu0, eps=eps, sigma=sigma, p=p, kappa=kappa)
    rejected = np.zeros(k, dtype=bool)
    n_pulls = np.zeros(k, dtype=int)
    n_epochs = np.ones(k, dtype=int)
    histories: list[list[float]] = [[] for _ in range(k)]

    arms: list[int] = []
    rewards: list[float] = []
    e_path: list[np.ndarray] = []
    r_path: list[np.ndarray] = []
    notes: list[str] = []
    if method == "nonrobust_stress":
        notes.append(
            "INVALIDITY STRESS TEST: non-robust e-process is not valid under "
            "contamination. Label NOT_COMPARABLE. Do not cite as a theorem refutation."
        )
    if method in {"eprocess_based", "algo3_heavytail"}:
        notes.append(
            "Algorithm 2 typesetting nests Update/e-BH under t>KB; we update "
            "every pull so burn-in produces evidence (needed for Figure 1)."
        )
    if method == "mean_based":
        notes.append(
            "Algorithm 1 arm rule: argmax LCB of α-trimmed mean "
            "(OCR said argmin μ̂+bonus; prose says pull the believed-best arm)."
        )
        if unit_batches:
            notes.append("unit_batches=True: t(s)=s test hook, not the paper schedule.")

    burnin_per_arm = eprocess_burnin_length(k, delta, eps)
    burnin_total = k * burnin_per_arm
    cursor: int | None = None

    def record(arm: int, reward: float) -> None:
        eproc.update(arm, reward)
        histories[arm].append(reward)
        n_pulls[arm] += 1
        arms.append(arm)
        rewards.append(reward)
        # e-BH after every observation (anytime). Mean-based still spends
        # the scheduled batch before choosing a new arm.
        mask = ebh_rejections(eproc.values, delta)
        rejected[:] = mask
        e_path.append(eproc.values.copy())
        r_path.append(rejected.copy())

    def pull(arm: int) -> bool:
        if rejected[arm] or not purse.can_spend(1):
            return False
        purse.spend(1)
        record(arm, env.observe(arm))
        return True

    if method == "mean_based" and not unit_batches:
        # Algorithm 1 initialization: each arm t(1) times.
        t1 = mean_based_schedule_t(1, k, eps, delta)
        for arm in range(k):
            for _ in range(t1):
                if not pull(arm):
                    break
        n_epochs[:] = 1

    while purse.remaining > 0 and not bool(np.all(rejected)):
        if method == "uniform" or method == "nonrobust_stress":
            arm = next_uniform_unrejected(len(arms) + 1, rejected, cursor)
            cursor = arm
            if not pull(arm):
                break
            continue

        if method in {"eprocess_based", "algo3_heavytail"}:
            t = len(arms) + 1
            if t <= burnin_total:
                arm = next_uniform_unrejected(t, rejected, cursor)
                cursor = arm
            else:
                arm = argmax_eprocess(eproc.values, rejected)
            if not pull(arm):
                break
            continue

        if method == "mean_based":
            arm = argmax_mean_lcb(histories, n_epochs, rejected, eps=eps, sigma=sigma)
            if unit_batches:
                n_take = 1
            else:
                nxt = int(n_epochs[arm]) + 1
                n_take = mean_based_schedule_t(nxt, k, eps, delta) - mean_based_schedule_t(
                    int(n_epochs[arm]), k, eps, delta
                )
                n_take = max(n_take, 1)
            pulled = 0
            for _ in range(n_take):
                if not pull(arm):
                    break
                pulled += 1
            if pulled == 0:
                break
            n_epochs[arm] += 1
            continue

        raise ValueError(f"unknown method {method}")

    purse.assert_not_overspent()
    env.assert_no_extra_generation(n_pulls)
    if not arms:
        empty_e = eproc.values.copy()
        empty_r = rejected.copy()
        return BanditMTResult(
            arms=np.zeros(0, dtype=int),
            rewards=np.zeros(0, dtype=float),
            evalues_path=empty_e.reshape(0, k),
            rejected_path=empty_r.reshape(0, k),
            n_pulls=n_pulls,
            budget_spent=purse.spent,
            method=method,
            notes=notes,
        )
    return BanditMTResult(
        arms=np.asarray(arms, dtype=int),
        rewards=np.asarray(rewards, dtype=float),
        evalues_path=np.vstack(e_path),
        rejected_path=np.vstack(r_path),
        n_pulls=n_pulls,
        budget_spent=purse.spent,
        method=method,
        notes=notes,
    )
