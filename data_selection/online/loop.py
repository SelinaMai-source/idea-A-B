"""Explicit evaluate → collect → label → select → update → persist loop.

The synthetic driver does not train Diffusion Policy. Optimizer steps stay at 0.
"""

from __future__ import annotations

import random
import time

from core.replay_buffer import ReplayBuffer
from core.seeds import set_seed
from core.status import SYNTHETIC_INTERFACE_SMOKE
from core.trajectories import make_synthetic_trajectory
from core.types import SelectionBudget, SelectionContext
from evaluation.metrics import final_success_rate, success_auc


PHASES = ("evaluate", "collect", "label", "select", "update", "persist")


class EvaluationScheduler:
    def __init__(self, steps: list[int]) -> None:
        if list(steps) != sorted(set(steps)) or len(steps) < 2:
            raise ValueError("evaluation steps must be strictly increasing and non-empty")
        self.steps = list(steps)

    def due(self, env_step: int) -> bool:
        return env_step in self.steps


def _synthetic_success(seed: int, round_index: int, episode_index: int) -> bool:
    draw = random.Random(f"{seed}:{round_index}:{episode_index}").random()
    return draw < min(0.9, 0.25 + 0.05 * round_index)


class OnlineLoop:
    def run(
        self,
        selector,
        *,
        seed: int,
        rounds: int,
        episodes_per_round: int,
        transitions_per_episode: int,
        eval_steps: list[int],
        budget: SelectionBudget,
    ) -> dict:
        if rounds < 1 or episodes_per_round < 1:
            raise ValueError("rounds and episodes_per_round must be positive")
        set_seed(seed)
        scheduler = EvaluationScheduler(eval_steps)
        buffer = ReplayBuffer(allow_hindsight=False)
        curve = []
        phases = []
        env_steps = 0
        started = time.perf_counter()
        rollout_seconds = 0.0
        select_seconds = 0.0
        last_selection = None
        for round_index in range(rounds):
            env_steps += episodes_per_round * transitions_per_episode
            if not scheduler.due(env_steps) and env_steps not in eval_steps:
                # Synthetic grid is the cumulative step after each round.
                # Callers must pass that grid. A mismatch is a protocol error.
                pass
            if env_steps not in scheduler.steps and round_index == rounds - 1:
                raise ValueError("final environment step is not on the predeclared grid")
            outcomes = [
                _synthetic_success(seed, round_index, episode_index)
                for episode_index in range(episodes_per_round)
            ]
            if env_steps in scheduler.steps:
                curve.append((env_steps, final_success_rate(outcomes)))
            phases.append("evaluate")
            tick = time.perf_counter()
            fresh = []
            for episode_index, success in enumerate(outcomes):
                origin = "successful_self_rollout" if success else "failed_rollout"
                fresh.append(
                    make_synthetic_trajectory(
                        f"r{round_index}_e{episode_index}",
                        transitions_per_episode,
                        success=success,
                        origin=origin,
                        collected_step=env_steps,
                        dataset_id="synthetic_online",
                    )
                )
            if round_index == 0:
                fresh.append(
                    make_synthetic_trajectory(
                        "r0_external_expert",
                        transitions_per_episode,
                        success=True,
                        origin="external_expert",
                        collected_step=env_steps,
                        dataset_id="synthetic_online_expert",
                    )
                )
            rollout_seconds += time.perf_counter() - tick
            phases.append("collect")
            for traj in fresh:
                buffer.add(traj)
            phases.append("label")
            tick = time.perf_counter()
            last_selection = selector.select(
                buffer.imitation,
                budget,
                SelectionContext(seed=seed + round_index, hindsight_labels={"forbidden": True}),
            )
            select_seconds += time.perf_counter() - tick
            phases.append("select")
            phases.append("update")
            phases.append("persist")
        elapsed = time.perf_counter() - started
        if [step for step, _rate in curve] != list(eval_steps):
            raise ValueError(
                "synthetic curve does not match eval_steps; set eval_steps to the "
                "cumulative environment steps of each round"
            )
        auc = success_auc(curve, eval_steps)
        return {
            "status": SYNTHETIC_INTERFACE_SMOKE,
            "protocol": "online_closed_loop_synthetic",
            "policy_model": "NONE_SYNTHETIC_NO_DP_UPDATE",
            "adaptation_status": "SYNTHETIC_INTERFACE_SMOKE",
            "phases": phases,
            "expected_phase_cycle": list(PHASES),
            "seed": seed,
            "final_success_rate": curve[-1][1],
            "success_auc": auc,
            "success_curve": curve,
            "environment_steps": env_steps,
            "training_optimizer_steps": 0,
            "selected_episodes": None if last_selection is None else last_selection.selected_episodes,
            "selected_transitions": None if last_selection is None else last_selection.selected_transitions,
            "selection_status": None if last_selection is None else last_selection.status,
            "expert_query_count": buffer.expert_query_count,
            "held_out": len(buffer.held_out),
            "imitation_pool": len(buffer.imitation),
            "hindsight_in_imitation": sum(
                1 for traj in buffer.imitation if traj.label_provenance == "hindsight_relabel"
            ),
            "selection_wallclock_seconds": select_seconds,
            "rollout_wallclock_seconds": rollout_seconds,
            "training_wallclock_seconds": 0.0,
            "total_wallclock_seconds": elapsed,
            "gpu_hours": "N/A",
            "cpu_hours": elapsed / 3600.0,
            "paper_result": False,
            "reason": "Synthetic interface smoke. Not a RoboMimic or LIBERO result and not a Diffusion Policy update.",
        }


class RealOnlineProtocol:
    """evaluate → collect → label → select → update → persist.

    The CLI keeps P2 BLOCKED until a real policy, simulator, and supervision
    signal are all true. A unit test may inject fakes; that result is not a
    paper number and does not copy the synthetic loop's optimizer count.
    """

    def run(
        self,
        *,
        seed: int,
        online_steps: int | None,
        policy_ready: bool,
        simulator_ready: bool,
        supervision_ready: bool,
        hooks: dict | None = None,
        resume_state: dict | None = None,
    ) -> dict:
        missing = []
        if not policy_ready:
            missing.append("policy")
        if not simulator_ready:
            missing.append("simulator")
        if not supervision_ready:
            missing.append("supervision")
        if missing or not hooks:
            return {
                "status": "BLOCKED",
                "protocol": "online_closed_loop",
                "paper_result": False,
                "training_optimizer_steps": None,
                "success_auc": "N/A",
                "final_success_rate": "NOT_RUN",
                "requested_online_steps": online_steps,
                "seed": seed,
                "missing": missing or ["hooks"],
                "phases_defined": list(PHASES),
                "phases_executed": [],
                "reason": (
                    "P2 online experiments stay BLOCKED until policy, simulator, "
                    "and supervision actually work"
                ),
            }
        state = dict(resume_state or {"completed_phases": [], "optimizer_steps": 0})
        executed = []
        for phase in PHASES:
            hook = hooks[phase]
            state = hook(state)
            executed.append(phase)
            state.setdefault("completed_phases", [])
            state["completed_phases"].append(phase)
        return {
            "status": "STATE_MACHINE_EXERCISED_WITH_FAKES" if hooks.get("fakes") else "RUNNING",
            "protocol": "online_closed_loop",
            "paper_result": False,
            "training_optimizer_steps": state.get("optimizer_steps"),
            "success_auc": "N/A",
            "final_success_rate": "NOT_RUN",
            "requested_online_steps": online_steps,
            "seed": seed,
            "missing": [],
            "phases_defined": list(PHASES),
            "phases_executed": executed,
            "state": state,
            "reason": "State machine ran the injected hooks. This is not a RoboMimic or LIBERO result.",
        }
