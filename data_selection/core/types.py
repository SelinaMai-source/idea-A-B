"""Typed trajectory and selection records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


PROVENANCE = (
    "successful_self_rollout",
    "failed_rollout",
    "corrected_demonstration",
    "hindsight_relabel",
    "external_expert",
)

IMITATION_PROVENANCE = {
    "successful_self_rollout",
    "corrected_demonstration",
    "external_expert",
}

EXPERT_QUERY_PROVENANCE = {
    "corrected_demonstration",
    "external_expert",
}


@dataclass
class Transition:
    t: int
    observation: Mapping
    action: tuple
    reward: float
    terminal: bool
    success: bool


@dataclass
class Trajectory:
    trajectory_id: str
    task_id: str
    episode_id: str
    transitions: list
    task_success: bool
    data_origin: str
    label_provenance: str
    policy_checkpoint_source: str | None
    dataset_id: str
    n_transitions: int
    score: float | None = None
    collected_step: int = 0
    observation_mode: str = "synthetic_vector"

    def truncated_copy(self, keep: int) -> "Trajectory":
        if keep <= 0 or keep > len(self.transitions):
            raise ValueError("keep must be a proper prefix of the episode")
        steps = []
        for index, step in enumerate(self.transitions[:keep]):
            steps.append(
                Transition(
                    t=index,
                    observation=step.observation,
                    action=step.action,
                    reward=step.reward,
                    terminal=index == keep - 1,
                    success=self.task_success if index == keep - 1 else False,
                )
            )
        return Trajectory(
            trajectory_id=self.trajectory_id,
            task_id=self.task_id,
            episode_id=self.episode_id,
            transitions=steps,
            task_success=self.task_success,
            data_origin=self.data_origin,
            label_provenance=self.label_provenance,
            policy_checkpoint_source=self.policy_checkpoint_source,
            dataset_id=self.dataset_id,
            n_transitions=keep,
            score=self.score,
            collected_step=self.collected_step,
            observation_mode=self.observation_mode,
        )


@dataclass
class SelectionBudget:
    max_episodes: int | None = None
    max_transitions: int | None = None
    retention_ratio: float | None = None

    def episode_cap(self, pool_size: int) -> int | None:
        caps: list[int] = []
        if self.max_episodes is not None:
            if self.max_episodes < 0:
                raise ValueError("max_episodes must be >= 0")
            caps.append(self.max_episodes)
        if self.retention_ratio is not None:
            if not 0.0 <= self.retention_ratio <= 1.0:
                raise ValueError("retention_ratio must be in [0, 1]")
            caps.append(int(self.retention_ratio * pool_size))
        if not caps:
            return None
        return min(caps)


@dataclass
class SelectionContext:
    seed: int
    policy_checkpoint: str | None = None
    target_demonstrations: list | None = None
    rollout_feedback: dict | None = None
    training_history: dict | None = None
    dataset_metadata: dict | None = None
    evaluation_budget: dict | None = None
    hindsight_labels: dict | None = None
    scores: dict | None = None
    score_provenance: str | None = None


@dataclass
class SelectionResult:
    method: str
    status: str
    reason: str
    selected_ids: list
    selected_episodes: int
    selected_transitions: int
    truncated: bool
    information_used: list
    extra_cost: dict = field(default_factory=dict)
    scores: dict | None = None
