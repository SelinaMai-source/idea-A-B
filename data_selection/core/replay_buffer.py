"""Candidate buffer. Failed rollouts and hindsight labels stay out of imitation data."""

from __future__ import annotations

from core.types import EXPERT_QUERY_PROVENANCE, IMITATION_PROVENANCE, Trajectory


class ReplayBuffer:
    def __init__(self, *, allow_hindsight: bool = False) -> None:
        self.allow_hindsight = allow_hindsight
        self.imitation: list[Trajectory] = []
        self.held_out: list[Trajectory] = []
        self.expert_query_count = 0

    def add(self, traj: Trajectory) -> str:
        provenance = traj.label_provenance
        if provenance == "failed_rollout":
            self.held_out.append(traj)
            return "held_out_failed"
        if provenance == "hindsight_relabel":
            if not self.allow_hindsight:
                self.held_out.append(traj)
                return "held_out_hindsight"
            self.imitation.append(traj)
            return "accepted_hindsight"
        if provenance not in IMITATION_PROVENANCE:
            raise ValueError(f"unknown provenance {provenance}")
        self.imitation.append(traj)
        if provenance in EXPERT_QUERY_PROVENANCE:
            self.expert_query_count += 1
        return "accepted"
