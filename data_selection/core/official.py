"""Wrappers for published methods. They never fall back to Random."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.selection_budget import guard_pool
from core.status import BLOCKED, SOURCE_UNAVAILABLE
from core.types import SelectionBudget, SelectionContext, SelectionResult, Trajectory

ROOT = Path(__file__).resolve().parents[1]


class OfficialMethodAdapter:
    name = ""
    source_name = ""
    kind = "published_method"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = True
    allowed_information: tuple = ()
    entry_candidates: tuple = ()

    def source_record(self) -> dict | None:
        path = ROOT / "manifests" / "source_repositories.yaml"
        if not path.is_file():
            return None
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for repo in payload.get("repositories") or []:
            if repo.get("name") == self.source_name:
                return repo
        return None

    def located_entrypoint(self) -> str | None:
        repo = self.source_record()
        if not repo or repo.get("clone_status") != "CLONED":
            return None
        base = ROOT / repo["local_path"]
        for relative in self.entry_candidates:
            if (base / relative).is_file():
                return relative
        return None

    def blocked_reason(self) -> tuple[str, str]:
        repo = self.source_record()
        if repo is None:
            return SOURCE_UNAVAILABLE, "source manifest has no record for this method"
        if repo.get("clone_status") != "CLONED":
            detail = repo.get("failure_reason") or "clone failed"
            return SOURCE_UNAVAILABLE, detail
        entry = self.located_entrypoint()
        sha = repo.get("commit_sha") or "UNKNOWN_SHA"
        if entry is None:
            return (
                BLOCKED,
                f"source cloned at {sha} but no audited entrypoint is present; "
                "the paper rule is not replaced by a heuristic",
            )
        return (
            BLOCKED,
            f"source cloned at {sha}; entrypoint {entry} is not executed because the "
            "official dependency stack is not installed and this adapter does not "
            "reimplement the paper rule",
        )

    def select(
        self,
        trajectories: list[Trajectory],
        budget: SelectionBudget,
        context: SelectionContext,
    ) -> SelectionResult:
        guard_pool(trajectories)
        _ = budget.episode_cap(len(trajectories))
        _ = context.seed
        status, reason = self.blocked_reason()
        return SelectionResult(
            method=self.name,
            status=status,
            reason=reason,
            selected_ids=[],
            selected_episodes=0,
            selected_transitions=0,
            truncated=False,
            information_used=[],
            extra_cost={
                "heuristic_fallback": False,
                "official_stack_invoked": False,
                "allowed_information": list(self.allowed_information),
            },
        )
