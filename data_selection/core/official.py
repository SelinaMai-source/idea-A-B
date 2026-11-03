"""Published-method adapters. Author code is invoked, or the call stays BLOCKED.

No paper name is implemented with a heuristic or with random scores.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml

from core.selection_budget import accounting, blocked_result, guard_pool, result_from_ordered
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
    dependency_modules: tuple = ()

    def source_record(self) -> dict | None:
        path = ROOT / "manifests" / "source_repositories.yaml"
        if not path.is_file():
            return None
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for repo in payload.get("repositories") or []:
            if repo.get("name") == self.source_name:
                return repo
        return None

    def checkout(self) -> Path | None:
        repo = self.source_record()
        if not repo:
            return None
        return ROOT / repo["local_path"]

    def located_entrypoint(self) -> str | None:
        repo = self.source_record()
        if not repo or repo.get("clone_status") != "CLONED":
            return None
        base = ROOT / repo["local_path"]
        for relative in self.entry_candidates:
            if (base / relative).is_file():
                return relative
        return None

    def protocol_block(self, context: SelectionContext) -> str | None:
        return None

    def missing_inputs(self, context: SelectionContext) -> list[str]:
        return []

    def official_command(self, context: SelectionContext) -> list[str] | None:
        return None

    def invoke_author(
        self,
        trajectories: list[Trajectory],
        budget: SelectionBudget,
        context: SelectionContext,
    ) -> SelectionResult:
        command = self.official_command(context)
        raise NotImplementedError(
            f"{self.name} invoke_author is not reachable without {command}"
        )

    def _blocked(self, reason: str, *, invoked: bool, entry: str | None, extra: dict | None = None) -> SelectionResult:
        cost = {
            "heuristic_fallback": False,
            "official_stack_invoked": invoked,
            "attempted_entrypoint": entry,
            "allowed_information": list(self.allowed_information),
        }
        if extra:
            cost.update(extra)
        return blocked_result(self.name, reason, cost)

    def _from_author_scores(
        self,
        trajectories: list[Trajectory],
        budget: SelectionBudget,
        context: SelectionContext,
        scores: dict,
        *,
        entry: str,
    ) -> SelectionResult:
        if self.selects != "episode":
            return self._blocked(
                f"{self.name} selects {self.selects}; episode ranking is not a substitute",
                invoked=True,
                entry=entry,
            )
        missing = [traj.trajectory_id for traj in trajectories if traj.trajectory_id not in scores]
        if missing:
            return self._blocked(
                f"author scores omitted {missing}; refusing to fill them",
                invoked=True,
                entry=entry,
            )
        ordered = sorted(trajectories, key=lambda item: (-float(scores[item.trajectory_id]), item.trajectory_id))
        result = result_from_ordered(
            self.name,
            ordered,
            budget,
            list(self.allowed_information),
            reason=f"ranked by author scores from {entry}",
            extra_cost={
                "official_stack_invoked": True,
                "attempted_entrypoint": entry,
                "score_source": entry,
                "allowed_information": list(self.allowed_information),
            },
            allow_partial=False,
            pool=trajectories,
        )
        result.scores = {key: float(scores[key]) for key in result.selected_ids}
        return result

    def select(
        self,
        trajectories: list[Trajectory],
        budget: SelectionBudget,
        context: SelectionContext,
        author_runner=None,
    ) -> SelectionResult:
        guard_pool(trajectories)
        repo = self.source_record()
        if repo is None:
            return self._blocked(
                "source manifest has no record for this method",
                invoked=False,
                entry=None,
            )
        if repo.get("clone_status") != "CLONED":
            detail = repo.get("failure_reason") or "clone failed"
            return self._blocked(detail, invoked=False, entry=None, extra={"status_override": SOURCE_UNAVAILABLE})
        entry = self.located_entrypoint()
        pin = repo.get("commit_sha") or "UNKNOWN_SHA"
        if entry is None:
            return self._blocked(
                f"source cloned at {pin} but no audited entrypoint is present",
                invoked=False,
                entry=None,
            )
        misuse = self.protocol_block(context)
        if misuse:
            return self._blocked(misuse, invoked=False, entry=entry)
        if author_runner is not None:
            produced = author_runner(trajectories, budget, context)
            if isinstance(produced, SelectionResult):
                produced.extra_cost["official_stack_invoked"] = True
                produced.extra_cost["heuristic_fallback"] = False
                produced.extra_cost["attempted_entrypoint"] = entry
                return produced
            return self._from_author_scores(trajectories, budget, context, produced, entry=entry)
        errors = []
        for module_name in self.dependency_modules:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - the import error is the blocker
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
        if errors:
            return self._blocked(
                f"source cloned at {pin}; entrypoint {entry} was not executed. "
                f"dependency import failed: {'; '.join(errors)}",
                invoked=False,
                entry=entry,
                extra={"dependency_error": "; ".join(errors)},
            )
        missing = self.missing_inputs(context)
        if missing:
            return self._blocked(
                f"entrypoint {entry} imports, but required author inputs are absent: {', '.join(missing)}",
                invoked=False,
                entry=entry,
                extra={"missing_inputs": missing},
            )
        try:
            return self.invoke_author(trajectories, budget, context)
        except Exception as exc:  # noqa: BLE001 - author failure stays BLOCKED
            return self._blocked(
                f"author entrypoint {entry} raised {type(exc).__name__}: {exc}",
                invoked=True,
                entry=entry,
            )


def apply_subtrajectory_budget(
    pieces: list[Trajectory],
    budget: SelectionBudget,
    *,
    method: str,
    entry: str,
    pool: list[Trajectory],
) -> SelectionResult:
    """Keep author-emitted sub-trajectories. Do not promote them to whole episodes."""
    return result_from_ordered(
        method,
        pieces,
        budget,
        ["visual_embeddings", "retrieval_references"],
        reason=f"sub-trajectories returned by {entry}",
        extra_cost={
            "official_stack_invoked": True,
            "attempted_entrypoint": entry,
            "granularity": "subtrajectory",
        },
        allow_partial=True,
        pool=pool,
    )
