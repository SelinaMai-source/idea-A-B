from core.selection_budget import guard_pool, result_from_ordered
from core.status import BLOCKED
from core.types import SelectionResult


class FixedScoreSelector:
    name = "fixed_score"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_score", "budget")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        scored = []
        for traj in trajectories:
            score = traj.score
            if score is None and context.scores is not None:
                score = context.scores.get(traj.trajectory_id)
            if score is None:
                return SelectionResult(
                    method=self.name,
                    status=BLOCKED,
                    reason="fixed_score requires an externally supplied score on every candidate",
                    selected_ids=[],
                    selected_episodes=0,
                    selected_transitions=0,
                    truncated=False,
                    information_used=[],
                    extra_cost={"heuristic_fallback": False},
                )
            scored.append((float(score), traj))
        scored.sort(key=lambda item: (-item[0], item[1].trajectory_id))
        ordered = [traj for _score, traj in scored]
        return result_from_ordered(
            self.name,
            ordered,
            budget,
            ["trajectory_score", "budget"],
            reason="control: rank by the supplied score; ties break on trajectory_id",
        )

SELECTOR = FixedScoreSelector()
