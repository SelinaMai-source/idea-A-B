from core.selection_budget import blocked_result, guard_pool, result_from_ordered


class FixedScoreSelector:
    name = "fixed_score"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_score", "score_provenance", "budget")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        provenance = (context.score_provenance or "").strip()
        if not provenance:
            return blocked_result(
                self.name,
                "fixed_score requires score provenance. Scores without a recorded source are rejected.",
            )
        scored = []
        for traj in trajectories:
            score = traj.score
            if score is None and context.scores is not None:
                score = context.scores.get(traj.trajectory_id)
            if score is None:
                return blocked_result(
                    self.name,
                    "fixed_score requires an externally supplied score on every candidate",
                    {"score_provenance": provenance},
                )
            scored.append((float(score), traj))
        scored.sort(key=lambda item: (-item[0], item[1].trajectory_id))
        ordered = [traj for _score, traj in scored]
        return result_from_ordered(
            self.name,
            ordered,
            budget,
            ["trajectory_score", "score_provenance", "budget"],
            reason="control: rank by the supplied score; ties break on trajectory_id",
            extra_cost={"score_provenance": provenance},
            pool=trajectories,
        )


SELECTOR = FixedScoreSelector()
