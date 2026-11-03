import random

from core.selection_budget import accounting, blocked_result, budget_is_explicit, guard_pool
from core.types import SelectionBudget


class ReservoirSelector:
    """Algorithm R over the arrival stream. This is not a shuffle of the full pool."""

    name = "reservoir"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_ids", "budget", "seed")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        if not budget_is_explicit(budget):
            return blocked_result(
                self.name,
                "reservoir requires an explicit episode budget and a seed",
                {"seed": context.seed, "algorithm": "algorithm_R"},
            )
        stream_budget = SelectionBudget(
            max_episodes=budget.max_episodes,
            retention_ratio=budget.retention_ratio,
        )
        cap = stream_budget.episode_cap(len(trajectories))
        if cap is None:
            return blocked_result(
                self.name,
                "reservoir needs max_episodes or retention_ratio. A transition cap alone is not a reservoir size.",
                {"seed": context.seed, "algorithm": "algorithm_R"},
            )
        if cap == 0:
            return accounting(
                self.name,
                trajectories,
                [],
                granularity="episode",
                information_used=["trajectory_ids", "budget", "seed"],
                reason="control: Algorithm R with an empty reservoir",
                extra={"seed": context.seed, "algorithm": "algorithm_R"},
            )
        rng = random.Random(context.seed)
        reservoir = []
        for index, traj in enumerate(trajectories):
            if index < cap:
                reservoir.append(traj)
            else:
                choice = rng.randrange(index + 1)
                if choice < cap:
                    reservoir[choice] = traj
        transitions = sum(item.n_transitions for item in reservoir)
        if budget.max_transitions is not None and transitions > budget.max_transitions:
            return blocked_result(
                self.name,
                "reservoir kept complete episodes whose transitions exceed max_transitions. "
                "The episode is not cut and the sample is not replaced by a shuffle.",
                {
                    "seed": context.seed,
                    "algorithm": "algorithm_R",
                    "eligible_pool_episodes": len(trajectories),
                    "granularity": "episode",
                },
            )
        return accounting(
            self.name,
            trajectories,
            reservoir,
            granularity="episode",
            information_used=["trajectory_ids", "budget", "seed"],
            reason="control: Algorithm R in arrival order. Membership is not a shuffle of the full pool.",
            extra={"seed": context.seed, "algorithm": "algorithm_R", "reservoir_size": cap},
        )


SELECTOR = ReservoirSelector()
