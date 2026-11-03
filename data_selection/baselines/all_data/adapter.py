from core.selection_budget import accounting, blocked_result, guard_pool


class AllDataSelector:
    name = "all_data"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_ids", "budget")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        _ = context
        pool_episodes = len(trajectories)
        pool_transitions = sum(item.n_transitions for item in trajectories)
        cap = budget.episode_cap(pool_episodes)
        if cap is not None and cap < pool_episodes:
            return blocked_result(
                self.name,
                "all_data uses the complete eligible pool. A smaller episode budget "
                "is not All Data; use another method name.",
                {"eligible_pool_episodes": pool_episodes, "granularity": "episode"},
            )
        if budget.max_transitions is not None and budget.max_transitions < pool_transitions:
            return blocked_result(
                self.name,
                "all_data uses every eligible transition. A smaller transition budget "
                "is not All Data and episodes are not cut to manufacture the budget.",
                {
                    "eligible_pool_episodes": pool_episodes,
                    "eligible_pool_transitions": pool_transitions,
                    "granularity": "episode",
                },
            )
        return accounting(
            self.name,
            trajectories,
            list(trajectories),
            granularity="episode",
            information_used=["trajectory_ids", "budget"],
            reason="control: complete eligible pool in arrival order, not an id-sorted prefix",
            extra={"seed": getattr(context, "seed", None)},
        )


SELECTOR = AllDataSelector()
