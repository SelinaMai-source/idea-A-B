from core.selection_budget import blocked_result, budget_is_explicit, guard_pool, result_from_ordered


class FifoSelector:
    name = "fifo"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_ids", "collected_step", "budget")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        _ = context
        if not budget_is_explicit(budget):
            return blocked_result(
                self.name,
                "fifo requires an explicit budget. The retained end is the newest collected_step.",
                {"retained_end": "newest"},
            )
        ordered = sorted(trajectories, key=lambda item: (-item.collected_step, item.trajectory_id))
        return result_from_ordered(
            self.name,
            ordered,
            budget,
            ["trajectory_ids", "collected_step", "budget"],
            reason=(
                "control: FIFO retains the newest end (higher collected_step first). "
                "The oldest episodes are dropped. Whole episodes only."
            ),
            extra_cost={"retained_end": "newest", "dropped_end": "oldest"},
            pool=trajectories,
        )


SELECTOR = FifoSelector()
