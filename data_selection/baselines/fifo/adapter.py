from core.selection_budget import guard_pool, result_from_ordered


class FifoSelector:
    name = "fifo"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_ids", "collected_step", "budget")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        ordered = sorted(trajectories, key=lambda item: (-item.collected_step, item.trajectory_id))
        return result_from_ordered(
            self.name,
            ordered,
            budget,
            ["trajectory_ids", "collected_step", "budget"],
            reason="control: newest episodes first under the declared budget",
        )

SELECTOR = FifoSelector()
