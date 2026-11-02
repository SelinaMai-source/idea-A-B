from core.selection_budget import guard_pool, result_from_ordered


class AllDataSelector:
    name = "all_data"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_ids", "budget")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        ordered = sorted(trajectories, key=lambda item: item.trajectory_id)
        return result_from_ordered(
            self.name,
            ordered,
            budget,
            ["trajectory_ids", "budget"],
            reason="control: id-sorted prefix under the declared budget; full pool when the budget covers it",
        )

SELECTOR = AllDataSelector()
