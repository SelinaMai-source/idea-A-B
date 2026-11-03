import random

from core.selection_budget import blocked_result, budget_is_explicit, guard_pool, result_from_ordered


class RandomSelector:
    name = "random"
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
                "random requires an explicit budget (max_episodes, max_transitions, or retention_ratio) and a seed",
                {"seed": context.seed},
            )
        rng = random.Random(context.seed)
        ordered = list(trajectories)
        rng.shuffle(ordered)
        return result_from_ordered(
            self.name,
            ordered,
            budget,
            ["trajectory_ids", "budget", "seed"],
            reason=f"control: Random({context.seed}) shuffle, then complete episodes inside the budget",
            extra_cost={"seed": context.seed, "rng": "random.Random"},
            pool=trajectories,
        )


SELECTOR = RandomSelector()
