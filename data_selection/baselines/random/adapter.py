import random

from core.selection_budget import guard_pool, result_from_ordered


class RandomSelector:
    name = "random"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_ids", "budget", "seed")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        rng = random.Random(context.seed)
        ordered = list(trajectories)
        rng.shuffle(ordered)
        return result_from_ordered(
            self.name,
            ordered,
            budget,
            ["trajectory_ids", "budget", "seed"],
            reason="control: seeded shuffle, then episode and transition caps",
        )

SELECTOR = RandomSelector()
