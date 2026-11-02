import random

from core.selection_budget import guard_pool, result_from_ordered
from core.types import SelectionBudget


class ReservoirSelector:
    name = "reservoir"
    kind = "control"
    leaderboard = "unified_image_dp"
    selects = "episode"
    paper_method = False
    allowed_information = ("trajectory_ids", "budget", "seed")

    def select(self, trajectories, budget, context):
        guard_pool(trajectories)
        cap = budget.episode_cap(len(trajectories))
        k = len(trajectories) if cap is None else cap
        rng = random.Random(context.seed)
        reservoir = []
        for index, traj in enumerate(trajectories):
            if index < k:
                reservoir.append(traj)
            else:
                choice = rng.randrange(index + 1)
                if choice < k:
                    reservoir[choice] = traj
        # Transition cap drops from the tail so the seed, not a second heuristic, decides membership.
        return result_from_ordered(
            self.name,
            reservoir,
            SelectionBudget(max_episodes=k, max_transitions=budget.max_transitions),
            ["trajectory_ids", "budget", "seed"],
            reason="control: Algorithm R over episodes, then the transition cap",
        )

SELECTOR = ReservoirSelector()
