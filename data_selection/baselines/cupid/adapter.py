from core.official import OfficialMethodAdapter


class CupidSelector(OfficialMethodAdapter):
    name = "cupid"
    source_name = "cupid"
    selects = "episode"
    allowed_information = ("policy_checkpoint", "rollout_feedback", "influence_estimates")
    entry_candidates = ("eval_demonstration_scores.py", "train_trak_diffusion.py", "finalize_trak.py")


SELECTOR = CupidSelector()
