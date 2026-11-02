from core.official import OfficialMethodAdapter


class DemoScoreSelector(OfficialMethodAdapter):
    name = "demo_score"
    source_name = "demo_score"
    selects = "episode"
    allowed_information = ("successful_rollouts", "failed_rollouts")
    entry_candidates = ("demo_score/demo_score/train_trajwise_classifier.py", "demo_score/demo_score/dataset.py")


SELECTOR = DemoScoreSelector()
