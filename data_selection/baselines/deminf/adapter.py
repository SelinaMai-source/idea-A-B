from core.official import OfficialMethodAdapter


class DeminfSelector(OfficialMethodAdapter):
    name = "deminf"
    source_name = "deminf"
    selects = "episode"
    allowed_information = ("demonstration_information",)
    entry_candidates = ("scripts/quality/estimate_quality.py", "scripts/train.py")


SELECTOR = DeminfSelector()
