from core.official import OfficialMethodAdapter


class StrapSelector(OfficialMethodAdapter):
    name = "strap"
    source_name = "strap"
    selects = "subtrajectory"
    allowed_information = ("visual_embeddings", "retrieval_references")
    entry_candidates = ("strap/retrieval/retrieval.py", "strap/embedding/encode_datasets.py")


SELECTOR = StrapSelector()
