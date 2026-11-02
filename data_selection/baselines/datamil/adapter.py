from core.official import OfficialMethodAdapter


class DatamilSelector(OfficialMethodAdapter):
    name = "datamil"
    source_name = "datamil"
    selects = "episode"
    allowed_information = ("datamodel_or_metagradient",)
    entry_candidates = ("scripts/perform_libero_dm.sh", "scripts/cotraining.py")


SELECTOR = DatamilSelector()
