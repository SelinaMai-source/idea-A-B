from core.official import OfficialMethodAdapter


OCTO_LINE = "--config.pretrained_path=hf://rail-berkeley/octo-small"


class DatamilSelector(OfficialMethodAdapter):
    name = "datamil"
    source_name = "datamil"
    selects = "episode"
    allowed_information = ("datamodel_or_metagradient",)
    entry_candidates = ("scripts/perform_libero_dm.sh", "scripts/cotraining.py")
    dependency_modules = ()

    def protocol_block(self, context):
        _ = context
        checkout = self.checkout()
        script = checkout / "scripts" / "perform_libero_dm.sh" if checkout else None
        quoted = OCTO_LINE
        if script is not None and script.is_file():
            text = script.read_text(encoding="utf-8", errors="replace")
            if OCTO_LINE not in text:
                quoted = "the Octo pretrained path was not found at the expected line"
        return (
            "BLOCKED: scripts/perform_libero_dm.sh calls "
            "datamil/metagradients/do_libero_data_selection_parallel.py with "
            f"{quoted}. Octo scores are not image CNN Diffusion Policy scores. "
            "No faithful Diffusion Policy adaptation is implemented, so this adapter "
            "does not execute the Octo script and does not substitute a heuristic."
        )


SELECTOR = DatamilSelector()
