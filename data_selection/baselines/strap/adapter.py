from core.official import OfficialMethodAdapter


class StrapSelector(OfficialMethodAdapter):
    name = "strap"
    source_name = "strap"
    selects = "subtrajectory"
    allowed_information = ("visual_embeddings", "retrieval_references")
    entry_candidates = (
        "strap/retrieval/retrieval.py",
        "strap/embedding/encode_datasets.py",
    )
    dependency_modules = ("torch",)

    def protocol_block(self, context):
        meta = context.dataset_metadata or {}
        if meta.get("use_author_stove_pot_example"):
            return (
                "refusing to run retrieval.get_args() stove-pot LIBERO example "
                "as a RoboMimic or unified Image-DP result"
            )
        return None

    def missing_inputs(self, context):
        meta = context.dataset_metadata or {}
        missing = []
        for key in ("task_dataset", "offline_dataset", "output_path", "action_chunk", "frame_stack"):
            if meta.get(key) in (None, ""):
                missing.append(key)
        if meta.get("embedding") not in (None, "", "DINOv2"):
            missing.append("embedding_must_be_DINOv2_or_unset")
        return missing

    def official_command(self, context):
        if self.missing_inputs(context):
            return None
        script = self.checkout() / "strap" / "retrieval" / "retrieval.py"
        meta = context.dataset_metadata or {}
        return [
            "python",
            str(script),
            "--model_key",
            "DINOv2",
            "--action_chunk",
            str(meta["action_chunk"]),
            "--frame_stack",
            str(meta["frame_stack"]),
            "--output_path",
            str(meta["output_path"]),
        ]

    def invoke_author(self, trajectories, budget, context):
        return self._blocked(
            "STRAP retrieval.py is importable, but this adapter does not call get_args(). "
            "A caller must supply RetrievalArgs for the requested task. "
            "Sub-trajectory boundaries, transition counts, and action chunks stay unfilled "
            "until that call returns.",
            invoked=False,
            entry="strap/retrieval/retrieval.py",
            extra={
                "granularity": "subtrajectory",
                "action_chunk": (context.dataset_metadata or {}).get("action_chunk"),
                "frame_stack": (context.dataset_metadata or {}).get("frame_stack"),
                "selected_transitions": "NOT_RUN",
            },
        )


SELECTOR = StrapSelector()
