from core.official import OfficialMethodAdapter


class DemoScoreSelector(OfficialMethodAdapter):
    name = "demo_score"
    source_name = "demo_score"
    selects = "episode"
    allowed_information = ("successful_rollouts", "failed_rollouts")
    entry_candidates = (
        "demo_score/demo_score/train_trajwise_classifier.py",
        "demo_score/demo_score/dataset.py",
    )
    dependency_modules = ("torch",)

    def protocol_block(self, context):
        meta = context.dataset_metadata or {}
        if meta.get("failed_actions_as_imitation"):
            return "refusing to turn failed actions into imitation labels"
        train_ids = set(meta.get("classifier_train_ids") or [])
        eval_ids = set(meta.get("eval_ids") or [])
        overlap = sorted(train_ids & eval_ids)
        if overlap:
            return (
                "refusing to leak classifier training episodes into evaluation: "
                + ",".join(overlap)
            )
        return None

    def missing_inputs(self, context):
        meta = context.dataset_metadata or {}
        missing = []
        if not meta.get("data_dir"):
            missing.append("data_dir")
        if not meta.get("data_root"):
            missing.append("data_root")
        if not meta.get("success_rollout_dir") or not meta.get("failure_rollout_dir"):
            missing.append("separate_success_and_failure_rollouts")
        return missing

    def official_command(self, context):
        if self.missing_inputs(context):
            return None
        meta = context.dataset_metadata or {}
        script = self.checkout() / "demo_score" / "demo_score" / "train_trajwise_classifier.py"
        return [
            "import",
            str(script),
            "train",
            str(meta["data_dir"]),
            str(meta["data_root"]),
        ]

    def invoke_author(self, trajectories, budget, context):
        import importlib.util

        meta = context.dataset_metadata or {}
        script = self.checkout() / "demo_score" / "demo_score" / "train_trajwise_classifier.py"
        spec = importlib.util.spec_from_file_location("demo_score_train_trajwise_classifier", script)
        if spec is None or spec.loader is None:
            return self._blocked(
                f"cannot load {script}",
                invoked=False,
                entry="demo_score/demo_score/train_trajwise_classifier.py",
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.train(
            meta["data_dir"],
            meta["data_root"],
            meta.get("run_name") or "demo_score_adapter",
        )
        return self._blocked(
            "train_trajwise_classifier.train returned without a score file. "
            "Refusing to rank episodes by a local heuristic.",
            invoked=True,
            entry="demo_score/demo_score/train_trajwise_classifier.py",
        )


SELECTOR = DemoScoreSelector()
