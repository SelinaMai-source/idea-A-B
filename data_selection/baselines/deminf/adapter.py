import sys

from core.official import OfficialMethodAdapter


class DeminfSelector(OfficialMethodAdapter):
    name = "deminf"
    source_name = "deminf"
    selects = "episode"
    allowed_information = ("demonstration_information",)
    entry_candidates = ("scripts/quality/estimate_quality.py", "scripts/train.py")
    dependency_modules = ("jax", "tensorflow", "absl")

    def missing_inputs(self, context):
        meta = context.dataset_metadata or {}
        missing = []
        estimator = meta.get("estimator")
        if not estimator:
            missing.append("estimator")
        elif estimator == "random":
            missing.append("estimator_random_is_not_a_fallback")
        if not meta.get("dataset_path"):
            missing.append("dataset_path")
        return missing

    def protocol_block(self, context):
        meta = context.dataset_metadata or {}
        if meta.get("estimator") == "random" and meta.get("allow_author_random_estimator") is not True:
            return (
                "refusing estimator=random as a substitute for DemInf. "
                "The author objective is unchanged."
            )
        return None

    def official_command(self, context):
        meta = context.dataset_metadata or {}
        if self.missing_inputs(context):
            return None
        script = self.checkout() / "scripts" / "quality" / "estimate_quality.py"
        command = [
            sys.executable,
            str(script),
            "--estimator",
            str(meta["estimator"]),
            "--batch_size",
            str(meta.get("batch_size") or 1024),
        ]
        for flag in ("obs_ckpt", "action_ckpt", "obs_action_ckpt", "path"):
            if meta.get(flag):
                command.extend([f"--{flag}", str(meta[flag])])
        return command

    def invoke_author(self, trajectories, budget, context):
        import subprocess

        command = self.official_command(context)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        tail = ((completed.stderr or "") + (completed.stdout or ""))[-2000:]
        return self._blocked(
            "scripts/quality/estimate_quality.py was started and did not return a parsed "
            f"quality file (exit {completed.returncode}). {tail}",
            invoked=True,
            entry="scripts/quality/estimate_quality.py",
        )


SELECTOR = DeminfSelector()
