import sys

from core.official import OfficialMethodAdapter
from core.selection_budget import blocked_result


class CupidSelector(OfficialMethodAdapter):
    name = "cupid"
    source_name = "cupid"
    selects = "episode"
    allowed_information = ("policy_checkpoint", "rollout_feedback", "influence_estimates")
    entry_candidates = (
        "eval_demonstration_scores.py",
        "train_trak_diffusion.py",
        "finalize_trak.py",
    )
    dependency_modules = ("torch", "hydra", "click")

    def protocol_block(self, context):
        meta = context.dataset_metadata or {}
        if meta.get("score_source") == "random":
            return "refusing random scores in place of CUPID influence"
        return None

    def missing_inputs(self, context):
        meta = context.dataset_metadata or {}
        required = ("exp_name", "eval_dir", "train_dir", "train_ckpt", "result_date")
        return [key for key in required if not meta.get(key)]

    def official_command(self, context):
        meta = context.dataset_metadata or {}
        if self.missing_inputs(context):
            return None
        script = self.checkout() / "eval_demonstration_scores.py"
        return [
            sys.executable,
            str(script),
            "--exp_name",
            str(meta["exp_name"]),
            "--eval_dir",
            str(meta["eval_dir"]),
            "--train_dir",
            str(meta["train_dir"]),
            "--train_ckpt",
            str(meta["train_ckpt"]),
            "--result_date",
            str(meta["result_date"]),
            "--device",
            str(meta.get("device") or "cuda:0"),
            "--seed",
            str(context.seed),
            "--eval_offline_policy_loss",
            "False",
            "--eval_offline_action_diversity",
            "False",
            "--eval_offline_state_diversity",
            "False",
            "--eval_online_state_similarity",
            "False",
            "--eval_online_demo_score",
            "False",
            "--eval_online_trak_influence",
            "True",
        ]

    def invoke_author(self, trajectories, budget, context):
        import subprocess

        command = self.official_command(context)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        tail = ((completed.stderr or "") + (completed.stdout or ""))[-2000:]
        if completed.returncode != 0:
            return self._blocked(
                f"eval_demonstration_scores.py exited {completed.returncode}: {tail}",
                invoked=True,
                entry="eval_demonstration_scores.py",
            )
        return self._blocked(
            "eval_demonstration_scores.py exited 0 but this adapter does not invent scores "
            "from unstructured logs. Point dataset_metadata['author_scores'] at the author file.",
            invoked=True,
            entry="eval_demonstration_scores.py",
            extra={"returncode": 0},
        )


SELECTOR = CupidSelector()
