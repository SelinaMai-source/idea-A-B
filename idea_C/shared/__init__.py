"""Shared Idea C v0 utilities: e-BH, metrics, budget, seeds, serialization."""

from idea_C.shared.budget import SampleBudget
from idea_C.shared.ebh import clip_evalues, ebh_rejection_set, ebh_rejections, ebh_threshold_index
from idea_C.shared.metrics import fdp, fdr, power, samples_at_target_tpr, sup_fdp, tpp, tpr
from idea_C.shared.seeds import seed_everything
from idea_C.shared.serialize import TABLE_METRIC_COLUMNS, write_run
from idea_C.shared.stats import mean_std_ci

__all__ = [
    "SampleBudget",
    "TABLE_METRIC_COLUMNS",
    "clip_evalues",
    "ebh_rejection_set",
    "ebh_rejections",
    "ebh_threshold_index",
    "fdp",
    "fdr",
    "mean_std_ci",
    "power",
    "samples_at_target_tpr",
    "seed_everything",
    "sup_fdp",
    "tpp",
    "tpr",
    "write_run",
]
