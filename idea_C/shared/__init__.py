"""Shared Idea C v0 utilities: e-BH, metrics, budget, seeds, serialization."""

from idea_C.shared.budget import SampleBudget
from idea_C.shared.ebh import clip_evalues, ebh_rejection_set, ebh_rejections, ebh_threshold_index
from idea_C.shared.metrics import (
    CENSORED,
    NOT_RUN,
    can_rank_by_conditional_mean,
    fdp,
    fdr,
    power,
    samples_at_target_metric_cells,
    samples_at_target_tpr,
    summarize_samples_at_target,
    sup_fdp,
    tpp,
    tpr,
)
from idea_C.shared.seeds import seed_everything
from idea_C.shared.serialize import TABLE_METRIC_COLUMNS, attach_censoring_metrics, write_run
from idea_C.shared.stats import mean_std_ci

__all__ = [
    "CENSORED",
    "NOT_RUN",
    "SampleBudget",
    "TABLE_METRIC_COLUMNS",
    "attach_censoring_metrics",
    "can_rank_by_conditional_mean",
    "clip_evalues",
    "ebh_rejection_set",
    "ebh_rejections",
    "ebh_threshold_index",
    "fdp",
    "fdr",
    "mean_std_ci",
    "power",
    "samples_at_target_metric_cells",
    "samples_at_target_tpr",
    "seed_everything",
    "summarize_samples_at_target",
    "sup_fdp",
    "tpp",
    "tpr",
    "write_run",
]
