"""Static Estimation track: a sequential log, a static estimand, no outcome-driven π."""

from idea_A.benchmarks.Sequential_Closed_Loop.static_estimation.track import (
    TRACK_METRICS,
    TRACK_NAME,
    run_static_estimation,
)

__all__ = ["TRACK_NAME", "TRACK_METRICS", "run_static_estimation"]
