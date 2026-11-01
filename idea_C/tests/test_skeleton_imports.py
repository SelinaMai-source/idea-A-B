"""Skeleton import tests. No Monte Carlo."""

from __future__ import annotations

from pathlib import Path

import idea_C
import idea_C.baselines.eclosure as eclosure
import idea_C.baselines.eps as eps
import idea_C.baselines.robust_bandit as robust_bandit
import idea_C.baselines.score as score
import idea_C.benchmarks.eps_gaussian as eps_gaussian
import idea_C.benchmarks.overlap_multiarm as overlap_multiarm
import idea_C.benchmarks.overlap_streaming as overlap_streaming
import idea_C.benchmarks.robust_gaussian as robust_gaussian
import idea_C.benchmarks.robust_heavytail as robust_heavytail
import idea_C.benchmarks.score_gaussian as score_gaussian
import idea_C.benchmarks.uai_local_dependence as uai_local_dependence


def test_package_version_is_skeleton() -> None:
    assert idea_C.__version__ == "0.0.0"


def test_baseline_packages_import() -> None:
    assert robust_bandit.__doc__
    assert score.__doc__
    assert eclosure.__doc__
    assert eps.__doc__


def test_benchmark_packages_import() -> None:
    assert robust_gaussian.__doc__
    assert robust_heavytail.__doc__
    assert eps_gaussian.__doc__
    assert score_gaussian.__doc__
    assert uai_local_dependence.__doc__
    assert overlap_multiarm.__doc__
    assert overlap_streaming.__doc__


def test_results_tree_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    for part in ("raw", "summaries", "figures"):
        assert (root / "results" / "v0" / part).is_dir()
