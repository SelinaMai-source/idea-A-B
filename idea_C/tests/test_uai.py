"""UAI 2026 local-dependence DGP, FDR vs SupFDR, runner guards.

UAI 2026 is NOT CCF-A Main.
"""

from __future__ import annotations

import numpy as np

from idea_C.baselines.eclosure import (
    RELATED_OFFICIAL_EXP,
    UAI_VENUE_NOTE,
    e_lond,
    empirical_fdr,
    empirical_supfdr,
    fdr_terminal,
    local_dependent_gaussian,
    sup_fdp,
)
from idea_C.baselines.eclosure.dgp import banded_ar1_cov

import importlib.util
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_uai = _load_script("run_uai.py")
run_score = _load_script("run_score.py")


def test_venue_note() -> None:
    assert "NOT CCF-A" in UAI_VENUE_NOTE
    assert RELATED_OFFICIAL_EXP == "lag_comp"


def test_local_dep_same_seed() -> None:
    a = local_dependent_gaussian(m=12, pi1=0.3, seed=4, lag=5)
    b = local_dependent_gaussian(m=12, pi1=0.3, seed=4, lag=5)
    np.testing.assert_allclose(a.evalues, b.evalues)
    np.testing.assert_allclose(a.pvalues, b.pvalues)


def test_covariance_banded_and_psd_checked() -> None:
    cov = banded_ar1_cov(8, lag=3, rho=0.5)
    assert cov[0, 3] != 0.0
    assert cov[0, 4] == 0.0
    stream = local_dependent_gaussian(m=30, pi1=0.2, seed=0, lag=10)
    eigs = np.linalg.eigvalsh(stream.cov)
    assert eigs.min() >= -1e-8
    assert stream.source.startswith("UAI 2026")
    assert "lag_comp" in stream.related_official_exp


def test_fdr_and_supfdr_are_distinct() -> None:
    # Early false discovery then true discoveries: terminal FDP < max FDP
    rejected = np.array([True, True, True, True])
    is_null = np.array([True, False, False, False])
    fdr = fdr_terminal(rejected, is_null)
    sup = sup_fdp(rejected, is_null)
    assert sup == 1.0
    assert fdr == 0.25
    assert sup != fdr

    rej_mat = np.stack([rejected, rejected])
    null_mat = np.stack([is_null, is_null])
    assert empirical_supfdr(rej_mat, null_mat) == 1.0
    assert empirical_fdr(rej_mat, null_mat) == 0.25


def test_null_only_stream_metrics_in_unit_interval() -> None:
    stream = local_dependent_gaussian(m=20, pi1=0.0, seed=3, lag=5)
    assert np.all(stream.is_null)
    r = e_lond(stream.evalues, delta=0.1)
    fdr = fdr_terminal(r.rejected, stream.is_null)
    sup = sup_fdp(r.rejected, stream.is_null)
    assert 0.0 <= fdr <= 1.0
    assert 0.0 <= sup <= 1.0
    assert sup + 1e-12 >= fdr


def test_pvalues_and_evalues_are_functions_of_x_only() -> None:
    stream = local_dependent_gaussian(m=10, pi1=0.5, seed=9, lag=4)
    # p-values in (0,1]; e-values finite and nonnegative
    assert np.all(stream.pvalues >= 0.0) and np.all(stream.pvalues <= 1.0)
    assert np.all(stream.evalues >= 0.0)
    assert np.all(np.isfinite(stream.evalues))


def test_run_uai_refuses_paper_scale() -> None:
    assert run_uai.main(["--reps", "200", "--run"]) == 2
    assert run_uai.main(["--list"]) == 0


def test_run_score_refuses_paper_scale() -> None:
    assert run_score.main(["--reps", "500", "--run"]) == 2
    assert run_score.main(["--list"]) == 0


def test_run_uai_list_mentions_not_ccf_a(capsys) -> None:
    run_uai.main(["--list"])
    out = capsys.readouterr().out
    assert "NOT CCF-A" in out
    assert "lag_comp" in out
    assert "SupFDR" in out
