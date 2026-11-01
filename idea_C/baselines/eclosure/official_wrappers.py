"""Wrappers around official clones. Do not modify the clone trees.

- neilzxu/eclosure SHA 8bbd5a37… is OFFLINE e-Closure (arXiv:2509.02517),
  NOT UAI 2026 online closed/donation e-LOND.
- neilzxu/evalue-omt SHA cda6e996… is AISTATS 2024 e-LOND (lag_comp / wor_comp).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from .result import (
    AISTATS_ELOND_STATUS,
    OFFLINE_ECLOSURE_STATUS,
    ProcedureResult,
)

_HERE = Path(__file__).resolve().parent
OFFICIAL_ECLOSURE = _HERE / "official"
EVALUE_OMT = _HERE / "evalue-omt"
EVALUE_OMT_SRC = EVALUE_OMT / "src"


def _methodtools_shim() -> None:
    if "methodtools" in sys.modules:
        return
    stub = types.ModuleType("methodtools")

    def lru_cache(*args, **kwargs):  # noqa: ANN001
        def decorator(fn):
            return fn

        if args and callable(args[0]) and not kwargs:
            return args[0]
        return decorator

    stub.lru_cache = lru_cache  # type: ignore[attr-defined]
    sys.modules["methodtools"] = stub


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def try_offline_closedeBH(evalues, alpha: float = 0.05) -> ProcedureResult:
    """OFFLINE closedeBH. Not UAI 2026 online closed e-LOND."""
    reason_prefix = (
        "offline e-Closure (arXiv:2509.02517 / neilzxu/eclosure). "
        "This is NOT UAI 2026 online closed e-LOND. "
    )
    src = OFFICIAL_ECLOSURE / "python" / "src"
    if not src.is_dir():
        return ProcedureResult.unavailable(
            "offline-closedeBH",
            reason_prefix + "clone missing",
            OFFLINE_ECLOSURE_STATUS,
        )
    inserted = False
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
        inserted = True
    try:
        from eclosure import closedeBH  # type: ignore
    except Exception as exc:
        return ProcedureResult.unavailable(
            "offline-closedeBH",
            reason_prefix + f"C++ bindings unavailable: {type(exc).__name__}: {exc}",
            OFFLINE_ECLOSURE_STATUS,
        )
    finally:
        if inserted:
            try:
                sys.path.remove(str(src))
            except ValueError:
                pass
    try:
        n_disc = int(closedeBH(np.asarray(evalues, dtype=float), alpha=float(alpha)))
    except Exception as exc:  # pragma: no cover
        return ProcedureResult.unavailable(
            "offline-closedeBH",
            reason_prefix + f"call failed: {exc}",
            OFFLINE_ECLOSURE_STATUS,
        )
    return ProcedureResult.ok(
        "offline-closedeBH",
        np.zeros(len(evalues), dtype=bool),
        np.zeros(len(evalues), dtype=float),
        OFFLINE_ECLOSURE_STATUS,
        extra={
            "n_discoveries": n_disc,
            "note": "offline API returns a discovery *count*, not an online path",
        },
    )


def load_aistats_lond_class():
    """Load evalue-omt LOND without executing alg/__init__.py (avoids methodtools)."""
    if not EVALUE_OMT_SRC.is_dir():
        raise FileNotFoundError(f"evalue-omt missing at {EVALUE_OMT_SRC}")
    _methodtools_shim()
    alg_dir = EVALUE_OMT_SRC / "alg"
    if "alg" not in sys.modules:
        pkg = types.ModuleType("alg")
        pkg.__path__ = [str(alg_dir)]  # type: ignore[attr-defined]
        sys.modules["alg"] = pkg
    if "alg.alg_dispatch" not in sys.modules:
        _load_module("alg.alg_dispatch", alg_dir / "alg_dispatch.py")
    if "alg.gamma_scheduler" not in sys.modules:
        _load_module("alg.gamma_scheduler", alg_dir / "gamma_scheduler.py")
    if "alg.LOND" not in sys.modules:
        _load_module("alg.LOND", alg_dir / "LOND.py")
    from alg.LOND import LOND  # type: ignore

    return LOND


def aistats_elond_on_evalues(
    evalues,
    delta: float,
    *,
    hypotheses: int | None = None,
    gamma_series: str = "easy",
) -> ProcedureResult:
    """AISTATS 2024 official LOND on p=min(1, 1/e).

    Finite-horizon renormalized γ ('easy' = 1/((t+1)(t+2))) differs from
    UAI γ_t = 1/(t(t+1)). Labelled aistats2024_official_wrapper.
    """
    evalues = np.asarray(list(evalues), dtype=float)
    pvalues = np.minimum(1.0, 1.0 / np.maximum(evalues, 1e-300))
    n = int(evalues.size)
    try:
        lond_cls = load_aistats_lond_class()
    except Exception as exc:
        return ProcedureResult.unavailable(
            "aistats2024-e-LOND",
            f"could not load evalue-omt LOND: {type(exc).__name__}: {exc}",
            AISTATS_ELOND_STATUS,
        )
    inst = lond_cls(
        delta=float(delta),
        gamma_series=gamma_series,
        reshaping=None,
        hypotheses=int(hypotheses or n),
    )
    rej = np.asarray(inst.run_fdr(pvalues), dtype=bool)
    alphas = np.asarray(inst.alpha, dtype=float)
    return ProcedureResult.ok(
        "aistats2024-e-LOND",
        rej,
        alphas,
        AISTATS_ELOND_STATUS,
        extra={
            "gamma_series": gamma_series,
            "note": "AISTATS 2024, not UAI 2026; γ is finite-horizon renormalized 'easy'",
        },
    )
