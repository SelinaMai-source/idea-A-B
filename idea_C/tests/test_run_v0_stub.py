"""The v0 runner stub must not start Monte Carlo."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v0.py"
SPEC = importlib.util.spec_from_file_location("run_v0", SCRIPT)
assert SPEC and SPEC.loader
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def test_list_exits_zero() -> None:
    assert RUN.main(["--list"]) == 0


def test_named_experiment_is_refused() -> None:
    assert RUN.main(["p0_aaai_robust_gaussian"]) == 2
