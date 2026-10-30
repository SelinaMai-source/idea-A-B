#!/usr/bin/env python3
"""Check the scaffolding environment. This does not run Idea A or Idea B."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("pyyaml", "yaml"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
)


def main() -> int:
    print(f"executable={sys.executable}")
    print(f"python={sys.version.split()[0]}")
    failed = False
    for dist, module_name in REQUIRED:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"MISSING {dist}: {type(exc).__name__}: {exc}")
            failed = True
            continue
        print(f"OK {dist}=={getattr(module, '__version__', 'unknown')}")

    try:
        import torch

        print(
            f"OK torch=={torch.__version__} "
            f"mps_available={torch.backends.mps.is_available()} "
            f"mps_built={torch.backends.mps.is_built()}"
        )
    except Exception as exc:
        print(f"TORCH_NOT_INSTALLED {type(exc).__name__}: {exc}")

    try:
        from shared.data_utils.seeds import seed_everything
        from shared.metrics.estimators import ipw_mean

        seed_everything(0)
        del ipw_mean
        print("OK shared imports")
    except Exception as exc:
        print(f"SHARED_IMPORT_FAILED {type(exc).__name__}: {exc}")
        failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
