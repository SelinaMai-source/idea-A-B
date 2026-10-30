"""Load Idea A v1.0 modules from the ``v1.0/`` directory.

The directory name contains a dot, so it cannot be imported as
``idea_A.methods.v1.0``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

V1_DIR = Path(__file__).resolve().parent / "v1.0"


def load_v1_module(stem: str) -> ModuleType:
    path = V1_DIR / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f"idea_A_methods_v1_{stem}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
