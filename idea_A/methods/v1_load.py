"""Load Idea A method modules from dotted ``v1.0/`` and ``v1.1/`` directories.

Those directory names contain a dot, so they cannot be imported as
``idea_A.methods.v1.0`` or ``idea_A.methods.v1.1``. v1.1 is registered as a
virtual package ``idea_A.methods.v1_1`` whose ``__path__`` points at ``v1.1/``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

V1_DIR = Path(__file__).resolve().parent / "v1.0"
V1_1_DIR = Path(__file__).resolve().parent / "v1.1"
V1_1_PACKAGE = "idea_A.methods.v1_1"


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


def register_v1_1_package() -> ModuleType:
    """Expose ``v1.1/`` as the virtual package ``idea_A.methods.v1_1``."""
    existing = sys.modules.get(V1_1_PACKAGE)
    if existing is not None and getattr(existing, "__path__", None) is not None:
        return existing
    pkg = ModuleType(V1_1_PACKAGE)
    pkg.__path__ = [str(V1_1_DIR)]  # type: ignore[attr-defined]
    pkg.__package__ = V1_1_PACKAGE
    pkg.__file__ = str(V1_1_DIR / "__init__.py")
    sys.modules[V1_1_PACKAGE] = pkg
    return pkg


def load_v1_1_module(stem: str) -> ModuleType:
    register_v1_1_package()
    path = V1_1_DIR / f"{stem}.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    name = f"{V1_1_PACKAGE}.{stem}"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(path):
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = V1_1_PACKAGE
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


register_v1_1_package()
