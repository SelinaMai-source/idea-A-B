"""Load per-directory adapters. Directories named ``_sources`` are not methods."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover(kind: str, attribute: str, root: Path | None = None) -> list:
    base = (root or ROOT) / kind
    found = []
    if not base.is_dir():
        return found
    for child in sorted(path for path in base.iterdir() if path.is_dir()):
        if child.name.startswith("_") or child.name.startswith("."):
            continue
        adapter = child / "adapter.py"
        if not adapter.is_file():
            continue
        module = _load(f"{kind}.{child.name}.adapter", adapter)
        if not hasattr(module, attribute):
            raise AttributeError(f"{adapter} is missing {attribute}")
        found.append(getattr(module, attribute))
    return found


def discover_baselines(root: Path | None = None) -> list:
    return discover("baselines", "SELECTOR", root)


def discover_benchmarks(root: Path | None = None) -> list:
    return discover("benchmarks", "BENCHMARK", root)


def discover_references(root: Path | None = None) -> list:
    return discover("references", "REFERENCE", root)
