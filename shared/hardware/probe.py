"""Record the machine and Python environment without running experiments.

``collect_environment`` only reads local facts. ``write_environment_manifests``
writes the same JSON to ``reports/environment_manifest.json`` and
``shared/hardware/environment_manifest.json``.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_NAMES = {
    "numpy": ("numpy",),
    "scipy": ("scipy",),
    "scikit-learn": ("scikit-learn",),
    "pandas": ("pandas",),
    "pyyaml": ("PyYAML", "pyyaml"),
    "matplotlib": ("matplotlib",),
    "pytest": ("pytest",),
    "torch": ("torch",),
    "pip": ("pip",),
    "setuptools": ("setuptools",),
    "transformers": ("transformers",),
    "datasets": ("datasets",),
    "vllm": ("vllm",),
    "bitsandbytes": ("bitsandbytes",),
    "flash-attn": ("flash-attn", "flash_attn"),
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _package_version(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


def _run(args: list[str], cwd: Path) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return 1, "", f"{type(exc).__name__}: {exc}"
    return completed.returncode, completed.stdout.strip(), completed.stderr.strip()


def _ram_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages) * int(page_size)
    except (AttributeError, OSError, ValueError):
        pass
    code, out, _err = _run(["sysctl", "-n", "hw.memsize"], Path("/"))
    if code == 0 and out.isdigit():
        return int(out)
    return None


def _torch_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "installed": False,
        "version": None,
        "mps_built": None,
        "mps_available": None,
        "mps_functional_check": "not_run",
        "cuda_available": None,
        "import_error": None,
    }
    try:
        import torch
    except Exception as exc:
        status["import_error"] = f"{type(exc).__name__}: {exc}"
        return status
    status["installed"] = True
    status["version"] = str(torch.__version__)
    try:
        status["mps_built"] = bool(torch.backends.mps.is_built())
        status["mps_available"] = bool(torch.backends.mps.is_available())
    except Exception as exc:
        status["import_error"] = f"{type(exc).__name__}: {exc}"
    try:
        status["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        status["cuda_available"] = None
        status["import_error"] = f"{type(exc).__name__}: {exc}"
    return status


def _git_status(root: Path) -> dict[str, Any]:
    code, version, err = _run(["git", "--version"], root)
    branch_code, branch, branch_err = _run(["git", "symbolic-ref", "--short", "HEAD"], root)
    commit_code, commit, commit_err = _run(["git", "rev-parse", "HEAD"], root)
    return {
        "version": version if code == 0 else None,
        "version_error": None if code == 0 else (err or version or "git --version failed"),
        "branch": branch if branch_code == 0 else None,
        "branch_error": None if branch_code == 0 else (branch_err or branch or "branch unavailable"),
        "commit": commit if commit_code == 0 else None,
        "commit_error": None if commit_code == 0 else (commit_err or "no commits yet"),
    }


def _pip_log_tail(root: Path) -> str | None:
    path = root / "shared" / "hardware" / "torch_install.log"
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-40:])


def collect_environment(root: Path | None = None) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of this machine and interpreter."""
    repo = Path(root) if root is not None else _repo_root()
    usage = shutil.disk_usage(repo if repo.exists() else Path.home())
    ram = _ram_bytes()
    uname = platform.uname()
    mac_release, mac_versioninfo, mac_machine = platform.mac_ver()
    torch_status = _torch_status()
    if not torch_status["installed"]:
        tail = _pip_log_tail(repo)
        if tail:
            torch_status["pip_log_tail"] = tail
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "platform": platform.platform(),
            "mac_ver": {
                "release": mac_release,
                "versioninfo": list(mac_versioninfo),
                "machine": mac_machine,
            },
            "machine": platform.machine(),
            "processor": platform.processor(),
            "uname": {
                "system": uname.system,
                "node": uname.node,
                "release": uname.release,
                "version": uname.version,
                "machine": uname.machine,
            },
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "version_string": sys.version,
        },
        "ram_bytes": ram,
        "ram_gib": None if ram is None else ram / (1024**3),
        "disk": {
            "path": str(repo),
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
            "free_gib": usage.free / (1024**3),
        },
        "torch": torch_status,
        "git": _git_status(repo),
        "packages": {name: _package_version(candidates) for name, candidates in PACKAGE_NAMES.items()},
        "experiments_run": False,
        "notes": [
            "Scaffolding phase only. No Idea A or Idea B experiment has been executed.",
            "MPS availability is torch.backends.mps.is_available(). No MPS tensor was allocated.",
            "transformers and datasets are recorded if present; this phase does not install them.",
        ],
    }


def write_environment_manifests(root: Path | None = None) -> dict[str, Any]:
    """Write the environment snapshot to the two manifest paths and return it."""
    repo = Path(root) if root is not None else _repo_root()
    payload = collect_environment(repo)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    destinations = (
        repo / "reports" / "environment_manifest.json",
        repo / "shared" / "hardware" / "environment_manifest.json",
    )
    for path in destinations:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return payload


if __name__ == "__main__":
    write_environment_manifests()
