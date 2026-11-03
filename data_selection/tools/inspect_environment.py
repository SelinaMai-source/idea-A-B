#!/usr/bin/env python3
"""Record the machine actually running the tests. Does not install packages."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _torch() -> dict:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}", "cuda": False}
    try:
        cuda = bool(torch.cuda.is_available())
    except Exception as exc:  # noqa: BLE001
        return {"importable": True, "version": getattr(torch, "__version__", ""), "cuda": False, "error": str(exc)}
    return {"importable": True, "version": torch.__version__, "cuda": cuda, "error": ""}


def _nvidia() -> str:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return "NOT_FOUND"
    try:
        proc = subprocess.run([binary, "-L"], check=False, capture_output=True, text=True, timeout=10)
    except Exception as exc:  # noqa: BLE001
        return f"FAILED {exc}"
    return (proc.stdout or proc.stderr or "").strip() or f"exit {proc.returncode}"


def _which(name: str) -> str:
    return shutil.which(name) or "NOT_FOUND"


def collect() -> dict:
    conda = _which("conda")
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": sys.version,
        "executable": sys.executable,
        "nvidia_smi": _nvidia(),
        "torch": _torch(),
        "conda": conda,
        "mamba": _which("mamba"),
        "cuda_training_supported_here": False if platform.system() == "Darwin" else None,
        "conda_envs_created": False,
        "notes": (
            "Parent venv is used as-is. No system-wide install. "
            "Conda environment creation is refused on Darwin even if conda exists. "
            "PyTorch 1.12.1+cu116 is the author pin and is not installed on this Mac. "
            "RTX 4090 compatibility with CUDA 11.6 is not claimed."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the local data_selection environment.")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "manifests" / "local_environment.json"))
    args = parser.parse_args()
    payload = collect()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cuda": payload["torch"]["cuda"], "machine": payload["machine"], "output": str(path)}))
    if args.require_cuda and not payload["torch"]["cuda"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
