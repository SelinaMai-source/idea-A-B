#!/usr/bin/env python3
"""Verify local files against recorded checksums. Missing checksums stay unverified."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify downloaded assets.")
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "artifacts.yaml"))
    parser.add_argument("--root", default=str(ROOT / "artifacts"))
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8")) or {}
    reports = []
    for asset in payload.get("assets") or []:
        relative = asset.get("local_relative_path") or ""
        path = Path(args.root) / relative if relative else None
        expected = asset.get("sha256") or ""
        if path is None or not path.is_file():
            reports.append({"name": asset["name"], "status": asset.get("status"), "verified": False, "reason": "file not present"})
            continue
        if not expected:
            reports.append({"name": asset["name"], "status": "PRESENT_WITHOUT_CHECKSUM", "verified": False, "reason": "no published sha256 recorded"})
            continue
        actual = sha256(path)
        reports.append({"name": asset["name"], "verified": actual == expected, "sha256": actual})
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
