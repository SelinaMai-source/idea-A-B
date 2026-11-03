#!/usr/bin/env python3
"""Verify local files against the artifact manifest.

A missing body stays at the manifest status. A checksum failure is
DOWNLOAD_FAILED and is not rewritten as SOURCE_VERIFIED_DEFERRED.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.http_transfer import inspect_archive, sha256_file  # noqa: E402


def verify_asset(asset: dict, root: Path) -> dict:
    relative = asset.get("local_relative_path") or ""
    path = root / relative if relative else None
    expected = asset.get("sha256") or ""
    expected_size = asset.get("content_length_bytes")
    manifest_status = asset.get("status")
    if path is None or not path.is_file():
        return {
            "name": asset["name"],
            "status": manifest_status,
            "verified": False,
            "reason": "file not present",
            "sha256": "",
            "bytes": 0,
        }
    actual_size = path.stat().st_size
    if expected_size is not None and actual_size != int(expected_size):
        return {
            "name": asset["name"],
            "status": "PARTIAL_DOWNLOAD" if actual_size < int(expected_size) else "DOWNLOAD_FAILED",
            "verified": False,
            "reason": f"size {actual_size} != content-length {expected_size}",
            "sha256": "",
            "bytes": actual_size,
        }
    if not expected:
        return {
            "name": asset["name"],
            "status": "PRESENT_WITHOUT_CHECKSUM",
            "verified": False,
            "reason": "no published sha256 recorded",
            "sha256": "",
            "bytes": actual_size,
            "archive": inspect_archive(path),
        }
    actual = sha256_file(path)
    if actual != expected:
        return {
            "name": asset["name"],
            "status": "DOWNLOAD_FAILED",
            "verified": False,
            "reason": "checksum mismatch",
            "sha256": actual,
            "bytes": actual_size,
        }
    return {
        "name": asset["name"],
        "status": "DOWNLOADED_VERIFIED",
        "verified": True,
        "reason": "",
        "sha256": actual,
        "bytes": actual_size,
        "archive": inspect_archive(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify downloaded assets.")
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "artifacts.yaml"))
    parser.add_argument("--root", default=str(ROOT / "artifacts"))
    args = parser.parse_args()
    payload = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8")) or {}
    reports = [verify_asset(asset, Path(args.root)) for asset in payload.get("assets") or []]
    print(json.dumps(reports, indent=2))
    failed = [row for row in reports if row["status"] == "DOWNLOAD_FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
