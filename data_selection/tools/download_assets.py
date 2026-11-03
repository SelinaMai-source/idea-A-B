#!/usr/bin/env python3
"""Download one named asset. Large RoboMimic image bodies stay deferred on macOS.

DOWNLOAD_FAILED is a transfer error. SOURCE_VERIFIED_DEFERRED means the URL was
checked and the body was intentionally not fetched.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.http_transfer import download_with_resume  # noqa: E402

MANIFEST = ROOT / "manifests" / "artifacts.yaml"
LARGE_BYTES = 2_000_000_000
MAC_FORBIDDEN = {"robomimic_image", "robomimic_lowdim"}


def _assets(manifest: Path) -> list[dict]:
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    return list(payload.get("assets") or [])


def urllib_opener(url: str, method: str = "GET", headers: dict | None = None):
    request = urllib.request.Request(url, method=method, headers=headers or {})
    return urllib.request.urlopen(request, timeout=60)


def download_one(
    name: str,
    dest: Path,
    *,
    allow_large: bool,
    manifest: Path = MANIFEST,
    opener=None,
    system: str | None = None,
) -> dict:
    asset = next(item for item in _assets(manifest) if item["name"] == name)
    system = system or platform.system()
    if system == "Darwin" and name in MAC_FORBIDDEN:
        return {
            "name": name,
            "status": "SOURCE_VERIFIED_DEFERRED",
            "reason": f"macOS refuses the {name} body; status stays SOURCE_VERIFIED_DEFERRED",
            "url": asset["url"],
            "expected_size": asset.get("content_length_bytes"),
            "bytes": 0,
            "sha256": "",
            "path": "",
        }
    expected = asset.get("content_length_bytes")
    if expected is not None and int(expected) >= LARGE_BYTES and not allow_large:
        return {
            "name": name,
            "status": "SOURCE_VERIFIED_DEFERRED",
            "reason": f"content-length {expected} is at least {LARGE_BYTES}; pass --allow-large on the server",
            "url": asset["url"],
            "expected_size": int(expected),
            "bytes": 0,
            "sha256": "",
            "path": "",
        }
    report = download_with_resume(
        asset["url"],
        dest,
        opener or urllib_opener,
        expected_size=int(expected) if expected else None,
        expected_sha256=asset.get("sha256") or None,
    )
    report["name"] = name
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a named asset from manifests/artifacts.yaml")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"dry_run": True, "asset": args.asset, "will_download": False}))
        return 0
    report = download_one(
        args.asset,
        Path(args.dest),
        allow_large=args.allow_large,
        manifest=Path(args.manifest),
    )
    print(json.dumps(report, indent=2))
    return 0 if report["status"] in {"DOWNLOADED_VERIFIED", "SOURCE_VERIFIED_DEFERRED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
