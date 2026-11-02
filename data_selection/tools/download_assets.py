#!/usr/bin/env python3
"""Resumable downloader. Refuses the RoboMimic image zip on macOS."""

from __future__ import annotations

import argparse
import platform
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "artifacts.yaml"
# 2 GB. The image zip is far above this and is also named below.
LARGE_BYTES = 2_000_000_000
MAC_FORBIDDEN = {"robomimic_image"}


def _assets() -> list[dict]:
    payload = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    return list(payload.get("assets") or [])


def _head(url: str) -> dict:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "data-selection-v0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        headers = {key.lower(): value for key, value in resp.headers.items()}
        return {"status": resp.status, "content_length": headers.get("content-length"), "accept_ranges": headers.get("accept-ranges")}


def download_one(name: str, dest: Path, allow_large: bool) -> dict:
    asset = next(item for item in _assets() if item["name"] == name)
    if platform.system() == "Darwin" and name in MAC_FORBIDDEN:
        raise SystemExit(
            f"refusing {name} on macOS. Content stays SOURCE_VERIFIED_DEFERRED. "
            "Run this command on the Linux GPU machine."
        )
    meta = _head(asset["url"])
    length = int(meta["content_length"]) if meta.get("content_length") else None
    if length is not None and length >= LARGE_BYTES and not allow_large:
        raise SystemExit(
            f"refusing {name}: content-length {length} bytes. Pass --allow-large on the server after checking free disk."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if length is not None and existing > length:
        raise SystemExit(f"{dest} is larger than content-length; not overwriting")
    if length is not None and existing == length:
        return {"name": name, "status": "ALREADY_COMPLETE", "bytes": existing, "path": str(dest)}
    headers = {"User-Agent": "data-selection-v0"}
    mode = "wb"
    if existing and meta.get("accept_ranges") == "bytes":
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
    req = urllib.request.Request(asset["url"], headers=headers)
    with urllib.request.urlopen(req, timeout=120) as resp, dest.open(mode) as handle:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    return {"name": name, "status": "DOWNLOADED_UNVERIFIED", "bytes": dest.stat().st_size, "path": str(dest)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a named asset from manifests/artifacts.yaml")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print({"dry_run": True, "asset": args.asset, "will_download": False})
        return 0
    print(download_one(args.asset, Path(args.dest), args.allow_large))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
