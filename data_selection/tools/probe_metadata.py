#!/usr/bin/env python3
"""Collect lightweight HTTP metadata. Does not download dataset bodies."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manifests" / "http_probe.json"

HEAD_URLS = [
    "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip",
    "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip",
]

GET_URLS = [
    "https://diffusion-policy.cs.columbia.edu/data/training/",
    "https://diffusion-policy.cs.columbia.edu/data/experiments/",
    "https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets",
    "https://hf-mirror.com/api/datasets/yifengzhu-hf/LIBERO-datasets",
    "https://huggingface.co/api/datasets/lerobot/libero",
    "https://hf-mirror.com/api/datasets/lerobot/libero",
    "https://huggingface.co/api/datasets/yifengzhu-hf/LIBERO-datasets/revision/main",
    "https://hf-mirror.com/api/datasets/yifengzhu-hf/LIBERO-datasets/tree/main",
]


def _request(url: str, method: str, limit: int) -> dict:
    req = urllib.request.Request(
        url,
        method=method,
        headers={"User-Agent": "data-selection-v0-metadata/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            raw = resp.read(limit)
            headers = {k.lower(): v for k, v in resp.headers.items()}
            return {
                "url": url,
                "method": method,
                "status": resp.status,
                "final_url": resp.geturl(),
                "headers": {
                    k: headers.get(k)
                    for k in (
                        "content-type",
                        "content-length",
                        "accept-ranges",
                        "etag",
                        "last-modified",
                        "location",
                    )
                },
                "body_prefix": raw.decode("utf-8", errors="replace")[:20000],
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        body = exc.read(2000).decode("utf-8", errors="replace")
        return {
            "url": url,
            "method": method,
            "status": exc.code,
            "final_url": url,
            "headers": {},
            "body_prefix": body[:2000],
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - probe must record the failure
        return {
            "url": url,
            "method": method,
            "status": None,
            "final_url": url,
            "headers": {},
            "body_prefix": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    rows = []
    for url in HEAD_URLS:
        rows.append(_request(url, "HEAD", limit=0))
        print(url, rows[-1]["status"], rows[-1]["headers"].get("content-length"), rows[-1]["error"], flush=True)
    for url in GET_URLS:
        rows.append(_request(url, "GET", limit=400_000))
        print(url, rows[-1]["status"], rows[-1]["error"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("WROTE", OUT)


if __name__ == "__main__":
    main()
