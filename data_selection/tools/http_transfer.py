"""Resumable HTTP transfer. Partial files are never marked complete."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError


class TransferError(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _headers(response) -> dict:
    raw = getattr(response, "headers", {}) or {}
    if hasattr(raw, "items"):
        return {str(key).lower(): value for key, value in raw.items()}
    return {str(key).lower(): value for key, value in dict(raw).items()}


def _read_body(response) -> bytes:
    read = getattr(response, "read", None)
    if read is None:
        return b""
    chunks = []
    while True:
        chunk = read(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def inspect_archive(path: Path) -> dict:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
        return {"format": "zip", "member_count": len(names), "members_preview": names[:20]}
    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        return {"format": "tar.gz", "member_count": None, "members_preview": []}
    return {"format": "not_an_archive", "member_count": None, "members_preview": []}


def _provenance_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".provenance.json")


def _load_provenance(dest: Path) -> dict | None:
    path = _provenance_path(dest)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_provenance(dest: Path, payload: dict) -> None:
    _provenance_path(dest).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def head_request(opener, url: str) -> dict:
    """Return headers when HEAD works. Servers that reject HEAD are recorded, not treated as success."""
    try:
        response = opener(url, method="HEAD", headers={"User-Agent": "data-selection-v0"})
    except HTTPError as exc:
        return {"supported": False, "status": exc.code, "headers": _headers(exc), "error": str(exc)}
    except URLError as exc:
        return {"supported": False, "status": None, "headers": {}, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"supported": False, "status": None, "headers": {}, "error": f"{type(exc).__name__}: {exc}"}
    status = getattr(response, "status", None)
    headers = _headers(response)
    if status in {405, 501} or status is None:
        return {"supported": False, "status": status, "headers": headers, "error": "HEAD not supported"}
    if status != 200:
        return {"supported": False, "status": status, "headers": headers, "error": f"HEAD status {status}"}
    return {"supported": True, "status": status, "headers": headers, "error": ""}


def _content_range_start(header: str | None) -> int | None:
    if not header or not header.lower().startswith("bytes"):
        return None
    try:
        span = header.split(" ", 1)[1].split("/")[0]
        start, _end = span.split("-")
        return int(start)
    except (IndexError, ValueError):
        return None


def download_with_resume(
    url: str,
    dest: Path,
    opener,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    retries: int = 3,
    backoff_seconds: float = 0.0,
    mirror_url: str | None = None,
) -> dict:
    """Download to a temporary partial file, then rename only after size and checksum checks."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")
    existing = _load_provenance(dest)
    if dest.is_file() and existing and existing.get("status") == "DOWNLOADED_VERIFIED":
        size_ok = expected_size is None or dest.stat().st_size == expected_size
        hash_ok = not expected_sha256 or existing.get("sha256") == expected_sha256
        if size_ok and hash_ok and existing.get("sha256") == sha256_file(dest):
            return {
                "status": "DOWNLOADED_VERIFIED",
                "path": str(dest),
                "bytes": dest.stat().st_size,
                "sha256": existing.get("sha256"),
                "url": existing.get("url") or url,
                "redownloaded": False,
            }
    head = head_request(opener, url)
    length_header = (head.get("headers") or {}).get("content-length")
    if expected_size is None and length_header:
        try:
            expected_size = int(length_header)
        except ValueError:
            expected_size = None
    last_error = head.get("error") or ""
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "data-selection-v0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            response = opener(url, method="GET", headers=headers)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(backoff_seconds * (2**attempt))
            continue
        status = getattr(response, "status", 200)
        response_headers = _headers(response)
        if offset and status == 200:
            # The server ignored Range. Restart instead of appending a second copy.
            if partial.exists():
                partial.unlink()
            offset = 0
        elif offset and status != 206:
            last_error = f"range request returned HTTP {status}"
            time.sleep(backoff_seconds * (2**attempt))
            continue
        if status == 206:
            start = _content_range_start(response_headers.get("content-range"))
            if start != offset:
                last_error = f"Content-Range start {start} does not match offset {offset}"
                time.sleep(backoff_seconds * (2**attempt))
                continue
        if status not in {200, 206}:
            last_error = f"HTTP {status}"
            time.sleep(backoff_seconds * (2**attempt))
            continue
        body = _read_body(response)
        mode = "ab" if offset and status == 206 else "wb"
        with partial.open(mode) as handle:
            handle.write(body)
        size = partial.stat().st_size
        if expected_size is not None and size < expected_size:
            return {
                "status": "PARTIAL_DOWNLOAD",
                "path": str(partial),
                "bytes": size,
                "expected_size": expected_size,
                "sha256": "",
                "url": mirror_url or url,
                "reason": "response ended before Content-Length",
            }
        if expected_size is not None and size > expected_size:
            return {
                "status": "DOWNLOAD_FAILED",
                "path": str(partial),
                "bytes": size,
                "expected_size": expected_size,
                "sha256": "",
                "url": mirror_url or url,
                "reason": "partial file is larger than Content-Length",
            }
        digest = sha256_file(partial)
        if expected_sha256 and digest != expected_sha256:
            return {
                "status": "DOWNLOAD_FAILED",
                "path": str(partial),
                "bytes": size,
                "sha256": digest,
                "url": mirror_url or url,
                "reason": "checksum mismatch",
            }
        partial.replace(dest)
        archive = inspect_archive(dest)
        provenance = {
            "status": "DOWNLOADED_VERIFIED",
            "url": mirror_url or url,
            "origin_url": url if mirror_url is None else url,
            "mirror_url": mirror_url,
            "bytes": dest.stat().st_size,
            "sha256": digest,
            "content_type": response_headers.get("content-type"),
            "archive": archive,
        }
        _write_provenance(dest, provenance)
        return {
            "status": "DOWNLOADED_VERIFIED",
            "path": str(dest),
            "bytes": dest.stat().st_size,
            "sha256": digest,
            "url": mirror_url or url,
            "archive": archive,
            "redownloaded": True,
        }
    return {
        "status": "DOWNLOAD_FAILED",
        "path": str(partial) if partial.exists() else "",
        "bytes": partial.stat().st_size if partial.exists() else 0,
        "sha256": "",
        "url": url,
        "reason": last_error or "download failed",
    }
