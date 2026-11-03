#!/usr/bin/env python3
"""Verify pinned official checkouts. Do not move a pin to upstream HEAD.

A checkout whose HEAD differs from the manifest pin is left untouched.
Partial trees are not deleted. Dirty worktrees are not reset.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_FILES = {
    "cupid": [
        "eval_demonstration_scores.py",
        "train_trak_diffusion.py",
        "finalize_trak.py",
    ],
    "deminf": ["scripts/quality/estimate_quality.py"],
    "datamil": ["scripts/perform_libero_dm.sh"],
    "demo_score": ["demo_score/demo_score/train_trajwise_classifier.py"],
    "strap": ["strap/retrieval/retrieval.py", "strap/embedding/encode_datasets.py"],
    "robomimic_strap": ["robomimic/algo/diffusion_policy.py"],
    "hinflow": [".gitmodules"],
    "diffusion_policy": [
        "diffusion_policy/config/train_diffusion_unet_image_workspace.yaml",
        "diffusion_policy/policy/diffusion_unet_image_policy.py",
        "diffusion_policy/workspace/train_diffusion_unet_image_workspace.py",
    ],
    "robomimic": ["robomimic/scripts/train.py"],
    "robosuite": ["robosuite/__init__.py"],
    "libero": ["benchmark_scripts/download_libero_datasets.py"],
}


def _default_run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"},
    )


def _owner_repo(url: str) -> tuple[str, str]:
    cleaned = url.removesuffix(".git").rstrip("/")
    parts = cleaned.split("/")
    return parts[-2], parts[-1]


def archive_url(url: str, sha: str) -> str:
    owner, repo = _owner_repo(url)
    return f"https://github.com/{owner}/{repo}/archive/{sha}.tar.gz"


def _expected_missing(dest: Path, name: str) -> list[str]:
    missing = []
    for relative in EXPECTED_FILES.get(name, []):
        if not (dest / relative).is_file():
            missing.append(relative)
    return missing


def _submodule_lines(dest: Path, runner, timeout: int) -> list[str]:
    if not (dest / ".gitmodules").exists():
        return []
    proc = runner(["git", "-C", str(dest), "submodule", "status", "--recursive"], timeout)
    if proc.returncode != 0:
        return [f"ERROR {proc.stderr.strip()}"]
    return [line for line in (proc.stdout or "").splitlines() if line.strip()]


def _dirty(dest: Path, runner, timeout: int) -> list[str]:
    proc = runner(["git", "-C", str(dest), "status", "--porcelain"], timeout)
    if proc.returncode != 0:
        return [f"ERROR {proc.stderr.strip()}"]
    return [line for line in (proc.stdout or "").splitlines() if line.strip()]


def verify_record(record: dict, runner, *, timeout: int = 30, root: Path = ROOT) -> dict:
    """Refresh verification fields. ``commit_sha`` is the pin and is not replaced."""
    updated = dict(record)
    pin = (record.get("commit_sha") or "").strip()
    updated["commit_sha"] = pin
    dest = root / record["local_path"]
    missing = _expected_missing(dest, record["name"]) if dest.exists() else list(EXPECTED_FILES.get(record["name"], []))
    updated["expected_files_missing"] = missing
    updated["verified_at"] = datetime.now(timezone.utc).isoformat()
    if not dest.exists():
        updated["pin_verified"] = False
        updated["clone_status"] = "SOURCE_UNAVAILABLE"
        updated["failure_reason"] = "checkout directory is absent"
        updated["observed_head"] = ""
        return updated
    if not (dest / ".git").exists():
        updated["pin_verified"] = False
        updated["clone_status"] = record.get("clone_status") or "SOURCE_UNAVAILABLE"
        updated["failure_reason"] = "destination exists without .git; left in place"
        updated["observed_head"] = ""
        return updated
    proc = runner(["git", "-C", str(dest), "rev-parse", "HEAD"], timeout)
    head = (proc.stdout or "").strip()
    updated["observed_head"] = head
    dirty = _dirty(dest, runner, timeout)
    updated["local_modifications"] = dirty
    if proc.returncode != 0 or not head:
        updated["pin_verified"] = False
        updated["failure_reason"] = "rev-parse failed; pin was not changed"
        return updated
    if head != pin:
        updated["pin_verified"] = False
        updated["failure_reason"] = (
            f"HEAD {head} differs from pin {pin}; checkout was not updated"
        )
        updated["submodule_status_lines"] = _submodule_lines(dest, runner, timeout)
        return updated
    updated["pin_verified"] = not missing
    updated["clone_status"] = "CLONED"
    updated["failure_reason"] = "" if not missing else "pinned HEAD is present but expected files are missing"
    updated["submodule_status_lines"] = _submodule_lines(dest, runner, timeout)
    return updated


def init_missing_submodules(record: dict, runner, *, timeout: int) -> dict:
    """Init submodules that are not checked out. Dirty paths are skipped."""
    dest = ROOT / record["local_path"]
    lines = list(record.get("submodule_status_lines") or [])
    notes = []
    if record.get("local_modifications"):
        notes.append("local modifications preserved; dirty paths were not reset")
    for line in lines:
        if not line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        path = parts[1]
        if any(path in item for item in record.get("local_modifications") or []):
            notes.append(f"skipped dirty submodule {path}")
            continue
        try:
            proc = runner(
                ["git", "-C", str(dest), "submodule", "update", "--init", "--depth", "1", path],
                timeout,
            )
        except subprocess.TimeoutExpired as exc:
            notes.append(f"missing submodule {path}: timed out after {timeout}s: {exc}")
            continue
        if proc.returncode != 0:
            detail = ((proc.stderr or "") + (proc.stdout or "")).strip()[-500:]
            notes.append(f"missing submodule {path}: {detail}")
        else:
            notes.append(f"initialized {path}")
    record = dict(record)
    record["submodule_init_notes"] = notes
    if any(note.startswith("missing submodule") for note in notes):
        record["submodules"] = {
            "status": "INCOMPLETE",
            "detail": " | ".join(notes)[-2000:],
        }
    return record


def fetch_pinned(record: dict, runner, *, timeout: int, archive_fetcher=None, root: Path = ROOT) -> dict:
    """Create a checkout at the pinned SHA. Never checks out a different revision."""
    pin = record["commit_sha"]
    dest = root / record["local_path"]
    if dest.exists():
        return verify_record(record, runner, timeout=timeout, root=root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    commands = [
        ["git", "init", str(dest)],
        ["git", "-C", str(dest), "remote", "add", "origin", record["url"]],
        ["git", "-C", str(dest), "fetch", "--depth", "1", "origin", pin],
    ]
    logs = []
    ok = True
    for command in commands:
        proc = runner(command, timeout)
        logs.append(((proc.stdout or "") + (proc.stderr or "")).strip()[-400:])
        if proc.returncode != 0:
            ok = False
            break
    if not ok:
        deeper = runner(["git", "-C", str(dest), "fetch", "origin", pin], timeout)
        logs.append(((deeper.stdout or "") + (deeper.stderr or "")).strip()[-400:])
        ok = deeper.returncode == 0
    if ok:
        checkout = runner(["git", "-C", str(dest), "checkout", "--detach", pin], timeout)
        logs.append(((checkout.stdout or "") + (checkout.stderr or "")).strip()[-400:])
        ok = checkout.returncode == 0
    if not ok and archive_fetcher is not None:
        url = archive_url(record["url"], pin)
        try:
            archive_fetcher(url, dest)
            record = dict(record)
            record["archive_url_used"] = url
            record["failure_reason"] = ""
            record["clone_status"] = "CLONED"
            record["notes"] = "Fetched the pinned commit archive after git fetch failed. Pin was not changed."
            return verify_record(record, runner, timeout=timeout, root=root)
        except Exception as exc:  # noqa: BLE001
            logs.append(f"archive failed: {type(exc).__name__}: {exc}")
    updated = verify_record(record, runner, timeout=timeout, root=root)
    if not updated.get("pin_verified"):
        updated["failure_reason"] = updated.get("failure_reason") or " ; ".join(logs)[-1000:]
        updated["clone_status"] = "SOURCE_UNAVAILABLE" if not (dest / ".git").exists() else updated["clone_status"]
    return updated


def load_manifest(path: Path) -> list[dict]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return list(payload.get("repositories") or [])


def write_manifest(path: Path, records: list[dict], original_pins: dict[str, str]) -> None:
    for record in records:
        pinned = original_pins[record["name"]]
        if record.get("commit_sha") != pinned:
            raise RuntimeError(f"refusing to overwrite pin for {record['name']}")
    path.write_text(
        yaml.safe_dump({"repositories": records}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pinned official sources. Does not move pins.")
    parser.add_argument("--manifest", default=str(ROOT / "manifests" / "source_repositories.yaml"))
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--init-submodules", action="store_true")
    parser.add_argument("--fetch-missing", action="store_true")
    args = parser.parse_args()
    path = Path(args.manifest)
    records = load_manifest(path)
    pins = {record["name"]: record.get("commit_sha") or "" for record in records}
    updated = []
    for record in records:
        print(f"VERIFY {record['name']}", flush=True)
        if args.fetch_missing and not (ROOT / record["local_path"]).exists():
            current = fetch_pinned(record, _default_run, timeout=args.timeout)
        else:
            current = verify_record(record, _default_run, timeout=args.timeout)
        if args.init_submodules and current.get("clone_status") == "CLONED" and current.get("observed_head") == pins[record["name"]]:
            current = init_missing_submodules(current, _default_run, timeout=args.timeout)
            current = verify_record(current, _default_run, timeout=args.timeout)
        updated.append(current)
        time.sleep(0)
    write_manifest(path, updated, pins)
    print(f"WROTE {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
