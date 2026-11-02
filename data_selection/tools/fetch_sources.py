#!/usr/bin/env python3
"""Clone official sources over HTTPS with retries. Do not vendor them into git.

Destinations are the per-method ``official/`` trees or ``benchmarks/_sources/``.
A failed clone is recorded as SOURCE_UNAVAILABLE and is not replaced by a
hand-written algorithm.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

SOURCES = [
    {
        "name": "cupid",
        "url": "https://github.com/agiachris/cupid",
        "local_path": "baselines/cupid/official",
        "role": "published_selector",
    },
    {
        "name": "deminf",
        "url": "https://github.com/jhejna/demonstration-information",
        "local_path": "baselines/deminf/official",
        "role": "published_selector",
    },
    {
        "name": "datamil",
        "url": "https://github.com/UT-Austin-RobIn/datamil",
        "local_path": "baselines/datamil/official",
        "role": "published_selector",
    },
    {
        "name": "demo_score",
        "url": "https://github.com/alessing/demo-score",
        "local_path": "baselines/demo_score/official",
        "role": "published_selector",
    },
    {
        "name": "strap",
        "url": "https://github.com/WEIRDLabUW/STRAP",
        "local_path": "baselines/strap/official",
        "role": "published_selector",
    },
    {
        "name": "robomimic_strap",
        "url": "https://github.com/WEIRDLabUW/robomimic_strap",
        "local_path": "baselines/strap/official_robomimic_strap",
        "role": "strap_policy_companion",
    },
    {
        "name": "hinflow",
        "url": "https://github.com/yzc0731/HinFlow",
        "local_path": "references/hinflow/official",
        "role": "online_learning_reference",
    },
    {
        "name": "diffusion_policy",
        "url": "https://github.com/real-stanford/diffusion_policy",
        "local_path": "policies/diffusion_policy_image/official",
        "role": "unified_policy",
    },
    {
        "name": "robomimic",
        "url": "https://github.com/ARISE-Initiative/robomimic",
        "local_path": "benchmarks/_sources/robomimic",
        "role": "benchmark_source",
    },
    {
        "name": "robosuite",
        "url": "https://github.com/ARISE-Initiative/robosuite",
        "local_path": "benchmarks/_sources/robosuite",
        "role": "benchmark_source",
    },
    {
        "name": "libero",
        "url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "local_path": "benchmarks/_sources/libero",
        "role": "benchmark_source",
    },
]


def _run(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "echo",
        },
    )


def _clone(url: str, dest: Path, timeout: int, retries: int) -> tuple[bool, str]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(retries):
        if dest.exists():
            break
        proc = _run(
            ["git", "clone", "--depth", "1", "--single-branch", url + ".git", str(dest)],
            timeout=timeout,
        )
        last = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0 and (dest / ".git").exists():
            return True, last.strip()
        if dest.exists() and not (dest / ".git").exists():
            # Incomplete tree. Leave it and report; do not delete user data
            # unless the directory is empty.
            try:
                dest.rmdir()
            except OSError:
                return False, "incomplete clone left at destination\n" + last.strip()
        time.sleep(min(2 ** attempt, 8))
    if (dest / ".git").exists():
        return True, "destination already had a git checkout"
    return False, last.strip() or "git clone failed"


def _sha(dest: Path) -> str:
    proc = _run(["git", "-C", str(dest), "rev-parse", "HEAD"], timeout=30)
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def _submodules(dest: Path, timeout: int) -> dict:
    if not (dest / ".gitmodules").exists():
        return {"status": "NO_SUBMODULES", "detail": ""}
    proc = _run(
        [
            "git",
            "-C",
            str(dest),
            "submodule",
            "update",
            "--init",
            "--depth",
            "1",
            "--recursive",
        ],
        timeout=timeout,
    )
    text = ((proc.stdout or "") + (proc.stderr or "")).strip()
    if proc.returncode == 0:
        return {"status": "INITIALIZED", "detail": text[-2000:]}
    return {"status": "SUBMODULE_FAILED", "detail": text[-2000:]}


def _license_hint(dest: Path) -> dict:
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING", "LICENCE"):
        path = dest / name
        if path.is_file():
            line = ""
            try:
                line = path.read_text(encoding="utf-8", errors="replace").splitlines()[0][:200]
            except OSError as exc:
                line = f"unreadable: {exc}"
            return {"file": name, "first_line": line, "spdx": "UNVERIFIED"}
    return {"file": "", "first_line": "", "spdx": "UNVERIFIED"}


def fetch_one(spec: dict, timeout: int, retries: int, submodule_timeout: int) -> dict:
    dest = ROOT / spec["local_path"]
    record = {
        "name": spec["name"],
        "url": spec["url"],
        "local_path": spec["local_path"],
        "role": spec["role"],
        "clone_depth": 1,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "commit_sha": "",
        "license": _license_hint(dest) if dest.exists() else {"file": "", "first_line": "", "spdx": "UNVERIFIED"},
        "submodules": {"status": "NOT_ATTEMPTED", "detail": ""},
        "clone_status": "SOURCE_UNAVAILABLE",
        "failure_reason": "",
        "entrypoint_files": [],
        "environment_files": [],
        "citation": "",
        "original_tasks": [],
        "data_artifacts_in_repo": False,
        "weights_in_repo": False,
        "seed_or_result_files": [],
        "missing_artifacts": [],
        "notes": "SHA is the remote HEAD observed at fetch time. Depth-1 clones do not contain full history.",
    }
    try:
        ok, log = _clone(spec["url"], dest, timeout=timeout, retries=retries)
    except subprocess.TimeoutExpired as exc:
        record["failure_reason"] = f"clone timed out after {timeout}s: {exc}"
        return record
    if not ok:
        record["failure_reason"] = log[-2000:] or "clone failed"
        return record
    sha = _sha(dest)
    if not sha:
        record["failure_reason"] = "checkout exists but rev-parse failed"
        return record
    record["clone_status"] = "CLONED"
    record["commit_sha"] = sha
    record["failure_reason"] = ""
    record["license"] = _license_hint(dest)
    try:
        record["submodules"] = _submodules(dest, timeout=submodule_timeout)
    except subprocess.TimeoutExpired as exc:
        record["submodules"] = {
            "status": "SUBMODULE_FAILED",
            "detail": f"submodule update timed out after {submodule_timeout}s: {exc}",
        }
    record["clone_log_tail"] = log[-1000:]
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone pinned official sources for data_selection v0.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--submodule-timeout", type=int, default=180)
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "manifests" / "source_repositories.yaml"),
    )
    args = parser.parse_args()
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                fetch_one,
                spec,
                args.timeout,
                args.retries,
                args.submodule_timeout,
            ): spec
            for spec in SOURCES
        }
        for fut in as_completed(futures):
            spec = futures[fut]
            record = fut.result()
            records.append(record)
            print(
                f"DONE {spec['name']} {record['clone_status']} {record['commit_sha']} {record['failure_reason'][:180]}",
                flush=True,
            )
    records.sort(key=lambda row: row["name"])
    payload = {"repositories": records}
    path = Path(args.manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
