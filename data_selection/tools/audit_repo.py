#!/usr/bin/env python3
"""Scan cloned trees for licenses, environment files, and README titles."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV_NAMES = {
    "environment.yml",
    "environment.yaml",
    "environment_macos.yml",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "requirements.txt",
    "conda_environment.yaml",
}
INTERESTING_SUFFIXES = {".py", ".yml", ".yaml", ".md", ".sh"}


def _rel_files(root: Path, limit: int = 400) -> list[str]:
    found = []
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_file() and (path.name in ENV_NAMES or path.suffix.lower() in INTERESTING_SUFFIXES):
            found.append(str(path.relative_to(root)))
            if len(found) >= limit:
                break
    return found


def main() -> int:
    manifest = yaml.safe_load((ROOT / "manifests" / "source_repositories.yaml").read_text(encoding="utf-8"))
    reports = []
    for repo in manifest["repositories"]:
        dest = ROOT / repo["local_path"]
        readme = ""
        for candidate in ("README.md", "README.rst", "README.txt"):
            path = dest / candidate
            if path.is_file():
                readme = "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[:80])
                break
        reports.append(
            {
                "name": repo["name"],
                "clone_status": repo["clone_status"],
                "commit_sha": repo.get("commit_sha"),
                "license": repo.get("license"),
                "submodules": repo.get("submodules"),
                "readme_excerpt": readme,
                "files": _rel_files(dest),
            }
        )
    out = ROOT / "manifests" / "audit_scan.json"
    out.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
