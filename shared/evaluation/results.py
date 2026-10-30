"""Serialize one run directory from caller-supplied objects.

Writes only ``config.yaml``, ``metrics.json``, ``metrics.csv``, ``run.log``,
and ``environment.json``. Numeric fields are the caller's values; this module
does not add scores, seeds, or summary statistics.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _metric_rows(metrics: Any) -> list[dict[str, Any]]:
    payload = _to_jsonable(metrics)
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise TypeError("list metrics must be a list of mappings")
        return [dict(item) for item in payload]
    if not isinstance(payload, dict):
        raise TypeError("metrics must be a mapping or a list of mappings")
    if len(payload) == 0:
        return []
    values = list(payload.values())
    if all(isinstance(item, list) for item in values):
        lengths = {len(item) for item in values}
        if len(lengths) != 1:
            raise ValueError("list-valued metrics must share one length")
        width = lengths.pop()
        return [{key: item[index] for key, item in payload.items()} for index in range(width)]
    rows: dict[str, Any] = {}
    for key, item in payload.items():
        if isinstance(item, (dict, list)):
            rows[key] = json.dumps(item, ensure_ascii=False, allow_nan=True)
        else:
            rows[key] = item
    return [rows]


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_run(
    result_dir: Any,
    config: dict[str, Any],
    metrics: Any,
    environment: dict[str, Any],
    log_text: str,
) -> Path:
    """Write the five run artifacts under ``result_dir`` and return that path.

    Existing files of those five names are overwritten. Other files in the
    directory, including anything under ``plots/``, are left in place.
    """
    if not isinstance(config, dict):
        raise TypeError("config must be a mapping")
    if not isinstance(environment, dict):
        raise TypeError("environment must be a mapping")
    if not isinstance(log_text, str):
        raise TypeError("log_text must be a str")

    directory = Path(result_dir)
    directory.mkdir(parents=True, exist_ok=True)

    config_payload = _to_jsonable(config)
    metrics_payload = _to_jsonable(metrics)
    environment_payload = _to_jsonable(environment)
    rows = _metric_rows(metrics)

    (directory / "config.yaml").write_text(
        yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_metrics_csv(directory / "metrics.csv", rows)
    (directory / "run.log").write_text(log_text, encoding="utf-8")
    (directory / "environment.json").write_text(
        json.dumps(environment_payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return directory
