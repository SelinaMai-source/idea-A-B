"""Round-trip of the five run artifacts. Values come from the caller."""

import csv
import json

import numpy as np
import yaml

from shared.evaluation.results import write_run


def test_write_run_roundtrip(tmp_path):
    config = {"experiment_id": "unit", "seed": 0, "budget": 0.1, "label": "审计"}
    metrics = {"ipw": np.float64(3.5), "snips": 1.0, "n": np.int64(4)}
    environment = {"python": "3.12", "experiments_run": False}
    log_text = "seed=0\n实验尚未运行\n"
    directory = write_run(tmp_path / "run", config, metrics, environment, log_text)

    assert yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8")) == config
    loaded = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert loaded == {"ipw": 3.5, "snips": 1.0, "n": 4}
    assert json.loads((directory / "environment.json").read_text(encoding="utf-8")) == environment
    assert (directory / "run.log").read_text(encoding="utf-8") == log_text
    with (directory / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["ipw", "snips", "n"]
    assert float(rows[0]["ipw"]) == 3.5
    assert sorted(path.name for path in directory.iterdir()) == [
        "config.yaml",
        "environment.json",
        "metrics.csv",
        "metrics.json",
        "run.log",
    ]


def test_write_run_columnar_metrics(tmp_path):
    write_run(
        tmp_path,
        {"id": "cols"},
        {"seed": [0, 1], "ipw": [1.0, 2.0]},
        {"host": "local"},
        "two seeds\n",
    )
    loaded = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert loaded == {"seed": [0, 1], "ipw": [1.0, 2.0]}
    with (tmp_path / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["ipw"] == "2.0"
    assert rows[0]["seed"] == "0"
