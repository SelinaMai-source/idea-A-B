"""The environment probe returns measured fields and no experiment metrics."""

from shared.hardware.probe import collect_environment


def test_collect_environment_schema():
    snapshot = collect_environment()
    assert snapshot["os"]["system"]
    assert snapshot["os"]["machine"]
    assert snapshot["python"]["version"]
    assert snapshot["python"]["executable"]
    assert "ram_bytes" in snapshot
    assert snapshot["disk"]["free_bytes"] > 0
    assert "mps_available" in snapshot["torch"]
    assert snapshot["torch"]["mps_functional_check"] == "not_run"
    assert snapshot["git"]["version"]
    assert "numpy" in snapshot["packages"]
    assert snapshot["experiments_run"] is False
