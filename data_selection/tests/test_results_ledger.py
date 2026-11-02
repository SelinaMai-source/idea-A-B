from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ledger_keeps_unpublished_cells_unrun():
    text = (ROOT / "data_selection_v0实验版本+结果.md").read_text(encoding="utf-8")
    lift = text.split("## 2. RoboMimic Can MH")[0]
    assert "NOT_RUN" in lift
    assert "|0.0|" not in lift
    assert "SYNTHETIC_INTERFACE_SMOKE" not in lift
    assert "SYNTHETIC_INTERFACE_SMOKE" in text
    assert "不是论文数字" in text
