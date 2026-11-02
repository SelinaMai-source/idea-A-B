"""Plot only series that were actually passed in. Missing matplotlib is recorded."""

from __future__ import annotations

import json
from pathlib import Path


def plot_success_curve(curve: list[tuple[int, float]], output_png: Path) -> dict:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        return {"status": "NOT_GENERATED", "reason": f"{type(exc).__name__}: {exc}"}
    if len(curve) < 2:
        return {"status": "NOT_GENERATED", "reason": "curve has fewer than two points"}
    xs = [step for step, _rate in curve]
    ys = [rate for _step, rate in curve]
    fig, axis = plt.subplots(figsize=(4, 3))
    axis.plot(xs, ys, marker="o")
    axis.set_ylim(0, 1)
    axis.set_xlabel("environment steps")
    axis.set_ylabel("success")
    fig.tight_layout()
    fig.savefig(output_png)
    plt.close(fig)
    return {"status": "WRITTEN", "path": str(output_png)}


def write_curve_csv(curve: list[tuple[int, float]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = ["environment_step,success_rate"]
    lines.extend(f"{step},{rate}" for step, rate in curve)
    output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
