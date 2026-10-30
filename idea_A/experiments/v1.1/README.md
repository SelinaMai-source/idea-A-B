# Idea A v1.1 experiments

冻结协议：`reports/V1_1_EVALUATION_PROTOCOL.md`。

运行：

```bash
PYTHONPATH=. .venv/bin/python idea_A/experiments/v1.1/run.py --tier 1
PYTHONPATH=. .venv/bin/python idea_A/experiments/v1.1/run.py --tier 2
PYTHONPATH=. .venv/bin/python idea_A/experiments/v1.1/run.py --tier ablations
```

或 `scripts/run_A_v1_1.sh`。

每个格子一个 `experiment_id` 目录，含 `config.yaml`、`metrics.json`、`metrics.csv`、`run.log`、`environment.json`、`plots/`。同一种子的方法共用一次 DGP。

结果写在 `idea_A/results/v1.1/`。摘要：`TIER1_SUMMARY.json`、`TIER2_SUMMARY.json`。配对 RMSE 对每次 bootstrap 重算 `sqrt(mean squared error)`，基线是 Uniform cross-fit DR。`n=3` 标 PRELIMINARY。JSON 里不写 PROMISING，除非协议阈值真正满足（本运行器默认只报数字）。
