# Idea B v1.1 experiments

冻结协议：`reports/V1_1_EVALUATION_PROTOCOL.md`。

运行：

```bash
PYTHONPATH=. .venv/bin/python idea_B/experiments/v1.1/run.py --tier 1
PYTHONPATH=. .venv/bin/python idea_B/experiments/v1.1/run.py --tier 2
PYTHONPATH=. .venv/bin/python idea_B/experiments/v1.1/run.py --tier ablations
```

或 `scripts/run_B_v1_1.sh`。

每个 `(env, method, budget, seed)` 一个目录。同一整数种子在一个格子的所有方法之间共享世界随机数。不写巨大 `.npy` / `.npz`。

结果：`idea_B/results/v1.1/`，摘要 `TIER1_SUMMARY.json`、`TIER2_SUMMARY.json`。配对 bootstrap 对 E-optimal 与 D-optimal，指标为 late worst-skill MSE 与 cumulative regret。更高特征值单独出现不算成功。
