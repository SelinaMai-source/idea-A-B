# v1.1 Next Research Steps

根据冻结协议与 TIER2 JSON，而不是根据想投哪个会。

当前：Idea A v1.1 = **NOT PROMISING**，Idea B v1.1 = **NOT PROMISING**。

## 不要做的事

- 不要把 PROMISING 用在任一 v1.1 主提案上。
- 不要只因为赢了 Uniform 随机 / observed-only / Greedy 就改口。
- 不要重跑 AMPLe n_rounds=100、HyPER TPU、ncvxpp mpiexec、或发明简化 HyPER。
- 不要把 PP-bandit 1-D demo 接到技能删失环境。
- 不要把 MultiPref / RewardBench / Qwen 烟测写进合成主结论。
- 不要在评价种子上调 `drift_kappa` / `information_discount`。
- 不要把特征值单独当成功。`higher_eigenvalue_is_not_success=true`。

## Idea A：值得跟的

1. 把 identification-via-audit 写成明确的 MNAR + 已知 q 审计理论，而不是再调 Neyman。
2. 诊断闭环上 v1.1c RMSE 恶化：哪些种子覆盖失败、倾向是否过小、history-respecting nuisance 是否不稳定。
3. 若继续分配研究，主对照必须仍是 Uniform cross-fit DR；A-v1.1b 已显示漂移项没有带来确认性增益。
4. JSON 已跳过消融 30 种子扩展：`['b=0.01 and b=0.20 not in confirmatory matrix', 'full Cartesian product not run', 'b=0.01 and b=0.20 leftover-time budgets not run', 'A ablation 30-seed expansion skipped: confirmatory v1.1c vs Uniform DR already decided on the main 30/50-seed cells']`。不要为了叙事再展开笛卡尔积。

## Idea B：值得跟的

1. 把 E-optimal 当作充分的经典对照。若新控制器不能在 worst-skill MSE 上过它，就停。
2. Env3：右删失使恢复时间 INCONCLUSIVE。加长 horizon 只允许走协议第 5 节预先登记，且不能按谁恢复更快来选。当前 locked_horizon 已因 Oracle 仍删失而保持 32。
3. 分开报告三种 Gramian；继续禁止用 `1/(1+MSE)` 做 PROMISING。
4. JSON 已跳过 B4/B5/B6 的 30 种子扩展：`['b=0.01 and b=0.20 leftover-time only; not in main conclusion', 'b=0.01 and b=0.20 leftover-time budgets not run', 'B4/B5/B6 30-seed ablation expansion skipped: H_B1 already identified on the main Env1/Env2 matrix']`。

## 证据门槛

- 主结论继续只用预先指定的 30/50 种子格子。
- `n=3` 只能标 PRELIMINARY。
- 配对区间覆盖 0 → INCONCLUSIVE；完全不利 → NOT PROMISING；PROMISING 需要协议第 3 节的全部条件。
