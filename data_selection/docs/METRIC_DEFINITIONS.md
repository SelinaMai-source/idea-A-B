# 指标

同一套字段在 `evaluation/metrics.py`。不适用写 `N/A`，还没测写 `NOT_RUN`。不把缺测写成 0。`n=1` 时 `ci_justified` 为 false，`ci95` 为 `N/A`。

成功率 AUC 只用预先声明的环境步网格做梯形积分。epoch AUC 和 step AUC 不混用。

`metric_applicability`：

- P0：论文自己的指标，测到之前是 `NOT_RUN`。不从 PDF 抄数字。
- P1：核心指标可用，在线 AUC 是 `N/A`。
- P2：只有真实在线协议才给 AUC 留 `NOT_RUN`。合成循环里的 AUC 留在 `SYNTHETIC_INTERFACE_SMOKE`，不进论文表。
- P3：`forgetting`、`backward_transfer`、`average_success` 只对 `libero_continual`。单任务 RoboMimic 是 `N/A`。
- `forward_transfer` 返回 `NOT_DEFINED`，表里不加这一列的数字。

失败运行和所有 seed 由 `preserve_runs` 保留。`summarize_results.py` 单独计数 `FAILED`，不把它算进论文结果。
