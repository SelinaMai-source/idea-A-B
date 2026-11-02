# 指标定义

RoboMimic 与 LIBERO 共用同一套字段。不适用写 `N/A`。没跑写 `NOT_RUN`。这两个字符串都不等于 0。

## 核心字段

`final_success_rate`、`success_auc`、`selected_episodes`、`selected_transitions`、`retention_ratio`、`environment_steps`、`training_optimizer_steps`、`selection_wallclock_seconds`、`training_wallclock_seconds`、`rollout_wallclock_seconds`、`total_wallclock_seconds`、`gpu_hours`、`cpu_hours`、`expert_query_count`、`seed`、`benchmark`、`task`、`policy_model`、`method`、`protocol`、`adaptation_status`。

`success_auc` 只在预先声明的环境步上做梯形积分。训练 epoch 的曲线不能和环境步曲线比。可选再除以 horizon，默认不除。

选择预算同时报告 episode 数和 transition 数。STRAP 选的是子轨迹，在官方入口真正跑起来之前，不能把 episode 预算当成它的结果。

## 遗忘只用于任务增量

记 `A[t, j]` 为第 t 轮之后任务 j 的成功率。任务 j 在第 j 轮训练。只有 `A[j, j]` 有值才算学过。

- 平均成功率：最后一行的均值。最后一行有空位就报错，不把空位当 0。
- 遗忘：对最后一个任务之外、且已经学过的任务，取学习之后到倒数第二轮的峰值，减去最终成功率，再平均。
- 后向迁移：`A[T-1, j] - A[j, j]` 的平均。

单任务 RoboMimic 的遗忘列是 `N/A`。LIBERO-10 原文协议也不是这里自建的任务序列，遗忘列同样是 `N/A`。只有 `libero_continual` 适用，而它目前是 `NOT_RUN`。

## 统计

n=1 不给置信区间。n≥2 时标准差用样本标准差，区间是正态近似 z=1.96，并在记录里写明。一次烟测不能写成复现了论文。
