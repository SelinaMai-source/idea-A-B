# Sequential Closed Loop

每轮用已观测数据更新策略（Continual）或使用外生策略日程（Static），下一轮自然观测概率依赖当前 `π`。总审计预算固定为 `b * n`。

两条实验轨道必须分开：

- `static_estimation/`：估计已实现有限总体的 `E[Y]`。策略不由结果模型更新。`policy_drift` 在中点外生翻转 `π`。
- `continual_adaptation/`：每轮用 `O=S∨A` 上的标签更新 linear/logistic 策略，下一轮 `S` 依赖新 `π`。

场景：`policy_drift`（`S` 依赖 `π`）、`mnar_audit`（`S` 依赖未观测 `Y`；识别走已知审计 `q`）。
