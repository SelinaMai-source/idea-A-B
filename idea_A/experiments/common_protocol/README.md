# Idea A common protocol (adapted)

标签：`adapted_to_common_protocol`。不是 AMPLe 官方复现。

## 比较什么

同一套已有合成选择性反馈环境（`idea_A` simulator），预算 `b=0.05`，机制仅 **MNAR** 与 **zero-overlap**，种子 0/1/2。配对方法：

- Uniform + 审计 IPW
- 经典 Neyman + 探索下界 + 审计 IPW（v1.0，不是新算法）
- 改编的层后验体积 / 最大方差启发式 + 审计 IPW

官方 AMPLe 的 `volume` 采集依赖 RM 打分的偏好对和单纯形上的 MH 后验，需要其 HuggingFace 数据集与 torch，不能在本合成环境里原样调用。这里实现的是数学类比：`v_h = σ_h² / (m_h + λ)`，`q_h ∝ v_h`。

## 不要混排

AMPLe 官方曲线、HyPER、Drago 2026 理论对照 **不进入** 本目录的排行榜。
