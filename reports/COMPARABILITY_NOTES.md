# 可比性说明（不是排行榜）

规则：目标、信息、预算对不齐的方法标 `NOT DIRECTLY COMPARABLE`，不进同一张表。数字只来自 JSON，不手填。

## 两张允许并排的表

1. **Idea A 共同协议**（`idea_A/experiments/common_protocol/`）  
   同一合成选择性反馈世界，`b=0.05`，仅 MNAR 与 zero-overlap，种子 0/1/2：Uniform IPW、Neyman IPW、改编体积启发式 IPW。  
   Uniform 的审计 IPW 不读自然通道 `S`。给定种子时 `Y` 在机制之前生成，因此 Uniform 在 MNAR 与 zero-overlap 上的审计 IPW 数字相同，这是设计如此，不是复制错误。

2. **Idea B 共同协议**（`idea_B/experiments/common_protocol/`）  
   环境 B2，5% 预算，种子 0/1/2：ObservabilityController、UniformExploration、GreedyDeployment。  
   Greedy 是通用对照，**不是**官方 Performative SGD / SGD-GD。

## 明确不可比

| 方法 | 状态 | 原因 |
|---|---|---|
| AMPLe 官方 | NOT DIRECTLY COMPARABLE | 多维偏好查询效率，不是内生删失下的审计识别 |
| AMPLe 改编体积启发式 | partial | 可与 Uniform/Neyman 同环境比分配；不是 AMPLe 官方 |
| HyPER | NOT DIRECTLY COMPARABLE，BLOCKED | 人/模型标注路由 + RewardBench；未发明简化版 |
| Drago 2026 | NOT DIRECTLY COMPARABLE | 理论对照，无官方代码 |
| PP-bandit 官方 | NOT DIRECTLY COMPARABLE | 1-D 零阶重参数化演示，无技能维删失 |
| PP-bandit 改编 | adapted-skipped / BLOCKED | 1-D Bernoulli `y²` + KL `LearnModel` 无法忠实接到 B2 |
| ncvxpp / Performative SGD | NOT DIRECTLY COMPARABLE | 决策依赖分布，不是 Idea B 的技能删失；官方 10 副本被 mpi4py 阻断 |

## 跳过

- 未发明简化 HyPER。  
- 未把 Greedy/Lazy 改贴成官方 SGD-GD。  
- 未把两点 ZO-SGD 改贴成改编 PP-bandit。
