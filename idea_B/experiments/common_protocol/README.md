# Idea B common protocol

可比较排行榜（同一环境 B2，预算 5%，种子 0/1/2）：

- ObservabilityController（Idea B v1.0）
- UniformExploration（通用对照）
- GreedyDeployment（通用对照；**不是**官方 Performative SGD / SGD-GD）

## 跳过项

ICML 2024 Performative Bandit 的零阶重参数化 **adapted-skipped / BLOCKED**。官方核心是一维 Bernoulli 映射 `θ ↦ θ²`、两点零阶梯度，以及用 Bernoulli KL 反演的 `LearnModel`。Idea B 是带技能维删失与有限审计预算的线性模型。把 1-D KL 反演硬接到 B2 等于改写论文；把两点 ZO-SGD 贴上 PP-bandit 标签则是假基线。官方数值实验仍在 `idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/`，不进入本榜。
