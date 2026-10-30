# Idea A baselines

`AMPLe/` 与 `HyPER/` 留给官方浅克隆，本目录的适配器**不写入**这两个文件夹。

统计基线（本仓库实现，不是官方 clone）：

- `Uniform_Audit/`：`q_t = b`。
- `IPS/`：已知审计倾向上的 Horvitz–Thompson。
- `SNIPS/`：Hájek / SNIPS。
- `Doubly_Robust/`：在 `A=1` 上拟合结果模型，再调用 `dr_mean`（`A=0` 不读 `Y`）。

其余统计对照（Observed-only、Fixed Stratified Audit、Uncertainty Sampling、Full Feedback Oracle、naive IPS）在 `idea_A/experiments/protocol.py` 中实现，同样不进入 `AMPLe/` 或 `HyPER/`。
