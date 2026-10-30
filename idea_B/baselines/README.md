# Idea B baselines

`Performative_Bandit_ICML2024/` 与 `Performative_SGD_NeurIPS2024/` 留给官方仓库浅克隆，本目录的通用对照**不写入**这两个文件夹。

下面这些是合成闭环协议里的 **generic controls**，不是 ICML 2024 / NeurIPS 2024 官方脚本的复现，也不要把它们的曲线标成 `official_reproduction`。

| 目录 | 角色 |
| --- | --- |
| `Uniform_Exploration/` | 均匀审计/探索 |
| `Greedy_Deployment/` | 把预算打在当前预测最高的 query 上 |
| `Lazy_Deployment/` | 贪心部署，参数隔若干轮才更新 |
| `Standard_Online_SGD/` | 均匀观测 + 在线 SGD |
| `Fixed_Exploration_Schedule/` | 固定探索率 |
| `Uncertainty_based_Exploration/` | 按 \(\phi^\top(W+\lambda I)^{-1}\phi\) 分配 |
| `No_Feedback_Controller/` | 不使用额外审计预算 |
| `Full_Feedback_Oracle/` | 全反馈上界 |

实现入口是各目录的 `algorithm.py`，分配规则集中在 `idea_B/methods/v1.0/feedback_controller.py`，避免把对照写进官方 clone 目录。
