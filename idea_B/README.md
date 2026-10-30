# Idea B — Feedback-Induced Observability Collapse

合成闭环烟测：线性模型的观测设计 Gram 矩阵

\[
W_T=\sum_t S_t\phi_t\phi_t^\top
\]

以及条件期望版本 \(W_T^{\mathrm{cond}}=\sum_t q_t\phi_t\phi_t^\top\)。这不是任意矩阵，也不叫动力系统的可观性 Gramian。

B2 是真闭环：当前参数给 query 打分 → 策略决定谁被服务、谁产生自然反馈 → 观测数据更新参数 → 新参数成为下一轮策略。B1 是外生缺失诊断。固定策略加固定抽样不会被标成内生坍缩。信息量和真实总体表现分开记录，不假设前者自动带来后者。

通用对照（Uniform / Greedy / Lazy / SGD / 固定探索 / 不确定度 / 无控制器 / Oracle）标成 **generic controls**，不是 ICML 2024 / NeurIPS 2024 官方仓库的复现。本阶段不克隆、不改写 `baselines/Performative_Bandit_ICML2024` 与 `baselines/Performative_SGD_NeurIPS2024`。

## 命令

仓库根目录，Python 3.12：

```bash
cd ~/Research/idea-A-B
export PYTHONPATH=.

# 正确性测试
.venv/bin/python -m pytest idea_B/tests -q

# 合成闭环主实验（B1/B2/B3，种子 0,1,2；先全部环境 5%，B2 全预算，时间允许再补其余预算）
.venv/bin/python -m idea_B.experiments.run_synthetic

# 消融 B1–B5
.venv/bin/python idea_B/ablations/B1_No_Observability_Control/run.py
.venv/bin/python idea_B/ablations/B2_No_Adaptive_Exploration/run.py
.venv/bin/python idea_B/ablations/B3_Fixed_vs_Adaptive_Feedback/run.py
.venv/bin/python idea_B/ablations/B4_No_Policy_Feedback_Loop/run.py
.venv/bin/python idea_B/ablations/B5_Observability_Metric_Variants/run.py
```

主实验产物在 `idea_B/results/<experiment_id>/`，含 `config.yaml`、`metrics.json`、`metrics.csv`、`run.log`、`environment.json`、`plots/`、`trajectories.json` 与 `trajectories.npz`。汇总在 `idea_B/results/SMOKE_SUMMARY.json`。

不下载 LLM，不安装 torch。不写 `reports/A_B_SMOKE_EXPERIMENT_REPORT.md`。
