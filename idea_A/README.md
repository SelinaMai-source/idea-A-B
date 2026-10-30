# Idea A

**Adaptive Identification under Endogenous Feedback Censoring.**

v1.0 分配规则是**分层 Neyman 分配加探索下界**，再用**已知审计倾向**做 IPW / Hájek / AIPW。这是经典两阶段抽样，**不是**新算法。要检验的是：在内生删失下，这条已知审计通道能否恢复目标量，以及同等预算下是否优于均匀审计。

自然观测若依赖未观测的 `Y`（MNAR）或存在零重叠，该通道不可识别。那是识别失败，不是估计器 bug。算法只能在 `O_t = S_t OR A_t` 时读取标签。

## 命令

在仓库根目录、使用本仓库的 Python 3.12：

```bash
cd ~/Research/idea-A-B
export PYTHONPATH="$PWD"

# 合成矩阵（机制 × 预算 × 种子）并写 idea_A/results/
.venv/bin/python -m idea_A.experiments.run_synthetic

# 消融 A1–A5（全部机制在 b=0.05，以及 MNAR / zero-overlap 的全部预算）
.venv/bin/python idea_A/ablations/A1_No_Adaptive_Allocation/run.py
.venv/bin/python idea_A/ablations/A2_No_Propensity_Correction/run.py
.venv/bin/python idea_A/ablations/A3_No_Anchor_Intervention/run.py
.venv/bin/python idea_A/ablations/A4_Fixed_vs_Adaptive_Budget/run.py
.venv/bin/python idea_A/ablations/A5_Misspecified_Propensity/run.py

# 一次跑完矩阵 + 消融，并刷新 SMOKE_SUMMARY.json
.venv/bin/python -m idea_A.experiments.run_all

# 正确性测试
.venv/bin/python -m pytest idea_A/tests -q
```

可选：`IDEA_A_SUBSET=1` 时主矩阵也改成「全部机制在 b=0.05 + MNAR/zero-overlap 的全部预算」。默认跑全矩阵。

不要从本目录下载 LLM、MultiPref 或 RewardBench，也不要写入 `baselines/AMPLe/` 或 `baselines/HyPER/`。

## 产物

- 每次运行：`idea_A/results/<experiment_id>/{config.yaml,metrics.json,metrics.csv,run.log,environment.json,plots/}`
- 汇总：`idea_A/results/SMOKE_SUMMARY.json`
- 对照图：`idea_A/results/aggregate_plots/`
- 消融：`idea_A/ablations/A*/results/`

## 估计量

总体均值奖励、偏好率 `P(Y>0)`、层条件均值、在**均匀层分布**这一既定目标分布下的策略价值。配对比较使用同一模拟世界。
