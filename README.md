# Idea A / Idea B 可行性烟测 v1

本地根目录：`/Users/grantwright/Research/idea-A-B`。目标分支：`v1`。远程：https://github.com/SelinaMai-source/idea-A-B.git

本仓库是 **research feasibility smoke test**，不是已完成的 ICML / NeurIPS / ACL 投稿。数字只来自磁盘上的 JSON/CSV。`n_seeds=3` 只是初步证据。

- **Idea A**：Adaptive Identification under Endogenous Feedback Censoring。v1.0 分配是经典分层 **Neyman allocation** + 探索下界，估计用已知审计倾向的 **IPW / Hájek / AIPW**。**不是新算法。**
- **Idea B**：Feedback-Induced Observability Collapse。信息量是线性模型观测设计 Gram 矩阵 \(W_T=\sum_t S_t\phi_t\phi_t^\top\)，不是动力系统 observability Gramian。

当前烟测判定（由 `scripts/generate_report.py` 按预先规则从 JSON 算出，不是事后改口）：

| Idea | Verdict | 一句话 |
| --- | --- | --- |
| A | **INCONCLUSIVE** | MNAR 下审计 IPW 明显降偏；Neyman 相对 Uniform 的 RMSE 配对区间不支持改进。 |
| B | **INCONCLUSIVE** | 控制器提高 \(\lambda_{\min}\)（配对 CI 排除 0）；B2 联合内生坍缩未成立；真实表现 CI 含 0。 |

主报告：[`reports/A_B_SMOKE_EXPERIMENT_REPORT.md`](reports/A_B_SMOKE_EXPERIMENT_REPORT.md)

## Environment / 环境

- macOS arm64，无 NVIDIA GPU。主实验用 NumPy/SciPy，**不要求 GPU**。
- 主 venv：`/Users/grantwright/Research/idea-A-B/.venv/bin/python`（Python 3.12；**无 torch**）。
- 隔离 venv（官方 AMPLe / Qwen / datasets）：`.isolated_baselines_venv`。
- 不安装 vLLM、bitsandbytes、flash-attn、CUDA、NCCL。

```bash
cd /Users/grantwright/Research/idea-A-B
bash scripts/setup_mac.sh
.venv/bin/python scripts/verify_installation.py
.venv/bin/python -m shared.hardware
```

`setup_mac.sh` 会建 `.venv` 并安装 NumPy 技术栈，然后尝试安装 torch；失败时保留 NumPy 环境。硬件清单：

- `reports/environment_manifest.json`
- `shared/hardware/environment_manifest.json`

## Install / Run / Report

仓库根必须在 `PYTHONPATH` 上。

```bash
cd /Users/grantwright/Research/idea-A-B
export PYTHONPATH="$PWD"

# Idea A synthetic matrix + ablations A1–A5
.venv/bin/python -m idea_A.experiments.run_synthetic
.venv/bin/python idea_A/ablations/A1_No_Adaptive_Allocation/run.py
.venv/bin/python idea_A/ablations/A2_No_Propensity_Correction/run.py
.venv/bin/python idea_A/ablations/A3_No_Anchor_Intervention/run.py
.venv/bin/python idea_A/ablations/A4_Fixed_vs_Adaptive_Budget/run.py
.venv/bin/python idea_A/ablations/A5_Misspecified_Propensity/run.py
.venv/bin/python idea_A/experiments/common_protocol/run.py

# Idea B closed-loop B1–B3 + ablations B1–B5
.venv/bin/python -m idea_B.experiments.run_synthetic
.venv/bin/python idea_B/ablations/B1_No_Observability_Control/run.py
.venv/bin/python idea_B/ablations/B2_No_Adaptive_Exploration/run.py
.venv/bin/python idea_B/ablations/B3_Fixed_vs_Adaptive_Feedback/run.py
.venv/bin/python idea_B/ablations/B4_No_Policy_Feedback_Loop/run.py
.venv/bin/python idea_B/ablations/B5_Observability_Metric_Variants/run.py
.venv/bin/python idea_B/experiments/common_protocol/run.py

# Tests (no GPU required)
.venv/bin/python -m pytest

# Rebuild the three Chinese reports from JSON
.venv/bin/python scripts/generate_report.py
```

真实数据 / 模型（隔离 venv，不是合成矩阵的一部分）：

```bash
.isolated_baselines_venv/bin/python scripts/download_benchmarks.py
.isolated_baselines_venv/bin/python scripts/download_models.py
```

PP-bandit 官方 1-D 数值实验：

```bash
.venv/bin/python idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/run_official_numerical.py
```

## Blocked / partial / 不要声称的事项

| Item | Status | 原因 |
| --- | --- | --- |
| AMPLe official | **partial** | `n_rounds=3` 而非 100；2-aspect 数据集 HTTP 401；acquisition 仍为 `volume`，`burnin=50000` 未改 |
| HyPER official | **blocked** | TPU v3-128、vLLM、8B–13B RM/DPO、Beaker/GCS；未发明简化版 HyPER |
| Drago et al. ICML 2026 | **blocked** | 无官方代码 |
| PP-bandit official | **complete** | 1-D reparameterized ZO demo；**NOT DIRECTLY COMPARABLE** 到 Idea B |
| PP-bandit adapted to B2 | **blocked** | 无法忠实把 1-D Bernoulli KL reparam 接到 3-D 技能删失 |
| ncvxpp / Performative SGD | **partial** | 官方 `mpiexec -np 10` 缺 mpi4py/mpicc；单进程 stub 不是 10-replica 复现 |
| RewardBench leaderboard | **skipped** | 没有兼容的小 reward model；Qwen2.5-0.5B-Instruct 是因果 LM |
| Qwen MPS `generate` | **partial** | MPS abort；同一 revision 的 CPU float32 generate 成功 |
| CCF-A 论文对比 | **not claimed** | 协议不可比时标 `NOT DIRECTLY COMPARABLE`，不进同一排行榜 |

## Artifacts

- Idea A 汇总：`idea_A/results/SMOKE_SUMMARY.json`
- Idea B 汇总：`idea_B/results/SMOKE_SUMMARY.json`
- 每个 run：`results/<experiment_id>/{config.yaml,metrics.json,metrics.csv,run.log,environment.json,plots/}`
- 图：`idea_A/results/aggregate_plots/`，`idea_B/results/plots/`
- 报告：`reports/A_B_SMOKE_EXPERIMENT_REPORT.md`，`reports/BASELINE_REPRODUCTION_AUDIT.md`，`reports/NOVELTY_AND_FAILURE_ANALYSIS.md`
- 模型权重（gitignore）：`.cache/models/Qwen2.5-0.5B-Instruct/`；`idea_A/models/` 与 `idea_B/models/` 只有 manifest 和相对符号链接
- MultiPref 小子集：`idea_A/benchmarks/MultiPref/`
- RewardBench schema sample：`idea_A/benchmarks/RewardBench/`

不提交：`.venv/`、`.isolated_baselines_venv/`、`.cache/`、权重、HF hub、checkpoint、`.env`、token。
