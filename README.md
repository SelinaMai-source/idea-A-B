**Current Idea C v0 / v0.1 Experimental Results:** [`idea_C/v0实验版本+结果.md`](idea_C/v0实验版本+结果.md) · v0.1 对照 [`idea_C/reports/v0.1/PAPER_VS_REPRODUCTION.md`](idea_C/reports/v0.1/PAPER_VS_REPRODUCTION.md) · 图 [`idea_C/results/v0.1/figures/`](idea_C/results/v0.1/figures/)

# Idea A / Idea B：v1 烟测与 v1.1 确认性比较

本地根目录：`/Users/grantwright/Desktop/科研/10.30新课题/v0`。目标分支：`v1.1`（v1 烟测在 `v1`）。远程：https://github.com/SelinaMai-source/idea-A-B.git

本仓库是 **research feasibility / confirmatory comparison**，不是已完成的 ICML / NeurIPS / ACL 投稿。数字只来自磁盘上的 JSON/CSV。`n_seeds=3` 只是初步证据；v1.1 主结论只用 30/50 种子的预先指定格子。

- **Idea A**：Adaptive Identification under Endogenous Feedback Censoring。v1.0 分配是经典分层 **Neyman allocation** + 探索下界，估计用已知审计倾向的 **IPW / Hájek / AIPW**。v1.1a 是 Uniform + cross-fit DR（主对照）；v1.1b/c 是 AIPW 方差分配（经典近似，**不是新最优设计定理**）。
- **Idea B**：Feedback-Induced Observability Collapse。信息量是线性模型观测设计 Gram 矩阵 \(W_T=\sum_t S_t\phi_t\phi_t^\top\)，不是动力系统 observability Gramian。E-optimal / D-optimal 是**经典对照**。

v1 烟测判定（由 `scripts/generate_report.py` 按预先规则从 JSON 算出）：

| Idea | Verdict | 一句话 |
| --- | --- | --- |
| A v1.0 | **INCONCLUSIVE** | MNAR 下审计 IPW 明显降偏；Neyman 相对 Uniform 的 RMSE 配对区间不支持改进。 |
| B v1.0 | **INCONCLUSIVE** | 控制器提高 \(\lambda_{\min}\)（配对 CI 排除 0）；B2 联合内生坍缩未成立；真实表现 CI 含 0。 |

v1.1 判定（由 `scripts/generate_v1_1_report.py` 按冻结协议 `reports/V1_1_EVALUATION_PROTOCOL.md` 从 TIER2 JSON 算出；**PROMISING 不能表示只打败均匀随机**）：

| Idea | Verdict | 一句话 |
| --- | --- | --- |
| A v1.1 | **NOT PROMISING** | identification-via-audit 仍支持；v1.1c 序贯分配未赢 Uniform DR，闭环格子更差。 |
| B v1.1 | **NOT PROMISING** | v1.1c 未在 worst-skill MSE 上赢 E-optimal；H_B1 要求 worst-skill **并且** regret 同时优于 E 与 D。 |

v1.2 判定（由 `scripts/generate_v1_2_report.py` 从锁定审计与已有 JSON 写出；**DIRECT = 0，可辩护 ADAPTED = 0**。不得把 Uniform DR / E-optimal 的内部结果当成击败 CCF-A 论文）：

| 问题 | Verdict |
| --- | --- |
| 相对已发表 CCF-A Main | **PUBLISHED BASELINE COMPARISON NOT ESTABLISHED** |
| Idea A vs DRTS / DeltaUCB / AMPLe / HyPER | 无合法 head-to-head（Protocol A2 BLOCKED / INCOMPARABLE） |
| Idea B vs PP-bandit / ncvxpp | 无合法 head-to-head（Protocol B2 INCOMPARABLE） |
| A/B 是否 PROMISING | **否**（v1.1 已相对经典对照判定 NOT PROMISING；v1.2 不改口） |
| v1.3 算法迭代 | **不成立**（本机无 live RM UQ / skill-free 映射） |

v1 主报告：[`reports/A_B_SMOKE_EXPERIMENT_REPORT.md`](reports/A_B_SMOKE_EXPERIMENT_REPORT.md)

v1.1 主报告：[`reports/V1_1_FULL_EXPERIMENT_REPORT.md`](reports/V1_1_FULL_EXPERIMENT_REPORT.md)

v1.2 主报告：[`reports/v1.2/CCFA_PUBLISHED_BASELINE_COMPARISON.md`](reports/v1.2/CCFA_PUBLISHED_BASELINE_COMPARISON.md)

## Environment / 环境

- macOS arm64，无 NVIDIA GPU。主实验用 NumPy/SciPy，**不要求 GPU**。
- 主 venv：工作副本 `.venv/bin/python`（Python 3.12；**无 torch**）。历史 JSON 里可能仍写着 `/Users/grantwright/Research/idea-A-B`。
- 隔离 venv（官方 AMPLe / Qwen / datasets）：`.isolated_baselines_venv`。
- 不安装 vLLM、bitsandbytes、flash-attn、CUDA、NCCL。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
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
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD"

# --- v1.1 confirmatory matrix (frozen protocol) ---
bash scripts/run_A_v1_1.sh
bash scripts/run_B_v1_1.sh
.venv/bin/python scripts/generate_v1_1_report.py

# --- v1 smoke matrix (kept) ---
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
.venv/bin/python -m pytest --import-mode=importlib

# Rebuild v1 Chinese smoke reports from JSON
.venv/bin/python scripts/generate_report.py

# Rebuild v1.1 Chinese reports from JSON
.venv/bin/python scripts/generate_v1_1_report.py

# Rebuild v1.2 Chinese published-baseline reports from locked audits / JSON
# (does not run Protocol A2/B2; those cells are INCOMPARABLE)
.venv/bin/python scripts/generate_v1_2_report.py
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
| v1.2 published head-to-head | **PUBLISHED BASELINE COMPARISON NOT ESTABLISHED** | DIRECT=0，ADAPTED=0；Protocol A2/B2 INCOMPARABLE/BLOCKED；selector_unit_check 不是对照 |

## Artifacts

- Idea A v1 汇总：`idea_A/results/SMOKE_SUMMARY.json`
- Idea B v1 汇总：`idea_B/results/SMOKE_SUMMARY.json`
- Idea A v1.1：`idea_A/results/v1.1/TIER1_SUMMARY.json`，`TIER2_SUMMARY.json`
- Idea B v1.1：`idea_B/results/v1.1/TIER1_SUMMARY.json`，`TIER2_SUMMARY.json`
- 每个 run：`results/<experiment_id>/{config.yaml,metrics.json,metrics.csv,run.log,environment.json,plots/}`
- 图：`idea_A/results/aggregate_plots/`，`idea_B/results/plots/`；v1.1 轻量汇总图在 `idea_A/results/v1.1/plots/` 与 `idea_B/results/v1.1/plots/`
- v1 报告：`reports/A_B_SMOKE_EXPERIMENT_REPORT.md`，`reports/BASELINE_REPRODUCTION_AUDIT.md`，`reports/NOVELTY_AND_FAILURE_ANALYSIS.md`
- v1.1 报告：`reports/V1_1_FULL_EXPERIMENT_REPORT.md`，`reports/V1_1_BASELINE_REPRODUCTION.md`，`reports/V1_1_NOVELTY_ASSESSMENT.md`，`reports/V1_1_NEXT_RESEARCH_STEPS.md`；协议 `reports/V1_1_EVALUATION_PROTOCOL.md`；审计 `reports/V1_TO_V1_1_AUDIT.md`
- v1.2 报告：`reports/v1.2/CCFA_PUBLISHED_BASELINE_COMPARISON.md`，`reports/v1.2/ORIGINAL_BASELINE_REPRODUCTION.md`，`reports/v1.2/BENCHMARK_COMPATIBILITY.md`，`reports/v1.2/A_B_RESEARCH_DECISION.md`，`reports/v1.2/comparison_summary.json`；锁文件 `reports/v1.2/00_EXISTING_EVIDENCE_AUDIT.md`–`03_COMPARABILITY_AUDIT.md`
- v1.2 selector_unit_check（不是对照格）：`results/v1.2/idea_A/original/activeuf_selector_unit_check/`
- 模型权重（gitignore）：`.cache/models/Qwen2.5-0.5B-Instruct/`；`idea_A/models/` 与 `idea_B/models/` 只有 manifest 和相对符号链接
- MultiPref 小子集：`idea_A/benchmarks/MultiPref/`
- RewardBench schema sample：`idea_A/benchmarks/RewardBench/`

不提交：`.venv/`、`.isolated_baselines_venv/`、`.cache/`、权重、HF hub、checkpoint、`.env`、token。
