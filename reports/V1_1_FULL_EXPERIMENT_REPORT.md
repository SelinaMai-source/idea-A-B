# Idea A / Idea B v1.1 完整实验报告

本文件由 `scripts/generate_v1_1_report.py` 从磁盘 JSON 生成。报告中的数字均来自下列产物，没有手填。

- 生成时间（UTC）：`2026-10-30T18:04:18.057320+00:00`
- 仓库：`/Users/grantwright/Desktop/科研/10.30新课题/v0`
- 分支：`v1.1`
- commit：`7de4bd5d1545f355a87444fea9fae42a9bf4a3cb`
- remote：`git@github.com:SelinaMai-source/idea-A-B.git`
- 缺失的主 JSON：无（所列主源均在磁盘上）
- 冻结协议：`reports/V1_1_EVALUATION_PROTOCOL.md`
- v1→v1.1 审计（本脚本不覆盖）：`reports/V1_TO_V1_1_AUDIT.md`

**判定规则预先冻结，不因结果好坏改口。** PROMISING 不能表示只打败了均匀随机。Idea A 的主对照是 Uniform audit + 2 折 cross-fit DR。Idea B 的主对照是 E-optimal 与 D-optimal；H_B1 要求 worst-skill MSE **并且** cumulative regret 相对 E 与 D 的配对区间都完全为负。`n_seeds<=3` 标 PRELIMINARY。主结论只用 30/50 种子的预先指定格子。

A-v1.1b 的方差分配与 B 的 E/D-optimal 是经典实验设计，不是新算法。

## 1. Executive Summary

- **Idea A v1.1 状态：`NOT PROMISING`**。主提案是 A-v1.1c（AIPW 方差分配 + 剩余预算 + 策略漂移）。
- **Idea B v1.1 状态：`NOT PROMISING`**。主提案是 B-v1.1c（信息折扣 + 一步 worst-skill 收益）。
- v1.0 **identification-via-audit 仍然支持**：v1 烟测 MNAR `b=0.05` observed-only bias mean = `1.1897`，Uniform IPW bias mean = `-0.104575`（`n_seeds=3`，PRELIMINARY）。 v1.1 静态 MNAR `b=0.05` 把该现象放到确认性格子：observed-only RMSE = `1.21739`（n=`50`），Uniform DR RMSE = `0.128779`。 identification_v11_supported=`true`。
- v1.1 **sequential allocation 不被支持**。静态 MNAR `b=0.05` n=`50`：v1.1c RMSE = `0.333698` vs Uniform DR `0.128779`，diff = `0.204919`，CI [`-0.0384978`, `0.442027`]，`interval_side=covers_zero` → `INCONCLUSIVE`。 闭环 MNAR-audit `b=0.05` n=`30`：v1.1c RMSE = `1.07494` vs Uniform DR `0.168591`，diff = `0.906352`，CI [`0.0565902`, `1.67815`]，`interval_side=entirely_positive` → `NOT PROMISING`（v1.1c 更差）。
- 探索性 10% RMSE 下降：**失败**。`exploratory_10pct_failed=true`。 四条闭环格子的 `hit_10pct_drop` 均为 false（见第 5 节）。达不到就记失败；达到也不会自动成为 PROMISING。
- Idea B H_B1 **未满足**。Env1 n=`30`：v1.1c late worst-skill MSE mean = `0.0322726` vs E-optimal `0.0315452`，diff = `0.000727427`，CI [`6.74253e-05`, `0.0014246`]，`entirely_positive`（v1.1c 更差）。 同格 regret diff = `-0.0358101`，CI [`-0.0647617`, `-0.00962598`]，`entirely_negative`（regret 更好，但 H_B1 要求 worst-skill **并且** regret 同时优于 E 与 D）。 Env2 n=`50` worst-skill diff = `0.000908411`，CI [`0.000365364`, `0.00154586`]，`entirely_positive`。
- Env3 恢复时间：**INCONCLUSIVE**。可恢复格 v1.1c censored fraction = `0.9`，E-optimal = `1`；两边都超过一半右删失。配对 recovery CI [`-0.3`, `0`]，`covers_zero`。
- 官方基线未在 v1.1 重跑。沿用 v1：AMPLe=`partial`，HyPER=`blocked`，Drago=`blocked`（无代码），PP-bandit=`complete`（1-D demo，NOT DIRECTLY COMPARABLE），ncvxpp=`partial`。不得声称 outperform 这些论文。
- 噪声底（Idea B）：`0.0225`（`noise_std=0.15`）。更高特征值不是学习成功。
- 初步建议：两个 v1.1 主提案都 **不能** 标 PROMISING。继续跟的是 Idea A 的审计识别通道，而不是自适应分配；Idea B 的经典 E-optimal 对照已经是更强的主指标基线。

## 2. Hardware and Software Environment

环境数字来自一次 v1.1 Tier 1 运行写入的 `environment.json`，不是事后改写的 dirty manifest。

- recorded_at_utc：`2026-10-30T17:48:55.389959+00:00`
- working_copy：`/Users/grantwright/Desktop/科研/10.30新课题/v0`
- python executable：`/Users/grantwright/Desktop/科研/10.30新课题/v0/.venv/bin/python`
- python version：`3.12.4`
- OS：`macOS-14.5-arm64-arm-64bit`
- ram_gib：`16`
- disk.path：`/Users/grantwright/Desktop/科研/10.30新课题/v0`
- disk.free_gib（该次运行时）：`75.5004`
- torch.installed：`false`；import_error：`ModuleNotFoundError: No module named 'torch'`
- packages.numpy：`2.5.4`；scipy：`1.18.1`；pytest：`9.2.0`
- 该 environment.json 记录的 git.branch：`v1.1`；git.commit：`8abd059c3007d5c32631c7d062a67445c42865ce`
- 主实验是 NumPy/SciPy，**不要求 GPU**。不安装 vLLM、bitsandbytes、flash-attn、CUDA、NCCL。
- 真实数据阶段（v1，未在 v1.1 重做）仓库路径当时为 `/Users/grantwright/Research/idea-A-B`；Qwen 只保留已有 CPU 烟测。

## 3. Literature and Baseline Audit

v1.1 **没有**重跑 AMPLe / HyPER / Drago / PP-bandit / ncvxpp。状态从 `REPRODUCTION_AUDIT.json` 与 `COMPARABILITY_MATRIX.json` 原样读出。

### Idea A 官方基线

| paper | venue | local_status | upstream_commit | comparability (truncated) |
| --- | --- | --- | --- | --- |
| Comparison-based Active Preference Learning for Multi-dimensional Personalization (AMPLe) | ACL 2025 Main (Long) | partial | 0cad341f6a9e77063ef451c27732bf5c1534c797 | NOT DIRECTLY COMPARABLE to Idea A Neyman-audit IPW. AMPLe optimizes multi-dimensional preference identification via volu |
| Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback (HyPER) | ACL 2025 Main (Long) | blocked | 4d96b27b989828b0b69a430f68d25b6b36e9015a | NOT DIRECTLY COMPARABLE. HyPER routes human vs LM annotations to maximize downstream RM / RewardBench scores. |
| Online Compatible Reward Identification from Preference Feedback | ICML 2026 | blocked | None | NOT DIRECTLY COMPARABLE. Theory-only contrast: globally compatible reward identification vs Idea A finite-budget audit e |

### Idea B 官方基线

| paper | venue | local_status | upstream_commit | comparability (truncated) |
| --- | --- | --- | --- | --- |
| Performative Prediction with Bandit Feedback: Learning through Reparameterization | ICML 2024 | complete | d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5 | Official numerical experiment is a 1-D reparameterized zeroth-order performative-risk demo (Appendix H / Figures 4-6). C |
| Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss | NeurIPS 2024 | partial | ccbeeddfbf9237aa06dca3a1579529faabd8fe99 | NOT a full paper reproduction. Synthetic-iter SGD-GD is a performative linear-model experiment; not the same target as I |

Comparability 规则：Do not mix incomparable methods into one leaderboard. Status is Direct / partial / NOT DIRECTLY COMPARABLE. Numbers below are copied from run JSON; none are invented.

不得把 AMPLe L2、PP-bandit performative risk、ncvxpp test_acc 写进 Idea A/B 主表。协议不可比时标 `NOT DIRECTLY COMPARABLE`。

跳过项（来自 COMPARABILITY_MATRIX.json）：

- `pp_bandit_zeroth_order_reparam_adapted` status=`BLOCKED / adapted-skipped`: Official PP-bandit core (bandit-icml24.ipynb / official_reproduction/run_official_numerical.py) is a 1-D reparameterization x = y^2 with two-point zeroth-order updates and an inner LearnModel that minimizes Bernoulli KL(x_target || y^2). That procedure queries an uncensored scalar performative-risk function. Idea B environment B2 is a 3-D linear skill model whose performative channel is skill-wise observation censoring, not a Bernoulli mean map, and labels are revealed only under a limited audit budget. A faithful LearnModel+reparameterization port is not feasible on this NumPy protocol without rewriting the paper's 1-D Bernoulli experiment. A two-point ZO-SGD with uniform audits would be a different generic optimizer and must not be relabeled as adapted PP-bandit. Official numerical demo remains in idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/ (COMPLETE, separate from this leaderboard).
- `HyPER simplified implementation` status=`not done (pre-declared)`: No simplified HyPER was invented.
- `three final Chinese reports` status=`out of scope for this agent`: Parent instruction: do not generate the three final Chinese reports.

## 4. Idea A — Formal Problem and Method

问题：在内生反馈删失下，用独立审计通道识别有限总体泛函 `E[Y]`。已知审计倾向 `q` 才是 MNAR 下的识别通道。有审计不等于把自然通道 MNAR 变成 MAR。普通 DR 的双重稳健性只针对「审计缺失、`q` 已知、`A ⊥ Y | X, history`」。

v1.0 行为保持在 `idea_A/methods/v1.0/`。v1.1 在 `idea_A/methods/v1.1/`。

| ID | 分配 | 估计 | 角色 |
| --- | --- | --- | --- |
| A-v1.0 | v1.0 Neyman + exploration floor | IPW | 参照臂，行为与 v1.0 相同 |
| A-v1.1a | Uniform `q_i = b` | 2 折 cross-fit AIPW | **必须比较的基线** |
| A-v1.1b | 审计残差的 AIPW 方差近似，`E[q]=b` | 同上 | 经典分层方差分配，**不是新最优设计定理** |
| A-v1.1c | v1.1b + 剩余预算 + 策略漂移上权 | 同上 | **主提案** |

估计量：`μ_Y` 是审计样本上的 ridge；`μ_P` 是单独 logistic，禁止把连续 `μ_Y` 直接当概率；`μ_V = w(x) μ_Y(x)`。静态 2 折 cross-fit；序贯用 history-respecting nuisance。名义 95% 区间冻结为影响函数 Wald。

闭环环境在 `idea_A/benchmarks/Sequential_Closed_Loop/`。Static Estimation 与 Continual Adaptation 分目录、分指标。确认性场景：policy-drift 与 MNAR-audit。

主指标：总体均值 RMSE、bias、CI coverage、审计成本。主对照：Uniform audit + 2 折 cross-fit DR。

## 5. Idea A — Experiments

Tier 1：`n_seeds=3`，label=`PRELIMINARY`，`preliminary=true`。written=9。skipped=['b=0.01 and b=0.20 not in Tier 1 core; leftover-time only']。
Tier 2：`n_seeds=30`（协议主格子 30；便宜且区间贴 0 的静态 `b=0.05` 扩到 50），label=`confirmatory_cells`，written_count=`240`。
seed_extensions：`{"static_mnar_and_zero_overlap_b0.05": {"n_seeds": 50, "reason": "v1.1c vs Uniform DR RMSE CI covered 0 at n=30 and the cell is cheap; extended seeds 0..49 only on this static b=0.05 cell."}}`
skipped：['b=0.01 and b=0.20 not in confirmatory matrix', 'full Cartesian product not run', 'b=0.01 and b=0.20 leftover-time budgets not run', 'A ablation 30-seed expansion skipped: confirmatory v1.1c vs Uniform DR already decided on the main 30/50-seed cells']

### 5.1 Tier 1（PRELIMINARY）

静态 MNAR `b=0.05`：

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 3 | 0.054651 | -0.0193015 | 1 | 0.0561667 | true | PRELIMINARY |
| neyman_ipw | 3 | 0.126444 | 0.0116416 | 1 | 0.0505 | true | PRELIMINARY |
| observed_only | 3 | 1.1902 | 1.1897 | 0 | 0 | false | PRELIMINARY |
| uniform_dr | 3 | 0.0579175 | 0.0205919 | 1 | 0.0518333 | true | PRELIMINARY |
| uniform_ipw | 3 | 0.162359 | -0.145102 | 1 | 0.0511667 | true | PRELIMINARY |
| v1.1b | 3 | 0.0851586 | -0.0126568 | 1 | 0.0491667 | true | PRELIMINARY |
| v1.1c | 3 | 1.27613 | 0.617679 | 1 | 0.0541667 | true | PRELIMINARY |

静态 MNAR `b=0.05` 配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 3 | 0.054651 | 0.0579175 | -0.00326647 | -0.0415961 | 0.0447101 | covers_zero | INCONCLUSIVE |
| neyman_ipw | 3 | 0.126444 | 0.0579175 | 0.0685266 | -0.0291782 | 0.151714 | covers_zero | INCONCLUSIVE |
| observed_only | 3 | 1.1902 | 0.0579175 | 1.13228 | 1.0702 | 1.22769 | entirely_positive | NOT PROMISING |
| uniform_ipw | 3 | 0.162359 | 0.0579175 | 0.104442 | 0.0698129 | 0.150177 | entirely_positive | NOT PROMISING |
| v1.1b | 3 | 0.0851586 | 0.0579175 | 0.0272411 | -0.00318856 | 0.0832979 | covers_zero | INCONCLUSIVE |
| v1.1c | 3 | 1.27613 | 0.0579175 | 1.21822 | 0.140624 | 2.10011 | entirely_positive | NOT PROMISING |

短闭环 MNAR-audit `b=0.05`：

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 3 | 0.390783 | -0.254037 | 1 | 0.0510417 | true | PRELIMINARY |
| neyman_ipw | 3 | 0.455335 | -0.232703 | 1 | 0.0427083 | true | PRELIMINARY |
| observed_only | 3 | 1.26153 | 1.25994 | 0 | 0 | false | PRELIMINARY |
| uniform_dr | 3 | 0.169116 | -0.0631153 | 1 | 0.046875 | true | PRELIMINARY |
| uniform_ipw | 3 | 0.326555 | -0.278568 | 1 | 0.0479167 | true | PRELIMINARY |
| v1.1b | 3 | 0.186688 | -0.146438 | 1 | 0.05625 | true | PRELIMINARY |
| v1.1c | 3 | 0.420653 | -0.0974713 | 0.666667 | 0.053125 | true | PRELIMINARY |

### 5.2 Tier 2 静态（确认性）

#### mnar b=0.05

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 50 | 0.117947 | -0.0203015 | 0.9 | 0.05017 | true | confirmatory |
| observed_only | 50 | 1.21739 | 1.21647 | 0 | 0 | false | confirmatory |
| uniform_dr | 50 | 0.128779 | -0.00445322 | 0.96 | 0.05094 | true | confirmatory |
| uniform_ipw | 50 | 0.194143 | 0.0246254 | 0.94 | 0.04839 | true | confirmatory |
| v1.1b | 50 | 0.126844 | -0.0222439 | 0.9 | 0.0494 | true | confirmatory |
| v1.1c | 50 | 0.333698 | 0.0300304 | 0.96 | 0.04932 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 50 | 0.117947 | 0.128779 | -0.0108321 | -0.0463305 | 0.0227505 | covers_zero | INCONCLUSIVE |
| observed_only | 50 | 1.21739 | 0.128779 | 1.08861 | 1.05897 | 1.11803 | entirely_positive | NOT PROMISING |
| uniform_ipw | 50 | 0.194143 | 0.128779 | 0.0653642 | 0.0143589 | 0.117096 | entirely_positive | NOT PROMISING |
| v1.1b | 50 | 0.126844 | 0.128779 | -0.0019349 | -0.0378599 | 0.0331711 | covers_zero | INCONCLUSIVE |
| v1.1c | 50 | 0.333698 | 0.128779 | 0.204919 | -0.0384978 | 0.442027 | covers_zero | INCONCLUSIVE |

#### mnar b=0.10

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0553766 | 0.0168516 | 1 | 0.09775 | true | confirmatory |
| observed_only | 30 | 1.20706 | 1.2063 | 0 | 0 | false | confirmatory |
| uniform_dr | 30 | 0.0800039 | -0.0121183 | 0.933333 | 0.100733 | true | confirmatory |
| uniform_ipw | 30 | 0.138452 | -0.0482613 | 0.9 | 0.0988333 | true | confirmatory |
| v1.1b | 30 | 0.0698061 | -0.00981287 | 0.966667 | 0.10165 | true | confirmatory |
| v1.1c | 30 | 0.0816862 | 0.00281368 | 1 | 0.0997167 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0553766 | 0.0800039 | -0.0246273 | -0.0513369 | 0.00236522 | covers_zero | INCONCLUSIVE |
| observed_only | 30 | 1.20706 | 0.0800039 | 1.12705 | 1.09751 | 1.15735 | entirely_positive | NOT PROMISING |
| uniform_ipw | 30 | 0.138452 | 0.0800039 | 0.0584485 | 0.0179054 | 0.0980829 | entirely_positive | NOT PROMISING |
| v1.1b | 30 | 0.0698061 | 0.0800039 | -0.0101978 | -0.0445912 | 0.0242149 | covers_zero | INCONCLUSIVE |
| v1.1c | 30 | 0.0816862 | 0.0800039 | 0.00168232 | -0.0281207 | 0.0304037 | covers_zero | INCONCLUSIVE |

#### zero_overlap b=0.05

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 50 | 0.134685 | -0.00688221 | 0.92 | 0.04975 | true | confirmatory |
| observed_only | 50 | 0.376099 | -0.373143 | 0 | 0 | false | confirmatory |
| uniform_dr | 50 | 0.128779 | -0.00445322 | 0.96 | 0.05094 | true | confirmatory |
| uniform_ipw | 50 | 0.194143 | 0.0246254 | 0.94 | 0.04839 | true | confirmatory |
| v1.1b | 50 | 0.126844 | -0.0222439 | 0.9 | 0.0494 | true | confirmatory |
| v1.1c | 50 | 0.333698 | 0.0300304 | 0.96 | 0.04932 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 50 | 0.134685 | 0.128779 | 0.00590599 | -0.0329703 | 0.0447624 | covers_zero | INCONCLUSIVE |
| observed_only | 50 | 0.376099 | 0.128779 | 0.24732 | 0.21572 | 0.277873 | entirely_positive | NOT PROMISING |
| uniform_ipw | 50 | 0.194143 | 0.128779 | 0.0653642 | 0.0143589 | 0.117096 | entirely_positive | NOT PROMISING |
| v1.1b | 50 | 0.126844 | 0.128779 | -0.0019349 | -0.0378599 | 0.0331711 | covers_zero | INCONCLUSIVE |
| v1.1c | 50 | 0.333698 | 0.128779 | 0.204919 | -0.0384978 | 0.442027 | covers_zero | INCONCLUSIVE |

#### zero_overlap b=0.10

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0656238 | 0.0181008 | 0.966667 | 0.0980333 | true | confirmatory |
| observed_only | 30 | 0.369377 | -0.366543 | 0 | 0 | false | confirmatory |
| uniform_dr | 30 | 0.0800039 | -0.0121183 | 0.933333 | 0.100733 | true | confirmatory |
| uniform_ipw | 30 | 0.138452 | -0.0482613 | 0.9 | 0.0988333 | true | confirmatory |
| v1.1b | 30 | 0.0698061 | -0.00981287 | 0.966667 | 0.10165 | true | confirmatory |
| v1.1c | 30 | 0.0816862 | 0.00281368 | 1 | 0.0997167 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0656238 | 0.0800039 | -0.0143801 | -0.0446371 | 0.0163056 | covers_zero | INCONCLUSIVE |
| observed_only | 30 | 0.369377 | 0.0800039 | 0.289373 | 0.258075 | 0.317589 | entirely_positive | NOT PROMISING |
| uniform_ipw | 30 | 0.138452 | 0.0800039 | 0.0584485 | 0.0179054 | 0.0980829 | entirely_positive | NOT PROMISING |
| v1.1b | 30 | 0.0698061 | 0.0800039 | -0.0101978 | -0.0445912 | 0.0242149 | covers_zero | INCONCLUSIVE |
| v1.1c | 30 | 0.0816862 | 0.0800039 | 0.00168232 | -0.0281207 | 0.0304037 | covers_zero | INCONCLUSIVE |

### 5.3 Tier 2 闭环（确认性比较；10% 阈值为探索性）

#### mnar_audit b=0.05

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.149525 | -0.0352796 | 0.966667 | 0.0490417 | true | confirmatory |
| uniform_dr | 30 | 0.168591 | 0.0261678 | 0.966667 | 0.0482917 | true | confirmatory |
| v1.1c | 30 | 1.07494 | -0.147584 | 0.666667 | 0.0510625 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.149525 | 0.168591 | -0.0190661 | -0.0800774 | 0.0446245 | covers_zero | INCONCLUSIVE |
| v1.1c | 30 | 1.07494 | 0.168591 | 0.906352 | 0.0565902 | 1.67815 | entirely_positive | NOT PROMISING |

#### mnar_audit b=0.10

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0914274 | -0.0236759 | 0.966667 | 0.099375 | true | confirmatory |
| uniform_dr | 30 | 0.0713944 | 0.0197936 | 1 | 0.0981875 | true | confirmatory |
| v1.1c | 30 | 0.337072 | 0.0930421 | 0.8 | 0.10075 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0914274 | 0.0713944 | 0.020033 | -0.0115172 | 0.0526486 | covers_zero | INCONCLUSIVE |
| v1.1c | 30 | 0.337072 | 0.0713944 | 0.265678 | 0.0611594 | 0.463244 | entirely_positive | NOT PROMISING |

#### policy_drift b=0.05

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.140829 | -0.0389765 | 1 | 0.0486667 | true | confirmatory |
| uniform_dr | 30 | 0.168156 | 0.022166 | 0.966667 | 0.0482917 | true | confirmatory |
| v1.1c | 30 | 0.339184 | 0.076784 | 0.7 | 0.0518333 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.140829 | 0.168156 | -0.0273273 | -0.0771479 | 0.0270977 | covers_zero | INCONCLUSIVE |
| v1.1c | 30 | 0.339184 | 0.168156 | 0.171028 | 0.063879 | 0.293817 | entirely_positive | NOT PROMISING |

#### policy_drift b=0.10

| method | n_seeds | RMSE | bias | coverage | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0962209 | -0.0215604 | 0.966667 | 0.099625 | true | confirmatory |
| uniform_dr | 30 | 0.0728726 | 0.0152251 | 1 | 0.0981875 | true | confirmatory |
| v1.1c | 30 | 0.344648 | 0.104108 | 0.666667 | 0.100896 | true | confirmatory |

配对 vs Uniform DR：

| candidate | n | RMSE cand. | RMSE Uniform DR | diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| neyman_dr | 30 | 0.0962209 | 0.0728726 | 0.0233483 | -0.00365905 | 0.0510434 | covers_zero | INCONCLUSIVE |
| v1.1c | 30 | 0.344648 | 0.0728726 | 0.271776 | 0.0893354 | 0.45697 | entirely_positive | NOT PROMISING |

### 5.4 探索性 10% RMSE 下降

| cell | relative_rmse_change (v1.1c vs Uniform DR) | hit_10pct_drop |
| --- | --- | --- |
| policy_drift_b=0.05 | 1.01707 | false |
| policy_drift_b=0.10 | 3.72946 | false |
| mnar_audit_b=0.05 | 5.37603 | false |
| mnar_audit_b=0.10 | 3.72127 | false |

全部 `hit_10pct_drop=false`。探索性检查 **失败**。`exploratory_10pct_failed=true`。

### 5.5 边界预算

`b=0.01` 与 `b=0.20` 不在确认性矩阵。JSON skipped 写明 leftover-time 也未跑。不进主结论。

## 6. Idea A — Interpretation

**区分两件事。**

1. **identification-via-audit（v1.0 已支持，v1.1 静态确认性格子仍然支持）。** observed-only 在 MNAR 上有大约 +1.2 的 bias / RMSE；Uniform cross-fit DR 把 RMSE 降到约 0.13（n=50）。审计通道 + 正确 DR 是识别修复。
2. **sequential / adaptive allocation（v1.1 主提案，不被支持）。** 相对主对照 Uniform DR，v1.1c 在静态 MNAR `b=0.05` 的点估计更差且 CI 覆盖 0；在闭环 MNAR-audit `b=0.05` 的 CI 完全落在不利一侧。探索性 10% 下降失败。

A-v1.1b 相对 Uniform DR 的静态 MNAR `b=0.05` 差接近 0 且覆盖 0，漂移项没有额外价值。Uniform DR 在这些格子上主导。不得把赢了 observed-only 或 Uniform IPW 写成 PROMISING。

总体 Idea A v1.1：`NOT PROMISING`。不能标 PROMISING。

## 7. Idea B — Formal Problem and Method

问题：闭环策略使部分技能失去自然反馈，观测表现可以看起来很好，而弱技能误差与设计信息恶化。信息矩阵保留为线性模型观测设计 Gramian `W = Σ S φ φᵀ`，并分开已实现观测信息、条件期望信息、全体候选样本信息。更高特征值不是学习成功。

v1.0 行为保持。v1.1 修正 EqualSkill 为等期望审计条数 `q_i ∝ 1/n_skill`，另设频率比例分配。E-optimal（抬 `λ_min`）与 D-optimal（`logdet`）是**经典对照，不是新算法**。

| 名称 | 作用 | 新颖性 |
| --- | --- | --- |
| B-v1.0 | 原样包装 v1.0 ObservabilityController | 参照臂 |
| B-v1.1a / EqualSkill | 等技能期望条数 | 正确性修复 |
| B-v1.1b / E-optimal | 预算下提高条件 Gramian 的 `λ_min` | 经典对照 |
| D-optimal | 提高 `logdet` | 经典对照 |
| B-v1.1c | 信息折扣 + 一步预期降低 worst-skill MSE | **主提案** |
| Uniform / Greedy | 通用对照 | 无 |

主指标：late worst-skill MSE、累计 regret（相对噪声底 0.0225）、Env3 恢复时间。`1/(1+MSE)` 只作次要。主对照：E-optimal 与 D-optimal。赢 Uniform 不等于赢主对照。

Env1：真闭环，初值让多数技能已经不错。Env2：观测支撑上分不开的潜变量。Env3：回合中点变化，可恢复 / 不可恢复。

## 8. Idea B — Experiments

Tier 1：`n_seeds=3`，label=`PRELIMINARY`，n_runs=`84`，`preliminary=true`。
Tier 2：n_runs_attempted=`610`。noise_floor=`0.0225`。higher_eigenvalue_is_not_success=`true`。
seed_extensions：`{"Env2_d3_b0.05": {"n_seeds": 50, "reason": "v1.1c vs E-optimal cumulative regret CI covered 0 at n=30; extended seeds 0..49 on Env2 primary methods only."}}`
skipped：['b=0.01 and b=0.20 leftover-time only; not in main conclusion', 'b=0.01 and b=0.20 leftover-time budgets not run', 'B4/B5/B6 30-seed ablation expansion skipped: H_B1 already identified on the main Env1/Env2 matrix']
env3_horizon：`{"locked_horizon": 32, "oracle_check_tier1": "FullFeedbackOracle recovery was right-censored at horizon 32 and a diagnostic horizon 64. Horizon not lengthened because Oracle still did not recover. Not a Tier 2 result."}`

### 8.1 Tier 1（PRELIMINARY）

读取 `TIER1_SUMMARY.json` 的 `cells`。

#### Env1

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | 3 | 0.0297353 | 0.432863 | NA | 0 | 0.0518229 | true | PRELIMINARY |
| B-v1.1c | 3 | 0.0305456 | 0.126292 | NA | 0 | 0.0544271 | true | PRELIMINARY |
| D-optimal | 3 | 0.030712 | 0.256141 | NA | 0 | 0.05625 | true | PRELIMINARY |
| E-optimal | 3 | 0.0296546 | 0.202033 | NA | 0 | 0.0528646 | true | PRELIMINARY |
| EqualSkill | 3 | 0.0308912 | 0.366105 | NA | 0 | 0.0557292 | true | PRELIMINARY |
| Greedy | 3 | 0.17204 | 1.62134 | NA | 0 | 0.05 | true | PRELIMINARY |
| Uniform | 3 | 0.0339025 | 0.493668 | NA | 0 | 0.0549479 | true | PRELIMINARY |

#### Env2

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | 3 | 0.0295914 | 0.464969 | NA | 0 | 0.0526042 | true | PRELIMINARY |
| B-v1.1c | 3 | 0.0304876 | 0.228774 | NA | 0 | 0.0533854 | true | PRELIMINARY |
| D-optimal | 3 | 0.0301941 | 0.298267 | NA | 0 | 0.05625 | true | PRELIMINARY |
| E-optimal | 3 | 0.0296931 | 0.239158 | NA | 0 | 0.0526042 | true | PRELIMINARY |
| EqualSkill | 3 | 0.0307269 | 0.400909 | NA | 0 | 0.0557292 | true | PRELIMINARY |
| Greedy | 3 | 0.172636 | 1.67342 | NA | 0 | 0.05 | true | PRELIMINARY |
| Uniform | 3 | 0.0335419 | 0.520082 | NA | 0 | 0.0549479 | true | PRELIMINARY |

#### Env3_recoverable

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | 3 | 0.396448 | 2.2431 | 32 | 1 | 0.0518229 | true | PRELIMINARY |
| B-v1.1c | 3 | 0.119907 | 1.07804 | 32 | 1 | 0.0544271 | true | PRELIMINARY |
| D-optimal | 3 | 0.36307 | 1.88435 | 32 | 1 | 0.0552083 | true | PRELIMINARY |
| E-optimal | 3 | 0.50297 | 2.32453 | 32 | 1 | 0.053125 | true | PRELIMINARY |
| EqualSkill | 3 | 0.399897 | 2.1735 | 32 | 1 | 0.0557292 | true | PRELIMINARY |
| Greedy | 3 | 0.340638 | 3.11282 | 30 | 0.333333 | 0.05 | true | PRELIMINARY |
| Uniform | 3 | 0.514178 | 2.50402 | 32 | 1 | 0.0549479 | true | PRELIMINARY |

#### Env3_unrecoverable

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | 3 | 2.07836 | 5.81715 | 32 | 1 | 0.0322917 | false | PRELIMINARY |
| B-v1.1c | 3 | 2.07855 | 5.50412 | 32 | 1 | 0.0533854 | true | PRELIMINARY |
| D-optimal | 3 | 2.07859 | 5.63619 | 32 | 1 | 0.0565104 | true | PRELIMINARY |
| E-optimal | 3 | 2.07848 | 5.58422 | 32 | 1 | 0.0546875 | true | PRELIMINARY |
| EqualSkill | 3 | 2.07861 | 5.74599 | 32 | 1 | 0.0460937 | true | PRELIMINARY |
| Greedy | 3 | 2.07832 | 6.60185 | 23.3333 | 0.333333 | 0.05 | true | PRELIMINARY |
| Uniform | 3 | 2.07863 | 5.867 | 32 | 1 | 0.0520833 | true | PRELIMINARY |

### 8.2 Tier 2 主格子 Env1 / Env2（确认性 H_B1）

#### Env1

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | 30 | 0.0315497 | 0.361337 | NA | 0 | 0.05125 | true | confirmatory/robustness |
| B-v1.1c | 30 | 0.0322726 | 0.150165 | NA | 0 | 0.0518229 | true | confirmatory/robustness |
| D-optimal | 30 | 0.0320146 | 0.248098 | NA | 0 | 0.0519792 | true | confirmatory/robustness |
| E-optimal | 30 | 0.0315452 | 0.185975 | NA | 0 | 0.0517188 | true | confirmatory/robustness |
| EqualSkill | 30 | 0.0319097 | 0.233741 | NA | 0 | 0.0527083 | true | confirmatory/robustness |
| Greedy | 30 | 0.155466 | 1.83582 | NA | 0 | 0.05 | true | confirmatory/robustness |
| Uniform | 30 | 0.0335707 | 0.459492 | NA | 0 | 0.051849 | true | confirmatory/robustness |

配对 vs E-optimal：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | late_worst_skill_mse | 30 | 4.48534e-06 | -0.000163758 | 0.000152046 | covers_zero | INCONCLUSIVE |
| B-v1.0 | cumulative_regret | 30 | 0.175362 | 0.121729 | 0.233027 | entirely_positive | NOT PROMISING |
| B-v1.1c | late_worst_skill_mse | 30 | 0.000727427 | 6.74253e-05 | 0.0014246 | entirely_positive | NOT PROMISING |
| B-v1.1c | cumulative_regret | 30 | -0.0358101 | -0.0647617 | -0.00962598 | entirely_negative | IMPROVEMENT |
| EqualSkill | late_worst_skill_mse | 30 | 0.000364543 | -6.98224e-05 | 0.000870717 | covers_zero | INCONCLUSIVE |
| EqualSkill | cumulative_regret | 30 | 0.0477662 | 0.016789 | 0.0841286 | entirely_positive | NOT PROMISING |
| Greedy | late_worst_skill_mse | 30 | 0.12392 | 0.0913111 | 0.161172 | entirely_positive | NOT PROMISING |
| Greedy | cumulative_regret | 30 | 1.64985 | 1.44245 | 1.87455 | entirely_positive | NOT PROMISING |
| Uniform | late_worst_skill_mse | 30 | 0.00202556 | 0.00105404 | 0.00311085 | entirely_positive | NOT PROMISING |
| Uniform | cumulative_regret | 30 | 0.273517 | 0.166388 | 0.399981 | entirely_positive | NOT PROMISING |

配对 vs D-optimal：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | late_worst_skill_mse | 30 | -0.000464973 | -0.000864689 | -0.000129918 | entirely_negative | IMPROVEMENT |
| B-v1.0 | cumulative_regret | 30 | 0.113238 | 0.0611267 | 0.170057 | entirely_positive | NOT PROMISING |
| B-v1.1c | late_worst_skill_mse | 30 | 0.000257968 | -0.000612014 | 0.00111685 | covers_zero | INCONCLUSIVE |
| B-v1.1c | cumulative_regret | 30 | -0.0979338 | -0.13311 | -0.0652949 | entirely_negative | IMPROVEMENT |
| EqualSkill | late_worst_skill_mse | 30 | -0.000104915 | -0.000573472 | 0.000242522 | covers_zero | INCONCLUSIVE |
| EqualSkill | cumulative_regret | 30 | -0.0143576 | -0.0587428 | 0.0311512 | covers_zero | INCONCLUSIVE |
| Greedy | late_worst_skill_mse | 30 | 0.123451 | 0.0909262 | 0.160545 | entirely_positive | NOT PROMISING |
| Greedy | cumulative_regret | 30 | 1.58772 | 1.38591 | 1.80779 | entirely_positive | NOT PROMISING |
| Uniform | late_worst_skill_mse | 30 | 0.0015561 | 0.000793517 | 0.00244536 | entirely_positive | NOT PROMISING |
| Uniform | cumulative_regret | 30 | 0.211393 | 0.107426 | 0.340399 | entirely_positive | NOT PROMISING |

#### Env2

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | 50 | 0.0315686 | 0.434062 | NA | 0 | 0.0513594 | true | confirmatory/robustness |
| B-v1.1c | 50 | 0.0324485 | 0.219149 | NA | 0 | 0.052 | true | confirmatory/robustness |
| D-optimal | 50 | 0.0318781 | 0.302712 | NA | 0 | 0.05225 | true | confirmatory/robustness |
| E-optimal | 50 | 0.0315401 | 0.251316 | NA | 0 | 0.0522031 | true | confirmatory/robustness |
| EqualSkill | 50 | 0.0318986 | 0.325668 | NA | 0 | 0.0521562 | true | confirmatory/robustness |
| Greedy | 50 | 0.16165 | 2.17637 | NA | 0 | 0.05 | true | confirmatory/robustness |
| Uniform | 50 | 0.0341738 | 0.554846 | NA | 0 | 0.0511875 | true | confirmatory/robustness |

配对 vs E-optimal：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | late_worst_skill_mse | 50 | 2.84591e-05 | -0.000160801 | 0.000191294 | covers_zero | INCONCLUSIVE |
| B-v1.0 | cumulative_regret | 50 | 0.182746 | 0.143405 | 0.221865 | entirely_positive | NOT PROMISING |
| B-v1.1c | late_worst_skill_mse | 50 | 0.000908411 | 0.000365364 | 0.00154586 | entirely_positive | NOT PROMISING |
| B-v1.1c | cumulative_regret | 50 | -0.0321672 | -0.067478 | 0.00468866 | covers_zero | INCONCLUSIVE |
| EqualSkill | late_worst_skill_mse | 50 | 0.000358465 | 2.42027e-06 | 0.000782829 | entirely_positive | NOT PROMISING |
| EqualSkill | cumulative_regret | 50 | 0.0743521 | 0.0383984 | 0.111948 | entirely_positive | NOT PROMISING |
| Greedy | late_worst_skill_mse | 50 | 0.13011 | 0.105764 | 0.155839 | entirely_positive | NOT PROMISING |
| Greedy | cumulative_regret | 50 | 1.92505 | 1.65475 | 2.25233 | entirely_positive | NOT PROMISING |
| Uniform | late_worst_skill_mse | 50 | 0.00263373 | 0.00144329 | 0.00418899 | entirely_positive | NOT PROMISING |
| Uniform | cumulative_regret | 50 | 0.30353 | 0.211061 | 0.409898 | entirely_positive | NOT PROMISING |

配对 vs D-optimal：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | late_worst_skill_mse | 50 | -0.000309585 | -0.000664013 | 1.07048e-05 | covers_zero | INCONCLUSIVE |
| B-v1.0 | cumulative_regret | 50 | 0.13135 | 0.0940012 | 0.169657 | entirely_positive | NOT PROMISING |
| B-v1.1c | late_worst_skill_mse | 50 | 0.000570367 | -0.000258821 | 0.0015117 | covers_zero | INCONCLUSIVE |
| B-v1.1c | cumulative_regret | 50 | -0.0835625 | -0.123566 | -0.0442099 | entirely_negative | IMPROVEMENT |
| EqualSkill | late_worst_skill_mse | 50 | 2.04217e-05 | -0.000387158 | 0.000364717 | covers_zero | INCONCLUSIVE |
| EqualSkill | cumulative_regret | 50 | 0.0229568 | -0.0202971 | 0.0674608 | covers_zero | INCONCLUSIVE |
| Greedy | late_worst_skill_mse | 50 | 0.129772 | 0.105486 | 0.155511 | entirely_positive | NOT PROMISING |
| Greedy | cumulative_regret | 50 | 1.87366 | 1.59967 | 2.20729 | entirely_positive | NOT PROMISING |
| Uniform | late_worst_skill_mse | 50 | 0.00229569 | 0.00117928 | 0.00387261 | entirely_positive | NOT PROMISING |
| Uniform | cumulative_regret | 50 | 0.252135 | 0.164102 | 0.352885 | entirely_positive | NOT PROMISING |

### 8.3 稳健性（10 种子，不单独改写 H_B1）

#### Env1 `b=0.10`

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | 10 | 0.0320923 | 0.269342 | NA | 0 | 0.0999219 | true | confirmatory/robustness |
| B-v1.1c | 10 | 0.032305 | 0.116634 | NA | 0 | 0.0969531 | true | confirmatory/robustness |
| D-optimal | 10 | 0.0320132 | 0.190143 | NA | 0 | 0.0989844 | true | confirmatory/robustness |
| E-optimal | 10 | 0.03219 | 0.159429 | NA | 0 | 0.100078 | true | confirmatory/robustness |
| EqualSkill | 10 | 0.0322485 | 0.150397 | NA | 0 | 0.0992969 | true | confirmatory/robustness |
| Greedy | 10 | 0.155795 | 1.72075 | NA | 0 | 0.1 | true | confirmatory/robustness |
| Uniform | 10 | 0.0324145 | 0.268741 | NA | 0 | 0.0995313 | true | confirmatory/robustness |

配对 vs E-optimal：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.0 | late_worst_skill_mse | 10 | -9.77876e-05 | -0.000377408 | 0.000168571 | covers_zero | INCONCLUSIVE |
| B-v1.0 | cumulative_regret | 10 | 0.109913 | 0.0337952 | 0.200618 | entirely_positive | NOT PROMISING |
| B-v1.1c | late_worst_skill_mse | 10 | 0.000114936 | -0.000415602 | 0.000637311 | covers_zero | INCONCLUSIVE |
| B-v1.1c | cumulative_regret | 10 | -0.042795 | -0.0831316 | -0.00906299 | entirely_negative | IMPROVEMENT |
| EqualSkill | late_worst_skill_mse | 10 | 5.84346e-05 | -0.000269151 | 0.000419733 | covers_zero | INCONCLUSIVE |
| EqualSkill | cumulative_regret | 10 | -0.00903131 | -0.0467409 | 0.0201914 | covers_zero | INCONCLUSIVE |
| Greedy | late_worst_skill_mse | 10 | 0.123605 | 0.0693019 | 0.190283 | entirely_positive | NOT PROMISING |
| Greedy | cumulative_regret | 10 | 1.56132 | 1.31108 | 1.87766 | entirely_positive | NOT PROMISING |
| Uniform | late_worst_skill_mse | 10 | 0.00022443 | -0.00025605 | 0.000747034 | covers_zero | INCONCLUSIVE |
| Uniform | cumulative_regret | 10 | 0.109312 | 0.0373553 | 0.200115 | entirely_positive | NOT PROMISING |

#### Env1 d=2

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | 10 | 0.0279609 | 0.128092 | NA | 0 | 0.0501562 | true | confirmatory/robustness |
| E-optimal | 10 | 0.0272836 | 0.238804 | NA | 0 | 0.0507031 | true | confirmatory/robustness |
| Uniform | 10 | 0.0289707 | 0.595424 | NA | 0 | 0.0514844 | true | confirmatory/robustness |

配对 vs E-optimal：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | late_worst_skill_mse | 10 | 0.000677349 | -0.000130706 | 0.00174782 | covers_zero | INCONCLUSIVE |
| B-v1.1c | cumulative_regret | 10 | -0.110712 | -0.206543 | -0.0285325 | entirely_negative | IMPROVEMENT |
| Uniform | late_worst_skill_mse | 10 | 0.00168714 | 0.000249313 | 0.00355472 | entirely_positive | NOT PROMISING |
| Uniform | cumulative_regret | 10 | 0.356621 | 0.14058 | 0.618451 | entirely_positive | NOT PROMISING |

#### Env1 d=5

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | 10 | 0.0406641 | 0.349588 | NA | 0 | 0.0494531 | true | confirmatory/robustness |
| E-optimal | 10 | 0.0410445 | 0.426125 | NA | 0 | 0.0508594 | true | confirmatory/robustness |
| Uniform | 10 | 0.043761 | 0.754283 | NA | 0 | 0.0514844 | true | confirmatory/robustness |

配对 vs E-optimal：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | late_worst_skill_mse | 10 | -0.000380416 | -0.00104203 | 0.000299912 | covers_zero | INCONCLUSIVE |
| B-v1.1c | cumulative_regret | 10 | -0.0765372 | -0.187896 | -0.000665228 | entirely_negative | IMPROVEMENT |
| Uniform | late_worst_skill_mse | 10 | 0.0027165 | 0.000342384 | 0.00541858 | entirely_positive | NOT PROMISING |
| Uniform | cumulative_regret | 10 | 0.328158 | 0.163188 | 0.517339 | entirely_positive | NOT PROMISING |

### 8.4 Env3 恢复时间

#### Env3_recoverable

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | 10 | 0.0923312 | 0.964943 | 31.9 | 0.9 | 0.0489062 | true | confirmatory/robustness |
| E-optimal | 10 | 0.376524 | 1.99566 | 32 | 1 | 0.0486719 | true | confirmatory/robustness |
| Uniform | 10 | 0.297482 | 2.04685 | 32 | 1 | 0.0514844 | true | confirmatory/robustness |

配对 vs E-optimal（worst-skill / regret）：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | late_worst_skill_mse | 10 | -0.284193 | -0.377022 | -0.196942 | entirely_negative | IMPROVEMENT |
| B-v1.1c | cumulative_regret | 10 | -1.03072 | -1.21139 | -0.836298 | entirely_negative | IMPROVEMENT |
| Uniform | late_worst_skill_mse | 10 | -0.0790424 | -0.15165 | 0.000136502 | covers_zero | INCONCLUSIVE |
| Uniform | cumulative_regret | 10 | 0.051194 | -0.242354 | 0.336932 | covers_zero | INCONCLUSIVE |

recovery vs E-optimal：n=`10` mean_difference=`-0.1` CI [`-0.3`, `0`] side=`covers_zero`。
recovery vs Uniform：n=`10` mean_difference=`-0.1` CI [`-0.3`, `0`] side=`covers_zero`。

#### Env3_unrecoverable

| method | n_seeds | late worst-skill MSE | cumulative regret | recovery time | censored frac | realized audit | budget_band_ok | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | 10 | 3.54294 | 9.15381 | 32 | 1 | 0.0494531 | true | confirmatory/robustness |
| E-optimal | 10 | 3.54369 | 9.20937 | 32 | 1 | 0.049375 | true | confirmatory/robustness |
| Uniform | 10 | 3.54377 | 9.42634 | 32 | 1 | 0.0478906 | true | confirmatory/robustness |

配对 vs E-optimal（worst-skill / regret）：

| candidate | metric | n | mean diff (cand−base) | CI low | CI high | interval_side | protocol label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B-v1.1c | late_worst_skill_mse | 10 | -0.000752617 | -0.00194576 | 0.000224256 | covers_zero | INCONCLUSIVE |
| B-v1.1c | cumulative_regret | 10 | -0.0555548 | -0.0956223 | -0.0168711 | entirely_negative | IMPROVEMENT |
| Uniform | late_worst_skill_mse | 10 | 8.30228e-05 | -0.000135428 | 0.000342863 | covers_zero | INCONCLUSIVE |
| Uniform | cumulative_regret | 10 | 0.216971 | 0.102491 | 0.337501 | entirely_positive | NOT PROMISING |

recovery vs E-optimal：n=`10` mean_difference=`0` CI [`0`, `0`] side=`covers_zero`。
recovery vs Uniform：n=`10` mean_difference=`0` CI [`0`, `0`] side=`covers_zero`。

检测延迟：协议规定没有预先定义的检测器时写「未定义」。JSON 字段 `detection_delay` 在单次 run 的 metrics 里为 `undefined`。本摘要不把恢复时间叫做检测延迟。

Greedy 是通用对照，不是信息对照的胜利条件。H_B1 的对手是 E-optimal 与 D-optimal。

## 9. Idea B — Interpretation

B-v1.1c 在 Env1 上 **regret 优于 E-optimal**（配对区间完全为负），但 **worst-skill MSE 劣于 E-optimal**（配对区间完全为正）。H_B1 要求两者同时成立，并且还要对 D-optimal 成立。因此 H_B1 失败。

Env2 的 worst-skill 同样劣于 E-optimal（区间完全为正）；regret vs E 覆盖 0。E-optimal 赢了不算新方法成功。v1.1c 的信息折扣与一步收益没有在主指标上打败经典 E-optimal。

Env3 恢复时间 INCONCLUSIVE（右删失）。horizon 锁定为 `32`。JSON 说明：`FullFeedbackOracle recovery was right-censored at horizon 32 and a diagnostic horizon 64. Horizon not lengthened because Oracle still did not recover. Not a Tier 2 result.`

总体 Idea B v1.1：`NOT PROMISING`。不能标 PROMISING。

## 10. Negative Results and Failure Cases

- Idea A v1.1c 在确认性闭环格子上相对 Uniform DR **更差**（MNAR-audit 与 policy-drift，两个预算，区间完全为正）。
- Idea A 探索性 10% RMSE 下降四格全部失败。
- Idea A 静态 MNAR/zero-overlap `b=0.05` 的 v1.1c vs Uniform DR 覆盖 0，点估计更差。
- Idea B v1.1c 相对 E-optimal 的主指标 worst-skill MSE 在 Env1（n=30）与 Env2（n=50）上都更差。
- Env3 恢复时间右删失，比较 INCONCLUSIVE。
- 消融 30 种子扩展被 JSON skipped：主格子已经决定。
- `b=0.01` / `b=0.20` 未进主结论。
- 官方 HyPER / Drago 仍 blocked；AMPLe / ncvxpp 仍 partial；PP-bandit 适配到技能删失仍 blocked。
- 真实数据未写入合成主结论。RewardBench 不打榜。

## 11. Ablation Analysis

消融只在主格子、种子 0–2 上各改一个组件，**PRELIMINARY**，不能单独支撑 PROMISING。

### Idea A A1–A8（闭环 MNAR-audit `b=0.05`）

| ablation | component | method | n | RMSE (from per-seed errors) | bias mean | coverage | audit | label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1_No_Adaptive_Allocation | allocation | uniform_dr | 3 | 0.142978 | -0.112859 | 1 | 0.051875 | PRELIMINARY |
| A1_No_Adaptive_Allocation | allocation | v1.1c | 3 | 0.307251 | 0.0417341 | 0.333333 | 0.0491667 | PRELIMINARY |
| A2_No_Remaining_Budget | remaining_budget | v1.1b | 3 | 0.311766 | 0.144381 | 0.666667 | 0.04875 | PRELIMINARY |
| A2_No_Remaining_Budget | remaining_budget | v1.1c | 3 | 0.307251 | 0.0417341 | 0.333333 | 0.0491667 | PRELIMINARY |
| A3_No_Policy_Drift | policy_drift | v1.1c_no_drift | 3 | 0.296823 | 0.0988076 | 0.333333 | 0.0504167 | PRELIMINARY |
| A4_Logistic_Policy | policy_class | v1.1c_logistic_policy | 3 | 0.145153 | -0.000736388 | 1 | 0.0520833 | PRELIMINARY |
| A5_Misspecified_Propensity | propensity | v1.1c_wrong_q | 3 | 0.15857 | 0.0265586 | 1 | 0.04875 | PRELIMINARY |
| A6_IPW_Instead_Of_DR | estimator | v1.1c_ipw | 3 | 0.400116 | -0.141611 | 0.666667 | 0.0491667 | PRELIMINARY |
| A7_Strong_Ridge | outcome_model | v1.1c_strong_ridge | 3 | 0.243874 | 0.130348 | 1 | 0.0485417 | PRELIMINARY |
| A8_No_Policy_Update | policy_update | v1.1c_frozen_policy | 3 | 0.277307 | -0.0104922 | 0.666667 | 0.0477083 | PRELIMINARY |

来源：`idea_A/results/v1.1/ABLATION_INDEX.json` 与对应 `metrics.json`。RMSE 由每颗种子的 `population_mean_reward.bias` 重算 `sqrt(mean squared error)`。

### Idea B B1–B9（Env1 `b=0.05`）

| ablation | component | method | n | late worst-skill MSE mean | regret mean | audit | label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1_No_Observability_Control | allocation | B-v1.1c | 3 | 0.0305456 | 0.126292 | 0.0544271 | PRELIMINARY |
| B1_No_Observability_Control | allocation | Uniform | 3 | 0.0339025 | 0.493668 | 0.0549479 | PRELIMINARY |
| B2_EqualSkill_Instead | equal_skill | B-v1.1c | 3 | 0.0305456 | 0.126292 | 0.0544271 | PRELIMINARY |
| B2_EqualSkill_Instead | equal_skill | EqualSkill | 3 | 0.0308912 | 0.366105 | 0.0557292 | PRELIMINARY |
| B3_No_Information_Discount | information_discount | B-v1.1c_no_discount | 3 | 0.0296732 | 0.115505 | 0.0513021 | PRELIMINARY |
| B4_E_Optimal | e_optimal_control | B-v1.1c | 3 | 0.0305456 | 0.126292 | 0.0544271 | PRELIMINARY |
| B4_E_Optimal | e_optimal_control | E-optimal | 3 | 0.0296546 | 0.202033 | 0.0528646 | PRELIMINARY |
| B5_D_Optimal | d_optimal_control | B-v1.1c | 3 | 0.0305456 | 0.126292 | 0.0544271 | PRELIMINARY |
| B5_D_Optimal | d_optimal_control | D-optimal | 3 | 0.030712 | 0.256141 | 0.05625 | PRELIMINARY |
| B6_No_Endogenous_Feedback | endogenous_feedback | B-v1.1c_frozen_theta | 3 | 0.228727 | 1.06129 | 0.0541667 | PRELIMINARY |
| B7_Greedy | greedy | B-v1.1c | 3 | 0.0305456 | 0.126292 | 0.0544271 | PRELIMINARY |
| B7_Greedy | greedy | Greedy | 3 | 0.17204 | 1.62134 | 0.05 | PRELIMINARY |
| B8_V10_Controller | v1_0_controller | B-v1.0 | 3 | 0.0297353 | 0.432863 | 0.0518229 | PRELIMINARY |
| B8_V10_Controller | v1_0_controller | B-v1.1c | 3 | 0.0305456 | 0.126292 | 0.0544271 | PRELIMINARY |
| B9_Proportional_Frequency | proportional_frequency | B-v1.1c | 3 | 0.0305456 | 0.126292 | 0.0544271 | PRELIMINARY |
| B9_Proportional_Frequency | proportional_frequency | Proportional | 3 | 0.040903 | 1.01184 | 0.0528646 | PRELIMINARY |

B4/B5/B6 的 30 种子扩展被跳过：H_B1 已在主矩阵上判定。

## 12. Novelty Assessment

| 类别 | Idea A | Idea B |
| --- | --- | --- |
| 既有文献结果 | 分层 Neyman / AIPW 方差分配、cross-fit DR、独立审计识别 | 观测设计 Gram 矩阵、E-optimal、D-optimal、performative prediction |
| 新观察到的实验现象 | v1.1 确认性格子上 Uniform DR 修复 MNAR observed-only；自适应分配未赢主对照 | v1.1c 可降低 regret，但不能在 worst-skill MSE 上赢 E-optimal |
| 新算法机制 | **无。** A-v1.1b/c 是可计算的分层 AIPW 方差近似加启发式 | **无被主指标支持的新机制。** E/D-optimal 是经典对照 |
| 未证明的理论猜想 | 内生删失 + 独立审计的有限样本保证 | 策略诱发可观性坍缩的一般条件 |
| 与已发表论文的潜在重叠 | AMPLe/HyPER/Drago 目标不同，NOT DIRECTLY COMPARABLE | PP-bandit / ncvxpp 研究 performative risk，不是技能级删失 Gramian |

经验证据不等于已证明的新颖性。不得声称 outperform AMPLe / HyPER / PP-bandit / ncvxpp。

## 13. Limitations

- MacBook Air M3 / 16 GB / 无 NVIDIA GPU；主 venv 无 torch。
- 合成环境：线性奖励 / 线性技能、短 horizon、已知层或已知特征。
- `n=3` 消融与 Tier 1 是 PRELIMINARY；主结论用 30/50 种子。
- Env3 恢复时间大量右删失。
- 没有新的识别或最优设计定理。
- 真实数据（MultiPref 50-prompt、RewardBench schema、Qwen CPU generate）不进合成主结论。
- 官方基线不完整：HyPER blocked，Drago 无代码，AMPLe n_rounds=3，ncvxpp 非 10 replica。
- 每 run 的 Idea B PNG 体积大约 266MB，有意不进 git。

## 14. Final Recommendation

### Idea A：`NOT PROMISING`

跟 identification-via-audit，不要把 v1.1c 的序贯分配写成新算法成功。Uniform cross-fit DR 是当前最强的确认性对照。下一步若继续，应先解释闭环上 v1.1c 方差爆炸/覆盖下降的原因，而不是加大种子去追 10% RMSE。

### Idea B：`NOT PROMISING`

不要把 E-optimal 的胜利算成新方法。v1.1c 若只在 regret 上局部更好，不能通过 H_B1。下一步要么接受 E-optimal 作为充分对照并停止宣称控制器新颖性，要么换一个 **worst-skill** 上能先过 E-optimal 的机制；不要再把特征值当成功。

对 ICML / NeurIPS / ACL Main：当前 v1.1 **不足以**支持「已有可投稿的新分配/新控制器」。可以继续做识别现象或可观性诊断，但必须先过冻结的 PROMISING 门槛。

## 15. Reproduction Commands

在仓库根目录执行。主 Python：工作副本的 `.venv/bin/python`。需要 torch / transformers / datasets 的步骤用 `.isolated_baselines_venv`。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD"

# v1.1 Idea A
bash scripts/run_A_v1_1.sh

# v1.1 Idea B
bash scripts/run_B_v1_1.sh

# Tests (no GPU required)
.venv/bin/python -m pytest --import-mode=importlib

# Rebuild v1.1 Chinese reports from JSON
.venv/bin/python scripts/generate_v1_1_report.py

# v1 smoke reports (unchanged generator)
.venv/bin/python scripts/generate_report.py
```

v1 合成矩阵命令仍可用，见根 README。不要重跑官方 AMPLe/HyPER/ncvxpp 大实验。

## 16. Complete Result Index

### 摘要 JSON

- `idea_A/results/v1.1/TIER1_SUMMARY.json` (idea_A_tier1, exists=yes)
- `idea_A/results/v1.1/TIER2_SUMMARY.json` (idea_A_tier2, exists=yes)
- `idea_B/results/v1.1/TIER1_SUMMARY.json` (idea_B_tier1, exists=yes)
- `idea_B/results/v1.1/TIER2_SUMMARY.json` (idea_B_tier2, exists=yes)
- `idea_A/baselines/REPRODUCTION_AUDIT.json` (idea_A_audit, exists=yes)
- `idea_B/baselines/REPRODUCTION_AUDIT.json` (idea_B_audit, exists=yes)
- `reports/COMPARABILITY_MATRIX.json` (comparability, exists=yes)
- `idea_A/results/v1.1/ABLATION_INDEX.json` (idea_A_ablation_index, exists=yes)
- `idea_B/results/v1.1/ABLATION_INDEX.json` (idea_B_ablation_index, exists=yes)
- `idea_A/results/v1.1/BORDERLINE.json` (idea_A_borderline, exists=yes)
- `idea_B/results/v1.1/BORDERLINE.json` (idea_B_borderline, exists=yes)
- `idea_A/results/SMOKE_SUMMARY.json` (idea_A_smoke_v1, exists=yes)
- `idea_B/results/SMOKE_SUMMARY.json` (idea_B_smoke_v1, exists=yes)
- `reports/REAL_DATA_FEASIBILITY.json` (real_data, exists=yes)
- `reports/V1_1_EVALUATION_PROTOCOL.md` (protocol, exists=yes)
- `reports/V1_TO_V1_1_AUDIT.md` (audit_v1_to_v11, exists=yes)
- `idea_A/results/v1.1/t1_static_mnar_b0.05_s0/environment.json` (env_run_A, exists=yes)

### 运行目录计数（含 metrics.json 的子目录）

- Idea A v1.1：`313`
- Idea B v1.1：`885`
- 每个 run：`config.yaml`、`metrics.json`、`metrics.csv`、`run.log`、`environment.json`；部分含 `plots/`。

### 轻量汇总图（已跟踪；每 run PNG 未进 git）

- `idea_A/results/v1.1/plots/tier1_continual/audit_fraction_by_method.png`
- `idea_A/results/v1.1/plots/tier1_continual/coverage_by_method.png`
- `idea_A/results/v1.1/plots/tier1_continual/rmse_by_method.png`
- `idea_A/results/v1.1/plots/tier1_static/audit_fraction_by_method.png`
- `idea_A/results/v1.1/plots/tier1_static/coverage_by_method.png`
- `idea_A/results/v1.1/plots/tier1_static/rmse_by_method.png`
- `idea_A/results/v1.1/plots/tier2_continual_mnar_audit_b0.05/audit_fraction_by_method.png`
- `idea_A/results/v1.1/plots/tier2_continual_mnar_audit_b0.05/coverage_by_method.png`
- `idea_A/results/v1.1/plots/tier2_continual_mnar_audit_b0.05/rmse_by_method.png`
- `idea_A/results/v1.1/plots/tier2_static_mnar_b0.05/audit_fraction_by_method.png`
- `idea_A/results/v1.1/plots/tier2_static_mnar_b0.05/coverage_by_method.png`
- `idea_A/results/v1.1/plots/tier2_static_mnar_b0.05/rmse_by_method.png`
- `idea_A/results/v1.1/plots/tier2_static_zero_overlap_b0.05/audit_fraction_by_method.png`
- `idea_A/results/v1.1/plots/tier2_static_zero_overlap_b0.05/coverage_by_method.png`
- `idea_A/results/v1.1/plots/tier2_static_zero_overlap_b0.05/rmse_by_method.png`
- `idea_B/results/v1.1/plots/tier1_Env1/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier1_Env1/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier1_Env1/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier1_Env2/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier1_Env2/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier1_Env2/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier1_Env3_recoverable/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier1_Env3_recoverable/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier1_Env3_recoverable/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier1_Env3_unrecoverable/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier1_Env3_unrecoverable/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier1_Env3_unrecoverable/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier2_Env1_b0.05/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier2_Env1_b0.05/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier2_Env1_b0.05/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier2_Env1_b0.10/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier2_Env1_b0.10/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier2_Env1_b0.10/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier2_Env1_d2/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier2_Env1_d2/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier2_Env1_d2/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier2_Env1_d5/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier2_Env1_d5/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier2_Env1_d5/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier2_Env2_b0.05/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier2_Env2_b0.05/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier2_Env2_b0.05/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier2_Env3_recoverable/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier2_Env3_recoverable/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier2_Env3_recoverable/worst_skill_mse_vs_round.png`
- `idea_B/results/v1.1/plots/tier2_Env3_unrecoverable/cumulative_regret.png`
- `idea_B/results/v1.1/plots/tier2_Env3_unrecoverable/worst_skill_mse.png`
- `idea_B/results/v1.1/plots/tier2_Env3_unrecoverable/worst_skill_mse_vs_round.png`

### Git

- branch `v1.1` commit `7de4bd5d1545f355a87444fea9fae42a9bf4a3cb` remote `git@github.com:SelinaMai-source/idea-A-B.git`

---

数字来源全部是磁盘 JSON。若某单元格在 JSON 中不存在，表中写 NA，而不是补造。本报告不覆盖 `V1_TO_V1_1_AUDIT.md` 与 `V1_1_EVALUATION_PROTOCOL.md`。
