# v1.2 已有证据审计

日期：2026-10-31。工作副本 `/Users/grantwright/Desktop/科研/10.30新课题/v0`。分支 `v1.2` 从 `v1.1` 的 `08b3772a55e1d2dfb89716c8010af5d062b54bbf`（跟踪 `origin/v1.1`）新建。本文件只登记已经落地的算法、仓库、数据和可复用数字。没有重跑 AMPLe、PP-bandit notebook、ncvxpp，也没有新的 Idea A/B 对照实验。

数字来自已提交的 JSON 与 v1.1 报告，不来自本阶段的新运行：

- `idea_A/baselines/REPRODUCTION_AUDIT.json`
- `idea_B/baselines/REPRODUCTION_AUDIT.json`
- `idea_A/results/v1.1/TIER2_SUMMARY.json`
- `idea_B/results/v1.1/TIER2_SUMMARY.json`
- `reports/V1_1_FULL_EXPERIMENT_REPORT.md`
- `reports/V1_1_BASELINE_REPRODUCTION.md`

## 1. v1.1 已经实现的算法

这些是本仓库自己的合成协议，不是已发表 CCF-A 论文的官方脚本。

### Idea A

估计量在审计样本上用 ridge 拟合 `μ_Y`，用单独 logistic 拟合 `μ_P`，`μ_V = w(x) μ_Y(x)`。静态设定用 2 折 cross-fit；序贯设定用 history-respecting nuisance。名义 95% 区间是影响函数 Wald。主指标是总体均值 RMSE、bias、CI coverage、审计成本。主对照是 Uniform audit + 2 折 cross-fit DR。

| 方法 | 分配 | 角色 |
| --- | --- | --- |
| A-v1.1a | Uniform `q_i = b` | 必须比较的基线 |
| A-v1.1b | 审计残差的 AIPW 方差近似，`E[q]=b` | 经典分层方差分配 |
| A-v1.1c | A-v1.1b + 剩余预算 + 策略漂移上权 | v1.1 主提案 |
| observed-only / Uniform IPW | 合成协议里的弱对照 | 赢它们不能支撑 PROMISING |

`idea_A/baselines/` 里的统计适配器（`Uniform_Audit`、`IPS`、`SNIPS`、`Doubly_Robust`）和 `idea_A/experiments/protocol.py` 里的 Observed-only、Fixed Stratified Audit、Uncertainty Sampling、Full Feedback Oracle、naive IPS，都是本仓库实现。Uncertainty Sampling 是通用不确定度规则。

### Idea B

v1.0 行为保留。v1.1 把 EqualSkill 改成等期望审计条数 `q_i ∝ 1/n_skill`，并另设频率比例分配。主指标是 late worst-skill MSE、相对噪声底 0.0225 的累计 regret、Env3 恢复时间。主对照是 E-optimal 与 D-optimal。`1/(1+MSE)` 只作次要指标。

| 方法 | 分配 | 角色 |
| --- | --- | --- |
| B-v1.1a / EqualSkill | 等技能期望条数 | 正确性修复 |
| B-v1.1b / E-optimal | 预算下提高条件 Gramian 的 `λ_min` | 经典对照 |
| D-optimal | `logdet` | 经典对照 |
| B-v1.1c | 信息折扣 + 一步预期降低 worst-skill MSE | v1.1 主提案 |
| Uniform / Greedy / Proportional | 通用对照 | 无论文归属 |

`idea_B/baselines/` 里的 Uniform、Greedy、Lazy、Standard Online SGD、Fixed Exploration Schedule、Uncertainty-based Exploration、No Feedback、Full Feedback Oracle 都是 generic controls。Uncertainty-based Exploration 按 `φᵀ(W+λI)⁻¹φ` 分配，目录说明写明它是常见不确定度采样。

## 2. 已下载的官方仓库与数据

| 对象 | 本地路径 | commit | 状态 |
| --- | --- | --- | --- |
| AMPLe | `idea_A/baselines/AMPLe/` | `0cad341f6a9e77063ef451c27732bf5c1534c797` | partial，已有 `official_reproduction/` |
| HyPER | `idea_A/baselines/HyPER/` | `4d96b27b989828b0b69a430f68d25b6b36e9015a` | blocked；submodule 未初始化；浅克隆约 632K |
| Drago ICML 2026 | `idea_A/baselines/Online_Compatible_Reward_ICML2026/THEORY_COMPARATOR.md` | 无 | blocked，无官方代码 |
| PP-bandit | `idea_B/baselines/Performative_Bandit_ICML2024/` | `d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5` | 数值脚本已跑完，见第 4 节 |
| ncvxpp | `idea_B/baselines/Performative_SGD_NeurIPS2024/` | `ccbeeddfbf9237aa06dca3a1579529faabd8fe99` | 单进程 partial |
| ActiveUltraFeedback | `idea_A/baselines/ActiveUltraFeedback/` | `3cf700a85bbe344b1b472890a488671e2916a8b4` | 本阶段新浅克隆；submodule 未初始化；约 1.9M。选择函数导入 blocked，见 `02_ORIGINAL_REPRODUCTION.md` |

AMPLe 部分运行使用的公开数据集是 `minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271`。README 的 `--dims 0 1` 会改写成未公开的 `minhyeonoh/feedback_assistant_harmlesshelpful_271`（当时 HTTP 401），所以那次运行省略了 `--dims`。HyPER 的 `open-instruct` 与 `reward-bench` 只有空占位。ActiveUltraFeedback 的 `.gitmodules` 列出 `resources/reward-bench`、`resources/olmes`、`resources/alpaca_eval`，工作区里没有 `resources/` 目录。

Hugging Face `ActiveUltraFeedback/{combined,ultrafeedback,tulu3,skywork}` 的数据卡已读，parquet 未下载。标注来源与覆盖写在 `01_PUBLISHED_BASELINE_REGISTRY.md`。这些分数是 LLM judge / reward model 分数。

## 3. 只是启发式、不能改名的对照

| 本地对象 | 它实际是什么 | 不能改称 |
| --- | --- | --- |
| `volume_adapted_ipw` | 同一合成世界上的分配启发式；官方 volume 打分的是 RM 标注偏好对上的单纯形 MH 后验 | AMPLe |
| Idea A Uncertainty Sampling | 协议里的通用不确定度规则 | HyPER、DeltaUCB、InfoMax、DRTS |
| Idea B Uncertainty-based Exploration | leverage 分配 | HyPER、UCB-OGD |
| Greedy / Lazy / Standard Online SGD | Idea B 合成协议上的通用优化对照 | ncvxpp 的 SGD-GD，或 PP-bandit 的两层零阶法 |
| `pp_bandit_zeroth_order_reparam_adapted` | v1 审计结论是：官方核心查询的是无删失标量 performative risk；B2 是三维技能删失。忠实移植当时判定不可行，也没有另写一个两层零阶法再贴上论文名字 | 官方 PP-bandit |

## 4. 可复用的原设定数字

本阶段没有重跑这些实验。

### AMPLe partial

来源：`idea_A/baselines/AMPLe/official_reproduction/metrics.json`，并写入 `REPRODUCTION_AUDIT.json`。

- 命令保持 `acquisition=volume`、`burnin=50000`，只把 README 的 `n_rounds=100` 缩短为 `n_rounds=3`。
- 数据集：`minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271`。
- `true_preference = [0.2, 0.7, 0.1]`。
- `estimation_t0 = [0.3249063193798065, 0.32413744926452637, 0.3509562909603119]`，`l2_to_true_t0 = 0.4688851731916284`。
- `estimation_t3 = [0.0035132216289639473, 0.9920580983161926, 0.004428689833730459]`，`l2_to_true_t3 = 0.3647449275776112`。
- `chosen_indices = [737, 95, 135]`，`wall_sec_approx = 97.8`。
- 隔离环境安装了 torch / datasets / wandb / accelerate。主项目 venv 当时记录为没有 torch。`WANDB_MODE=offline`。

这个 L2 是三维个性化偏好向量到 `true_preference` 的距离。Idea A 的总体泛函 RMSE 与它 **INCOMPARABLE**。两边的数字不相减。

### PP-bandit 数值运行

来源：`idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/metrics.json`。算法单元来自 `bandit-icml24.ipynb`，本地脚本加了 `numpy` seed 0；notebook 本身没有 seed。

- 单次，`num_iterations=500`，`x0=0.3`：`final_PR_phi = -0.28164988861568363`，`final_PR_theta = -0.2962457875065718`。
- 10 个初值、每个 100 步：`PR_phi_final_mean = -0.2883591540722531`（std `0.0071395956688855965`），`PR_theta_final_mean = -0.291834361799254`（std `0.007228858237631779`）。
- 墙钟 `elapsed_sec = 0.814453125`。依赖是主 venv 的 numpy / scipy / matplotlib。

环境是一维 Bernoulli performative risk，`LearnModel` 用 Bernoulli KL 把 `φ(θ)` 配到目标分布参数。notebook 源码里没有 skill group。B-v1.1 的技能分配在这个原环境上 **INCOMPARABLE**。论文 Appendix H 的图 4–7 没有可抄的端点表，因此这里不把这次运行写成“对上了图 4–6”。对照写在 `02_ORIGINAL_REPRODUCTION.md`。

### ncvxpp 单进程 partial

来源：`idea_B/baselines/REPRODUCTION_AUDIT.json`。官方全文命令 `cd synthetic-iter && mpiexec -np 10 python main.py` 仍 blocked：没有 `mpi4py`，PATH 上没有 `mpicc` / `mpiexec`。已跑的是 `rank=0, size=1` 的原 per-process 循环，CPU，`eps ∈ {0, 0.1, 10, 100}`，`batch=5`，`log_maxiter=5`，墙钟 540.486 秒。

| npy | iter_last | loss_last | train_acc_last | test_acc_last |
| --- | --- | --- | --- | --- |
| `eps0-batch5-res0.npy` | 99426 | 0.3281349241733551 | 0.9004424778761062 | 1.0 |
| `eps0.1-batch5-res0.npy` | 99426 | 0.3458000421524048 | 0.9004424778761062 | 1.0 |
| `eps10-batch5-res0.npy` | 99426 | 0.3553709387779236 | 0.9004424778761062 | 1.0 |
| `eps100-batch5-res0.npy` | 99426 | 0.9450080394744873 | 0.09955752212389381 | 0.0 |

`dataset.py` 按 `Y==1` 与 `Y==0` 平衡，而 `data.csv` 的标签是 `{+1,-1}`，所以只有 `+1` 类留下来。这是官方代码和数据的交互，审计当时没有改写。单进程结果不是 `mpiexec -np 10` 的全文复现。上游 README 把 `synthetic-iter` 标成线性模型实验，把 `spam-iter` 标成神经网络实验；`spam-iter` 本阶段没有跑。

### HyPER 与 Drago

HyPER `4d96b27`：blocked。官方流程要 Google Cloud TPU v3-128、EasyLM、Tulu-2 13B / Llama-3.1-8B、vLLM、Beaker、GCS，以及未初始化的 submodule。没有简化版 HyPER。

Drago 等，*Online Compatible Reward Identification from Preference Feedback*，ICML 2026，PMLR `drago26a`：blocked。当时没有找到官方代码，也没有自写实现。

## 5. v1.1 主结论（沿用，不改口）

判定预先冻结在 `reports/V1_1_FULL_EXPERIMENT_REPORT.md`。PROMISING 不能只表示打败均匀随机。`n_seeds<=3` 标 PRELIMINARY。

**Idea A v1.1：`NOT PROMISING`。** A-v1.1c 没有打败 Uniform cross-fit DR。

- 静态 MNAR `b=0.05`，n=`50`：v1.1c RMSE `0.333698`，Uniform DR RMSE `0.128779`，diff `0.204919`，CI `[-0.0384978, 0.442027]`，`covers_zero`，协议标签 `INCONCLUSIVE`。
- 闭环 MNAR-audit `b=0.05`，n=`30`：v1.1c RMSE `1.07494`，Uniform DR RMSE `0.168591`，diff `0.906352`，CI `[0.0565902, 1.67815]`，`entirely_positive`，协议标签 `NOT PROMISING`。
- 四条闭环格子的探索性 10% RMSE 下降都失败（`exploratory_10pct_failed=true`）。
- identification-via-audit 在静态确认性格子上仍然成立：同格 observed-only RMSE `1.21739`（n=`50`），Uniform DR RMSE `0.128779`。这是审计通道，不是自适应分配的胜利。

**Idea B v1.1：`NOT PROMISING`。** B-v1.1c 的 worst-skill MSE 没有打败 E-optimal。H_B1 要求 worst-skill MSE 和 cumulative regret 相对 E 与 D 的配对区间同时完全为负。

- Env1，n=`30`：v1.1c late worst-skill MSE `0.0322726`，E-optimal `0.0315452`，diff `0.000727427`，CI `[6.74253e-05, 0.0014246]`，`entirely_positive`（v1.1c 更差）。
- 同格 regret diff `-0.0358101`，CI `[-0.0647617, -0.00962598]`，`entirely_negative`。regret 这一项更好，H_B1 仍不满足。
- Env2，n=`50`：worst-skill diff `0.000908411`，CI `[0.000365364, 0.00154586]`，`entirely_positive`。
- Env3 恢复时间 `INCONCLUSIVE`：可恢复格 v1.1c censored fraction `0.9`，E-optimal `1`；配对 recovery CI `[-0.3, 0]`，`covers_zero`。

## 6. 本阶段预先标成 INCOMPARABLE 的对比

这些标签在后续适配器运行之前就锁定。本文件没有新证据去推翻它们。

| 对比 | 标签 | 原因 |
| --- | --- | --- |
| AMPLe 个性化偏好向量的 L2 vs Idea A 总体泛函 RMSE | INCOMPARABLE | 目标、预算、信息都不同。AMPLe 保留原协议上的估计误差和查询轮数 |
| HyPER 的 human/LM 路由与 RewardBench vs Idea A 审计泛函 | INCOMPARABLE | 本机 blocked。不确定度规则保持原名 |
| Drago 的全局兼容奖励识别 vs Idea A 有限预算审计 | INCOMPARABLE | 无官方代码，只有理论对照笔记 |
| PP-bandit 一维 performative risk vs B-v1.1 技能分配 | INCOMPARABLE | 原 notebook 没有技能组，也没有审计动作 |
| ncvxpp 的 performative 线性 SGD vs Idea B observability Gramian | INCOMPARABLE | 单进程 partial 也不是同一目标。SGD 不替换 PP-bandit 的两层零阶法 |
| ActiveUltraFeedback 的 RM/DPO/judge 偏好对 vs Idea A 总体 RMSE | INCOMPARABLE | 全文响应生成、RM 训练、DPO/CPO 在本机 blocked。选择函数本阶段未能导入，没有形成 DIRECT/ADAPTED 格子 |
| UCB-OGD 的 CIFAR-10 / VOC2012 / VQAv2 / testbed 指标 vs Idea A 审计 RMSE 或 Idea B worst-skill MSE | INCOMPARABLE | 无官方代码；`independent_paper_based` 本阶段 not done。图像张数、QA 对、类别准确率、mAP 与审计次数不是同一预算单位 |
| A-v1.1c 或 B-v1.1c 相对 Uniform / Greedy / 不确定度启发式的差距 | 不能当成发表基线胜利 | A 的主对照是 Uniform cross-fit DR；B 的主对照是 E-optimal 与 D-optimal。两边主指标都已是 NOT PROMISING |

预算单位继续分开记：标注次数、偏好对、标注回复、prompt 数、人类成本、judge 成本。论文表里的准确率或 mAP 不放进 Idea A/B 的 RMSE 或 worst-skill 列。

## 7. 本阶段明确没有做的事

没有改 v1.1 算法，没有写新的分配器，没有实现 head-to-head 适配器，没有跑 3-seed smoke 或 10/30-seed 主格子，没有写 `03_COMPARABILITY_AUDIT.md`，没有推送。未纳入本次提交的脏文件仍留在工作区：`reports/A_B_SMOKE_EXPERIMENT_REPORT.md`、`reports/environment_manifest.json`、`shared/hardware/environment_manifest.json`，以及未跟踪的 `idea_B/results/v1.1/**/plots/`。
