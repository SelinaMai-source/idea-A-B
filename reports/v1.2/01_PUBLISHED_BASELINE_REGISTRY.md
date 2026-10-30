# v1.2 已发表基线登记

日期：2026-10-31。分支 `v1.2`，基线提交 `08b3772a55e1d2dfb89716c8010af5d062b54bbf`。本表只登记论文、仓库和可比性。适配状态在没有官方选择函数导入成功、也没有 DIRECT/ADAPTED 协议之前，一律还不是“已对照”。

预算单位分列，不合并成一个 “budget”：

- 标注次数：人类或审计通道实际揭开的标签条数
- 偏好对：一次比较里被选中的一对回复
- 标注回复：被生成或被打分的回复条数
- prompt 数
- 人类成本
- judge 成本（LLM-as-a-Judge 或奖励模型打分）

硬件以论文或官方 README 的自述为准。本机是 MacBook Air M3 16GB，arm64，主 venv `.venv` 的 Python 3.12.4、numpy 2.5.4，没有 torch。本阶段没有往主 venv 安装 torch、CUDA 或 vLLM。

## 总表

| ID | 论文 | Venue | 算法 | 仓库 / commit | 复现状态 | 与 Idea A/B 主表 |
| --- | --- | --- | --- | --- | --- | --- |
| AMPLe | Comparison-based Active Preference Learning for Multi-dimensional Personalization | ACL 2025 Main | volume / random acquisition，单纯形偏好向量 | https://github.com/ml-postech/AMPLe `0cad341` | partial | INCOMPARABLE |
| HyPER | Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback | ACL 2025 Main | human vs LM 路由 | https://github.com/allenai/hybrid-preferences `4d96b27` | blocked | INCOMPARABLE |
| Drago | Online Compatible Reward Identification from Preference Feedback | ICML 2026 | 理论：全局兼容奖励识别 | 无官方仓库 | blocked | INCOMPARABLE |
| ActiveUF | ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning | ICML 2026 Main | DRTS、DeltaUCB；对照含 InfoMax、DTS、MaxMinLCB | https://github.com/lasgroup/ActiveUltraFeedback `3cf700a` | 选择函数导入 blocked；全文 blocked | INCOMPARABLE（本阶段） |
| UCB-OGD | Adaptive Data Collection for Robust Learning Across Multiple Distributions | ICML 2025 Main | UCB + OGD | 未找到官方仓库 | `independent_paper_based = not done`；CIFAR/VOC/VQA blocked | INCOMPARABLE |
| PP-bandit | Performative Prediction with Bandit Feedback: Learning through Reparameterization | ICML 2024 | 两层零阶重参数化 | https://github.com/UCSC-REAL/PP-bandit-feedback `d5daeff` | 数值脚本已跑完；未宣称对齐图 4–6 | 技能分配 INCOMPARABLE |
| ncvxpp | Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss | NeurIPS 2024 | performative SGD / SGD-GD | https://github.com/Qiang-CU/nips2024-ncvxpp `ccbeedd` | 单进程 partial；`mpiexec -np 10` blocked | INCOMPARABLE |

## AMPLe

- 题目：Comparison-based Active Preference Learning for Multi-dimensional Personalization
- Venue：ACL 2025 Main (Long)
- 算法：在 RM 打分的回复对上，用 volume 或 random acquisition 识别单纯形上的多维个性化偏好。本地 partial 运行保持 `acquisition=volume`
- 仓库：https://github.com/ml-postech/AMPLe
- Commit：`0cad341f6a9e77063ef451c27732bf5c1534c797`（2025-12-01，Add README.md）
- 原任务：给定 `true_preference`，用比较查询估计个性化偏好向量
- 原指标：偏好向量估计误差。本地保存的是到 `true_preference` 的 L2，t0 `0.4688851731916284`，t3 `0.3647449275776112`
- 预算单位：查询轮数 `n_rounds`（README 示例 100；本地 3）。burn-in 50000 是初始化采样，单独记，不并进 Idea A 的审计比例 `b`。没有人类成本账本。RM 分数来自已发布数据，不是本机新产生的 judge 成本
- 硬件：本地 partial 用隔离 venv 的 CPU torch，约 97.8 秒。全文 100 轮未跑
- 复现状态：`partial`
- 适配：无。`volume_adapted_ipw` 是合成协议启发式，不进入本行
- 可比性：个性化向量 L2 与 Idea A 总体泛函 RMSE **INCOMPARABLE**

## HyPER

- 题目：Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback
- Venue：ACL 2025 Main (Long)
- 算法：把实例路由到人类或 LM 标注，以下游 RM / RewardBench 为目标
- 仓库：https://github.com/allenai/hybrid-preferences
- Commit：`4d96b27b989828b0b69a430f68d25b6b36e9015a`（2025-07-22）
- 原任务：混合人类与 AI 反馈，训练奖励模型
- 原指标：下游 RM 与 RewardBench。本地没有数字
- 预算单位：人类标注条数与 LM 标注条数分开；RewardBench 题数是评测规模，不是 Idea A 的审计次数
- 硬件：官方流程需要 TPU v3-128、EasyLM、Tulu-2 13B / Llama-3.1-8B、vLLM、Beaker、GCS。submodule `open-instruct`、`reward-bench` 未初始化
- 复现状态：`blocked`
- 适配：无。没有简化 HyPER，也没有把不确定度采样改叫 HyPER
- 可比性：**INCOMPARABLE**

## Drago 等，ICML 2026

- 题目：Online Compatible Reward Identification from Preference Feedback
- Venue：ICML 2026。PMLR https://proceedings.mlr.press/v306/drago26a.html
- 算法：全局兼容奖励识别（理论）
- 仓库：无。v1 审计检查了 PMLR / poster / 作者页面，没有 GitHub。`github.com/marcomussi/SleepingRL` 是另一篇 ICML 2025，不记入本行
- Commit：无
- 原任务 / 原指标：理论识别，不是有限预算审计 RMSE
- 预算单位：论文的查询复杂度，不是本地审计比例 `b`
- 硬件：无本地运行
- 复现状态：`blocked`。没有自写作者实现
- 适配：无
- 可比性：**INCOMPARABLE**

## ActiveUltraFeedback

- 题目：ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning
- Venue：ICML 2026 Main。arXiv:2603.09692。README 写明 2026-04-30 接收
- 作者：Davit Melikidze, Marian Schneider, Jessica Lam, Martin Wertich, Ido Hakimi, Barna Pásztor, Andreas Krause
- 算法：流水线是生成回复、不确定度奖励模型、选一对、oracle 标注、再训练 RM。论文新选择器是 DRTS（源码类名 `DoubleReverseThompsonSampling`）和 `DeltaUCB`。`InfoMax` 在同一仓库里，README 把它标成已有 dueling-bandit 方法（Saha, 2021），不是这篇的新算法。同目录还有 DTS、MaxMinLCB、Random、UltraFeedback
- 仓库：https://github.com/lasgroup/ActiveUltraFeedback
- Commit：`3cf700a85bbe344b1b472890a488671e2916a8b4`（2026-05-02，Merge pull request #3 from lasgroup/readme_update）。浅克隆，`--recurse-submodules=no`。`.gitmodules` 中的 reward-bench、olmes、alpaca_eval 未检出
- 原任务：从 prompt 池生成偏好数据，再用 RM 或 DPO/CPO 做下游评测
- 原指标：仓库 `scripts/statistics/generate_tables_from_dir.py` 里有 GSM8K、IFEval、TruthfulQA、AlpacaEval 以及 RewardBench 风格切片（Factuality、Focus、Math、Precise IF、Safety、Ties）。这些是脚本里的表，不是本机实测。本阶段不把它们抄进 Idea A 主表
- 预算单位：
  - prompt / 样本：数据卡自称 UltraFeedback 60k、Skywork 80k、combined 140k、Tulu 3 272k。本阶段没有数 parquet 行，这些是卡片原文
  - 标注回复：`full` 配置写明每个 prompt 30 条生成回复
  - 偏好对：10 个成对配置每个 prompt 只留 chosen/rejected 两条，也就是一对
  - 人类成本：数据卡没有把这些偏好写成人类标注
  - judge 成本：标注者是 LLM-as-a-Judge。`chosen_score` / `rejected_score` 的卡片说明是 reward model score。`full.completions` 里还有按原则给出的 LLM judge 评价
- 硬件：README 的 Docker 面向 NVIDIA GH200（aarch64）CUDA 13.1；本地安装在 RTX 5090、CUDA 13.0 上测过，要求 torch 2.10.0+cu130 与 vLLM 0.19.0。`configs/loop.yaml` 的奖励模型是 `Skywork/Skywork-Reward-V2-Qwen3-4B`（ENN）。示例 oracle 使用 `Qwen/Qwen3-32B`
- 复现状态：全文响应生成、RM 训练、DPO/CPO **blocked**。DRTS / DeltaUCB / InfoMax 在主 venv 导入失败，精确错误见 `02_ORIGINAL_REPRODUCTION.md`。没有 `selector_unit_check`
- 适配：无。导入失败，所以没有调用官方选择函数的适配器
- 可比性：与 Idea A 总体 RMSE **INCOMPARABLE**。后续只有在动作、可见信息、预算对齐并真正调用这三个类时，才可能另标 ADAPTED

### Hugging Face `combined` 及同组织配置

读的是数据卡，不是 parquet。组织下四个数据集结构相同，各有 11 个 `train` 配置：`full`，以及 `{rm,dpo}_{infomax,dts,maxminlcb,deltaucb,drts}`。

| 数据集 | 卡片写的规模 | prompt 来源 | 偏好从哪来 |
| --- | --- | --- | --- |
| `ActiveUltraFeedback/ultrafeedback` | 60k samples | `allenai/ultrafeedback_binarized_cleaned`（UltraFeedback，AllenAI 发布版） | ActiveUltraFeedback 流水线生成回复，再用主动选择挑一对，由 LLM-as-a-Judge 标注 |
| `ActiveUltraFeedback/skywork` | 80k samples | `Skywork/Skywork-Reward-Preference-80K-v0.2` | 同上 |
| `ActiveUltraFeedback/combined` | 140k samples | 上述 UltraFeedback 与 Skywork 的合并 | 同上 |
| `ActiveUltraFeedback/tulu3` | 272k samples | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 同上 |

覆盖：卡片说 `full` 保留每个 prompt 的 30 条回复；其余配置只保留被选中的一对。本阶段没有核对每个源 prompt 是否都出现，也没有核对 30 条是否在每一行都齐。

采集过程能否重建：成对配置的列是 `prompt`、`chosen`、`rejected`、模型名、`chosen_score`、`rejected_score` 和许可证。没有选择当时的 `rewards`、`lower_bounds`、`upper_bounds`，也没有落选回复。`full` 的 `completions` 含模型名、回复和 LLM judge 的原则评价。源码里 DRTS / DeltaUCB / InfoMax 的 `__call__` 要的是这三个 torch 张量，卡片没有把 judge 原则分定义成这三列。因此本阶段不能把 `combined` 标成可重建的 offline replay，也不能把 judge 分数当成人类偏好。

## UCB-OGD

- 题目：Adaptive Data Collection for Robust Learning Across Multiple Distributions
- Venue：ICML 2025 Main。PMLR 267:73974–73994，https://proceedings.mlr.press/v267/zang25a.html
- 作者：Chengbo Zang, Mehmet Kerem Turkcan, Gil Zussman, Zoran Kostic, Javad Ghaderi
- 算法：每一轮选一个数据源采样并标注，用 OGD 更新参数。UCB-OGD 是 Algorithm 1 的 UCB 选择加 OGD。论文同时有 Eps-OGD 与 Rand-OGD。定理给出凸损失下的 minimax regret `O(T^{1/2}(K ln T)^{1/2})`
- 仓库：本阶段没有找到官方代码。检索记录见 `02_ORIGINAL_REPRODUCTION.md`。不把 `jacobvolz8/Online-Learning` 或其他课程 UCB/OGD 笔记记成作者仓库
- Commit：无
- `independent_paper_based`：**not done**。本阶段不实现，也不做多源 logistic 诊断实验
- 原任务与原指标（论文 Table 1，是论文报告值，不是本地复现）：

| 数据 | 预算 | 模型 | 指标 | 论文中的 UCB-OGD |
| --- | --- | --- | --- | --- |
| CIFAR-10，类即数据源 | 10,000 张图像 | 三层 CNN | min / mean class accuracy | 52.3 / 66.5 |
| PASCAL VOC2012，四组共现源 | 3,000 张图像 | SSD300，VGG16 backbone | min / mean mAP@50 | 44.7 / 53.0 |
| VQAv2，按题型分三源 | 1,000 个问答对 | SmolVLM-256M-Base | min / mean per-token accuracy | 22.6 / 22.9 |
| COSMOS testbed，三相机 | 2,500 张图像 | SSD300，输入 320 | min / mean AP@50 | 61.7 / 69.2 |
| MNIST（正文分析，不是 Table 1 主行） | 2,000 张；1,000 轮，每 4 轮一批 8 张 | 同上分类设定 | 类准确率曲线 | 图 1，无单独端点表 |

- 预算单位：图像张数或问答对。与 Idea A 的审计次数、偏好对、judge 调用都不是同一列。人类标注成本论文没有按美元给出
- 硬件：CIFAR/VOC/VQA/testbed 的 CNN、SSD300、SmolVLM 训练在这台 Mac 上 **blocked**。论文摘录没有给出可在 CPU 上复现 Table 1 的官方命令
- 复现状态：未复现
- 适配：无
- 可比性：论文表与 Idea A/B 主表 **INCOMPARABLE**

## PP-bandit

- 题目：Performative Prediction with Bandit Feedback: Learning through Reparameterization
- Venue：ICML 2024。PMLR https://proceedings.mlr.press/v235/chen24al.html
- 作者：Yatong Chen, Wei Tang, Chien-Ju Ho, Yang Liu
- 算法：把 performative risk 重参数化到诱导分布参数 `φ` 上，外层零阶优化 `φ`，内层 `LearnModel` 找能诱导目标分布的 `θ`
- 仓库：https://github.com/UCSC-REAL/PP-bandit-feedback
- Commit：`d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5`（2024-05-31）。License：MIT，REAL@UCSC
- 原任务：Appendix H / Example 3 的一维玩具环境。notebook 用 Bernoulli KL，分布参数由模型参数诱导（`y**2` 型重参数化）。没有技能组，没有审计动作
- 原指标：performative risk `PR(φ)` 与 `PR(θ)` 的曲线。本地端点见下一份报告。论文正文图注只给出横轴区间，没有端点表
- 预算单位：部署次数 / 每次部署抽到的 performative 样本数。不是技能审计条数，也不是偏好对
- 硬件：numpy 数值实验，本地主 venv 约 0.81 秒
- 复现状态：官方数值脚本已跑完。图 4–6 的像素对齐 **未建立**
- 适配：无。v1 已拒绝把两层零阶法改写成 B2 上的另一个 SGD 再命名
- 可比性：B-v1.1 技能分配 vs 该一维环境 **INCOMPARABLE**

## ncvxpp

- 题目：Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss
- Venue：NeurIPS 2024。arXiv:2405.17922
- 作者：Qiang Li, Hoi-To Wai
- 算法：决策依赖分布下的 performative SGD / SGD-GD
- 仓库：https://github.com/Qiang-CU/nips2024-ncvxpp
- Commit：`ccbeeddfbf9237aa06dca3a1579529faabd8fe99`（2024-09-27）。README 标 MIT，树内没有 `LICENSE` 文件
- 原任务：`synthetic-iter` / `synthetic-nos` 是线性模型合成实验；`spam-iter` / `spam-nos` 是神经网络真实数据实验
- 原指标：训练损失、训练/测试准确率、梯度范数。论文图没有在本阶段被数字化。本地单进程终点见 `00` 与 `02`
- 预算单位：SGD 迭代与 batch（本地 batch 5，约 99426 次迭代）。不是技能审计预算 `b`
- 硬件：官方依赖含 `mpi4py` 与 torch。已跑子集是 CPU、单进程。`mpiexec -np 10` blocked。`spam-iter` 未跑
- 复现状态：`partial`
- 适配：无
- 可比性：与 Idea B 的 observability Gramian / closed-loop collapse **INCOMPARABLE**
