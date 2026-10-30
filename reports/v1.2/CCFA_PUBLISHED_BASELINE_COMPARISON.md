# v1.2 已发表 CCF-A 基线对照（主交付）

本文件由 `scripts/generate_v1_2_report.py` 从磁盘 JSON 与已锁定审计写出。数字均来自下列产物，没有手填对照 RMSE，也没有发明 10/30-seed CCF-A head-to-head。

- 生成时间（UTC）：`2026-10-30T20:17:05.246492+00:00`
- 仓库：`/Users/grantwright/Desktop/科研/10.30新课题/v0`
- 分支：`v1.2`
- commit：`fd00badccc8f1e4a8012b91c2b9d473914f4f3c5`
- remote：`git@github.com:SelinaMai-source/idea-A-B.git`
- 缺失的主源：无
- 可比性锁：`reports/v1.2/03_COMPARABILITY_AUDIT.md`
- 主判定：`PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`
- scientifically valid DIRECT 对照数：`0`；可辩护 ADAPTED 对照数：`0`

**判定预先冻结在 `03_COMPARABILITY_AUDIT.md`。** 没有 DIRECT 格子，也没有可辩护的 ADAPTED 格子。Protocol A2 与 Protocol B2 均为 INCOMPARABLE / BLOCKED。隔离 venv 的 `selector_unit_check` **不是** head-to-head。不得把 v1.1 相对 Uniform DR / E-optimal 的结果写成击败 CCF-A 论文。Idea A / Idea B **不得**标为 PROMISING；v1.1 已相对经典对照判定 **NOT PROMISING**。

## 1. Executive Summary

- **相对已发表 CCF-A Main 方法的对照状态：`PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`。**
- scientifically valid DIRECT 对照数：`0`。可辩护 ADAPTED 对照数：`0`。
- Protocol A2（A-v1.1 vs 官方 DRTS / DeltaUCB / InfoMax）：**BLOCKED / INCOMPARABLE**。缺少 live RM UQ；A-v1.1 的方差分配到“选哪一对响应”的映射在看测试结果之前已被否决。
- Protocol B2（B-v1.1 vs 官方两层零阶 + 部分反馈）：**INCOMPARABLE**。PP-bandit 一维环境没有技能组，也没有可分配的审计动作。
- Idea A v1.1 相对经典对照：`NOT PROMISING`（输给 Uniform cross-fit DR）。这 **不能** 改写成发表基线胜利或失败。
- Idea B v1.1 相对经典对照：`NOT PROMISING`（worst-skill MSE 输给 E-optimal）。同样 **不能** 代替 CCF-A 对照。
- Idea A / Idea B 在本阶段 **均不得** 称为 PROMISING。
- v1.3 算法迭代：**不成立**。本机当前没有合法对照路径（无 RM UQ，也无 skill-free 映射）。

## 2. Verified Published Papers

| ID | 题目 | Venue | 仓库 / commit | 复现状态 | 与 Idea A/B |
| --- | --- | --- | --- | --- | --- |
| AMPLe | Comparison-based Active Preference Learning for Multi-dimensional Personalization | ACL 2025 Main | https://github.com/ml-postech/AMPLe `0cad341f6a9e77063ef451c27732bf5c1534c797` | partial | INCOMPARABLE |
| HyPER | Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback | ACL 2025 Main | https://github.com/allenai/hybrid-preferences `4d96b27b989828b0b69a430f68d25b6b36e9015a` | blocked | INCOMPARABLE |
| Drago | Online Compatible Reward Identification from Preference Feedback | ICML 2026 | 无官方仓库 | blocked | INCOMPARABLE |
| ActiveUF | ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning | ICML 2026 Main | https://github.com/lasgroup/ActiveUltraFeedback `3cf700a85bbe344b1b472890a488671e2916a8b4` | 全文 blocked；选择器 unit check 仅 DIAGNOSTIC | INCOMPARABLE |
| UCB-OGD | Adaptive Data Collection for Robust Learning Across Multiple Distributions | ICML 2025 Main | 未找到官方仓库 | independent_paper_based = not done；CIFAR/VOC/VQA blocked | INCOMPARABLE |
| PP-bandit | Performative Prediction with Bandit Feedback: Learning through Reparameterization | ICML 2024 | https://github.com/UCSC-REAL/PP-bandit-feedback `d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5` | 数值脚本已跑完；未对齐图 4–6 端点 | 技能分配 INCOMPARABLE |
| ncvxpp | Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss | NeurIPS 2024 | https://github.com/Qiang-CU/nips2024-ncvxpp `ccbeeddfbf9237aa06dca3a1579529faabd8fe99` | 单进程 partial；`mpiexec -np 10` blocked | INCOMPARABLE |

InfoMax 出现在 ActiveUltraFeedback 仓库中，但 README 把它标成已有 dueling-bandit 方法（Saha, 2021），**不是** 该论文的新算法。DRTS（源码类名 `DoubleReverseThompsonSampling`）与 `DeltaUCB` 才是该文的新选择器。

## 3. Original Problem Formulations

- **AMPLe**：在 RM 打分的回复对上，用 volume / random acquisition 识别单纯形上的多维个性化偏好向量。指标是到 `true_preference` 的向量误差，不是有限总体标量泛函 RMSE。
- **HyPER**：把实例路由到人类或 LM 标注，优化下游 RM / RewardBench。
- **Drago et al.**：理论工作，全局兼容奖励识别，不是有限预算审计 RMSE。
- **ActiveUltraFeedback**：从 prompt 池生成回复，用不确定度奖励模型选出一对，经 LLM-as-a-Judge 标注，再训练 RM / DPO/CPO。动作是选 completion 下标对，可见信息是 `(rewards, lower_bounds, upper_bounds)`。
- **UCB-OGD**：每轮选一个数据源采样并标注，用 OGD 更新；论文 Table 1 是 CIFAR-10 / VOC2012 / VQAv2 / COSMOS 上的类准确率或 mAP。
- **PP-bandit**：把 performative risk 重参数化到诱导分布参数 `φ`，外层零阶优化 `φ`，内层 `LearnModel` 找能诱导目标分布的 `θ`。官方 notebook 是一维 Bernoulli，没有技能组。
- **ncvxpp**：决策依赖分布下的 performative SGD / SGD-GD。官方 `synthetic-iter` 是线性模型合成实验。
- **Idea A v1.1**：有限总体上对每个单位给出审计概率 `q_i`，估计标量泛函。动作不是选回复对。
- **Idea B v1.1**：在技能群体上分配审计，主指标 late worst-skill MSE。官方 PP-bandit 环境没有这些对象。

## 4. Original Benchmark Reproduction

本阶段没有重跑 AMPLe burn-in、PP-bandit notebook、ncvxpp。下列数字来自已提交 JSON。

### AMPLe partial

- 来源：`idea_A/baselines/AMPLe/official_reproduction/metrics.json`
- `acquisition=volume`，`burnin=50000`，`n_rounds=3`（README 示例 100 未跑）
- 数据集：`minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271`
- `true_preference = [0.2, 0.7, 0.1]`
- `l2_to_true_t0 = 0.4688851731916284`
- `l2_to_true_t3 = 0.3647449275776112`
- `estimation_t0 = [0.3249063193798065, 0.32413744926452637, 0.3509562909603119]`
- `estimation_t3 = [0.0035132216289639473, 0.9920580983161926, 0.004428689833730459]`
- `chosen_indices` 来自 chosen 1/2/3：`[737, 95, 135]`
- `elapsed_wall_sec_approx = 97.8`
- 该 L2 **不可** 与 Idea A RMSE 相减。

### PP-bandit 数值运行

- 来源：`idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/metrics.json`
- `numpy_seed = 0`，`elapsed_sec = 0.814453125`
- 单次 `num_iterations=500`，`x0=0.3`：`final_PR_phi = -0.2816498886156836`，`final_PR_theta = -0.2962457875065718`
- 10 个初值各 100 步：`PR_phi_final_mean = -0.2883591540722531`（std `0.007139595668885597`），`PR_theta_final_mean = -0.291834361799254`（std `0.007228858237631779`）
- 论文 Appendix H 图 4–7 没有可抄端点表。**不声称对齐图 4–6。**

### ncvxpp 单进程 partial

- 来源：`idea_B/baselines/REPRODUCTION_AUDIT.json`
- `local_status = partial`；官方 `mpiexec -np 10` blocked
- `eps_list = [0, 0.1, 10, 100]`，`batch = 5`，`wall_sec = 540.486`
| npy | iter_last | loss_last | train_acc_last | test_acc_last |
| --- | --- | --- | --- | --- |
| eps0-batch5-res0.npy | 99426 | 0.3281349241733551 | 0.9004424778761062 | 1 |
| eps0.1-batch5-res0.npy | 99426 | 0.3458000421524048 | 0.9004424778761062 | 1 |
| eps10-batch5-res0.npy | 99426 | 0.3553709387779236 | 0.9004424778761062 | 1 |
| eps100-batch5-res0.npy | 99426 | 0.9450080394744873 | 0.09955752212389381 | 0 |

### ActiveUF 选择器 unit check（不是复现论文表）

- 来源：`results/v1.2/idea_A/original/activeuf_selector_unit_check/metrics.json`
- `check_type = selector_unit_check`，`comparability_cell = null`，`protocol_a2 = BLOCKED_INCOMPARABLE`
- 隔离 venv Python：`/Users/grantwright/Desktop/科研/10.30新课题/v0/.isolated_baselines_venv/bin/python`；`torch_version = 2.15.0`；`torch_cuda = false`；`vllm_imported = false`
- 合成池：`n_prompts = 2`，`n_completions_per_prompt = 4`，每 prompt 预算 1 对
| 选择器 | 官方类 | file_is_official | selected_pairs |
| --- | --- | --- | --- |
| DRTS / DoubleReverseThompsonSampling | DoubleReverseThompsonSampling | True | [[3, 0], [1, 0]] |
| DeltaUCB | DeltaUCB | True | [[3, 0], [1, 0]] |
| InfoMax | InfoMax | True | [[2, 0], [2, 0]] |
- 这只证明官方类能在合成 RM UQ 张量上返回一对索引。它 **不是** DRTS vs A-v1.1 的对照。

### HyPER / Drago / UCB-OGD / ActiveUF 全文

- HyPER：blocked（TPU v3-128、vLLM、8B/13B RM）。无本地数字。
- Drago：blocked，无官方代码。无本地数字。
- UCB-OGD：无官方仓库；`independent_paper_based = not done`。CIFAR / VOC / VQA blocked。论文 Table 1 数字不抄进对照列。
- ActiveUF 全文响应生成、RM 训练、DPO/CPO：blocked（CUDA 13 / vLLM 0.19）。

## 5. Reproduction Differences

- AMPLe：`n_rounds` 从 README 100 改为 3；`--dims` 省略（2-aspect 数据集当时 HTTP 401）；`burnin=50000` 与 `acquisition=volume` 未改。
- PP-bandit：本地脚本加了 `numpy` seed 0；官方 notebook 无 seed。因此 JSON 是同一算法、固定种子的一次执行，不是对论文图像素核对。
- ncvxpp：用 `mpi4py` stub，`rank=0, size=1`。不是 `mpiexec -np 10`。`dataset.py` 按 `Y==1`/`Y==0` 平衡，而 `data.csv` 标签是 `{+1,-1}`，只有 `+1` 类留下——官方代码与数据的交互，未改写。`spam-iter` 未跑。
- ActiveUF：浅克隆、submodule 未初始化；主 venv 无 torch；隔离 venv 只跑选择器 unit check，不跑 vLLM / RM。
- HyPER：浅克隆，submodule 未初始化，无训练。
- UCB-OGD：没有作者代码，因此没有任何“复现差异”可列，只有缺失。

## 6. Benchmark Compatibility

完整十三维填写见 `reports/v1.2/03_COMPARABILITY_AUDIT.md` 与 `reports/v1.2/BENCHMARK_COMPATIBILITY.md`。摘要：

- DIRECT 格子数：`0`。ADAPTED 格子数：`0`。
- Protocol A2：**BLOCKED**。动作（`q_i` vs `(i,j)`）、可见信息（审计残差 vs live RM UQ）、预算单位（审计比例 `b` vs 偏好对）均不齐。映射 M1–M5 预先否决。
- Protocol B2：**INCOMPARABLE**。官方环境暴露的是一维 `x` 与完整标量 PR 查询，没有技能标签，也没有分配向量。给 B-v1.1 私有技能标签会破坏信息对齐。
- AMPLe L2、HyPER RewardBench、UCB-OGD 类准确率 / mAP、PP-bandit PR、ncvxpp loss/acc 都 **不得** 放进 Idea A RMSE 或 Idea B worst-skill 的同一列。

## 7. Idea A Head-to-Head Results

**没有合法 head-to-head。** 本阶段没有 3-seed smoke，也没有 10/30-seed Protocol A2 主格子。下表 Performance / Cost 列留空，因为格子是 INCOMPARABLE，不能填合成 RMSE，也不能填论文原表数字。

| Method | Published venue | Protocol | Label budget | Main metric | Performance | Cost | Comparable? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ActiveUF DRTS | ICML 2026 Main | Protocol A2 BLOCKED | preference pairs; live RM UQ missing | not aligned with Idea A RMSE |  |  | INCOMPARABLE |
| ActiveUF DeltaUCB | ICML 2026 Main | Protocol A2 BLOCKED | preference pairs; live RM UQ missing | not aligned with Idea A RMSE |  |  | INCOMPARABLE |
| ActiveUF InfoMax | ICML 2026 framework (Saha 2021 dueling-bandit; not ActiveUF's new algorithm) | Protocol A2 BLOCKED | preference pairs; live RM UQ missing | not aligned with Idea A RMSE |  |  | INCOMPARABLE |
| AMPLe | ACL 2025 Main | original partial n_rounds=3 | query rounds (local 3; README 100); burn-in 50000 separate | preference-vector L2 (original setting only) |  |  | INCOMPARABLE |
| HyPER | ACL 2025 Main | official pipeline blocked | human vs LM annotation counts | RewardBench / downstream RM |  |  | INCOMPARABLE |
| Our A-v1.1 | Proposed (this repo) | v1.1 synthetic Selective Feedback; not Protocol A2 | audit fraction b | functional RMSE |  |  | no valid published head-to-head |
| Uniform DR | Classical | v1.1 internal control only | same b as A-v1.1 | functional RMSE |  |  | classical internal; not a CCF-A paper |

selector_unit_check（seed 0，隔离 venv）选出的下标见第 4 节。它的 `comparability_cell` 为 `null`，`protocol_a2 = BLOCKED_INCOMPARABLE`。不能从这些下标构造 DRTS vs A-v1.1 的 RMSE 差或置信区间。

v1.1 内部经典对照（**不是** CCF-A 对照，不进入上表 Performance 列）来自 `idea_A/results/v1.1/TIER2_SUMMARY.json`：

- 静态 MNAR `b=0.05`，n=`50`：A-v1.1c RMSE `0.3336984041151296` vs Uniform DR `0.1287791571117902`，diff `0.2049192470033395`，CI [`-0.03849778187446253`, `0.4420267658634702`]，`interval_side=covers_zero`。
- 闭环 MNAR-audit `b=0.05`，n=`30`：A-v1.1c RMSE `1.074943643456929` vs Uniform DR `0.1685912838364942`，diff `0.9063523596204344`，CI [`0.05659021316146642`, `1.6781512193068`]，`interval_side=entirely_positive`。
- 同静态格 observed-only RMSE `1.217386193505511` vs Uniform DR `0.1287791571117902`：这是 identification-via-audit，不是自适应分配赢了发表方法。

问题回答（任务第 20 节）：

1. 我们的方法能否在 ActiveUltraFeedback 原任务上运行？**不能**（无 live RM UQ，动作空间不是选回复对）。
2. 同等标注预算下是否超过 DRTS 或 DeltaUCB？**未建立对照，因此没有该检验。**
3. 若不能，失败条件是什么？映射不成立；live RM UQ 缺失；全文流水线 blocked。
4. 内生反馈删失是否暴露了发表方法的可测弱点？**没有合法实验回答这一点。**
5. identification-aware 干预是否针对该弱点？**没有对照，不能声称。**
6. 改进是否大于 Uniform DR？v1.1 内部结果是 A-v1.1c **没有**赢 Uniform DR；这仍不是 CCF-A 结论。
7. 任何优势是否来自额外数据 / 调参 / 算力？不存在相对发表方法的已测优势。
8. 结果是否支持独立 Method Paper？**不支持。** 发表基线比较未建立，且 v1.1 对经典对照已是 NOT PROMISING。

## 8. Idea B Head-to-Head Results

**没有合法 head-to-head。** Protocol B2 未跑。Performance / Cost 列留空。

| Method | Published venue | Protocol | Feedback budget | Main metric | Performance | Cost | Comparable? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Performative Bandit | ICML 2024 | original 1-D notebook; Protocol B2 INCOMPARABLE | deployments / two-point queries | PR(φ), PR(θ) |  |  | INCOMPARABLE |
| Performative SGD | NeurIPS 2024 | single-process partial; mpiexec blocked | SGD iterations × batch | loss / acc |  |  | INCOMPARABLE |
| Our B-v1.1 | Proposed (this repo) | v1.1 Env1–3; not Protocol B2 | expected audit fraction b | late worst-skill MSE |  |  | no valid published head-to-head |
| E-optimal | Classical | v1.1 internal control only | same b as B-v1.1 | late worst-skill MSE |  |  | classical internal; not a CCF-A paper |
| D-optimal | Classical | v1.1 internal control only | same b as B-v1.1 | late worst-skill MSE |  |  | classical internal; not a CCF-A paper |
| Uniform | Classical | v1.1 internal control only | same b as B-v1.1 | late worst-skill MSE |  |  | classical internal; not a CCF-A paper |

v1.1 内部经典对照（**不是** CCF-A 对照）来自 `idea_B/results/v1.1/TIER2_SUMMARY.json`：

- Env1 n=`30`：B-v1.1c late worst-skill MSE mean `0.03227260531905402` vs E-optimal `0.0315451787894095`；配对 diff `0.0007274265296445326`，CI [`6.742531107403646e-05`, `0.001424601651999807`]，`interval_side=entirely_positive`。
- 同格 cumulative regret 配对 diff `-0.03581008120369523`，CI [`-0.06476172173322903`, `-0.009625981971318698`]，`interval_side=entirely_negative`。H_B1 要求 worst-skill **并且** regret 相对 E 与 D 同时完全为负；该条件未满足。
- Env2 worst-skill 配对 diff `0.0009084109172013277`，CI [`0.0003653636019892656`, `0.001545863140273617`]，`interval_side=entirely_positive`。

问题回答：

1. 能否在原 performative benchmark 上合法评估我们的方法？**不能**（原环境无技能组 / 审计动作）。
2. 是否相对发表优化器降低了 performative risk？**未建立对照。**
3. 内生反馈观测扩展是否改变排序？Protocol B2 未实现，无排序。
4. 同等反馈成本下是否改进长期学习？无合法发表对照。
5. 在相关处是否超过经典信息最优控制？v1.1 内部：worst-skill 未超过 E-optimal；这不是 CCF-A 结论。
6. 是否需要发表算法得不到的信息？若强行加技能标签，是的，格子仍 INCOMPARABLE。
7. 发现的失败是在原定理假设内还是外？没有对发表方法做原任务失败诊断；B2 是原环境无法承载分配决策。
8. 是否支持独立算法或理论贡献？**不支持** 以“超过 PP-bandit / ncvxpp”为依据的贡献声明。

## 9. Extended Feedback Experiments

没有新的 policy-induced selection 扩展实验。Stage 4 的 3-seed smoke 与 Stage 5 的 10/30-seed 主矩阵按锁 **不启动**。
v1.1 闭环格子仍是合成 Selective Feedback / Env1–3，不是 ActiveUF 或 PP-bandit 的扩展协议。

## 10. Cost and Budget Fairness

预算单位继续分列，不合并成一个 budget：

- 标注次数 / 审计比例 `b`：Idea A/B 合成协议。
- 偏好对：ActiveUF 选择器输出。
- 标注回复：ActiveUF `full` 配置自称每 prompt 30 条生成回复。
- 查询轮数：AMPLe `n_rounds`（本地 3）。burn-in 50000 另计。
- 图像张数 / QA 对：UCB-OGD Table 1。
- 部署次数 / 两点查询：PP-bandit。
- SGD 迭代 × batch：ncvxpp（本地 batch 5，约 99426 次迭代）。
- 人类成本 vs judge 成本：ActiveUF 数据卡写的是 LLM-as-a-Judge / RM score，不是人类偏好。HyPER 人类条数与 LM 条数应分开，但本机未跑。

因为单位不齐，**不存在** 已匹配的 label budget 或 feedback budget 对照。selector_unit_check 的预算是每 prompt 1 对，候选池 2×4，与 Idea A 的 `b` 不是同一账本。

## 11. Implementation and Leakage Audit

- 没有实现 Protocol A2 / B2 对照适配器。测试 `idea_A/tests/test_v1_2_comparability.py` 禁止 `protocol_a2_adapter` 等启发式文件名。
- Uncertainty Sampling **没有**改名为 DRTS / DeltaUCB / HyPER。
- selector_unit_check：`oracle_labels_in_acquisition_scores = false`；`oracle_labels_passed` 对三个选择器均为 false；`call_args = ["rewards", "lower_bounds", "upper_bounds"]`。
- `all_official_files = true`。调用的是官方类文件，不是本地重写。
- `vllm_imported = false`。主 venv 仍无 torch。
- A-v1.1 分配不读隐藏 `Y`；官方选择器 `__call__` 也不读 oracle 标签。这一点本来可以对齐，但任务、动作、信息、预算已经失败，因此不构成 ADAPTED。
- learner 只能看见已付费标签：本阶段没有新对照 run 去违反该规则。

## 12. Negative Results

- **主负结果：发表基线比较未建立。** 这是 Stage 3 锁的科学结论，不是执行失败。
- Protocol A2 映射 M1–M5 全部否决；无 live RM UQ 时禁止启发式替身。
- Protocol B2 原环境不能支持分配决策。
- ActiveUF / HyPER / UCB-OGD 视觉实验 / ncvxpp MPI 在本机 blocked 或 partial。
- v1.1 已经判定 A-v1.1c、B-v1.1c 相对经典对照 **NOT PROMISING**。本阶段不把该内部失败改写成“超过发表方法失败”或“超过发表方法成功”。
- 闭环 A-v1.1c 相对 Uniform DR：`interval_side=entirely_positive`（候选更差）。
- Env1 B-v1.1c 相对 E-optimal worst-skill：`interval_side=entirely_positive`（候选更差）。

## 13. Statistical Robustness

- 对已发表方法：**没有** 配对检验、没有 10 个开发种子、没有 30 个测试种子。不得发明这些数字。
- selector_unit_check 只有 seed 0 的一次合成调用。
- AMPLe partial 是 `--seed=1` 的一次 3 轮运行。
- PP-bandit 本地脚本 seed 0；多初值 10 个点各 100 步是同一脚本内的数值重复，不是论文种子协议。
- v1.1 内部 30/50 种子结果只描述经典对照稳健性，不迁移到 CCF-A 主表。

## 14. Novelty Implications

在没有合法发表对照时，不能声称相对 DRTS / DeltaUCB / AMPLe / HyPER / PP-bandit / ncvxpp 的方法贡献。
v1.1 仍支持的现象是 identification-via-audit（MNAR 下审计通道降偏），那是相对 observed-only / Uniform DR 的识别事实，不是相对 CCF-A 主动学习论文的新算法胜利。
名字不同（审计 vs 主动偏好、技能可观测性 vs performative risk）本身不是 novelty。

## 15. Research Decision

- Idea A：继续称为 PROMISING：**否**。相对发表方法：**比较未建立**。相对经典对照：已是 NOT PROMISING。
- Idea B：同样 **否**。
- 是否同时推进 A 与 B 的 v1.3 算法迭代：**否**。当前 Mac 上没有合法对照路径（需要 live RM UQ，或 skill-free、且不发明私有技能标签的映射）。
- 详细逐条见 `reports/v1.2/A_B_RESEARCH_DECISION.md`。

## 16. Next Algorithmic Improvements

本阶段 **不提出** 新分配算法。用户任务明确：先建立合法发表对照，再考虑算法迭代。
若未来硬件/数据允许，唯一与本锁一致的下一步是 **对照路径**，不是再写一个 Neyman 变体：

1. 获得合法 live RM UQ `(rewards, lower_bounds, upper_bounds)` 与真实 completion 池后，才能重新评估 Protocol A2 是否仍被映射否决。
2. 若 PP-bandit 原环境仍无分配动作，不要为 B-v1.1 私加技能标签。
3. 在对照路径出现之前，不投资 v1.3 算法。

## 17. Exact Reproduction Commands

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD"

# 主 venv 测试（不要求 GPU / torch）
.venv/bin/python -m pytest --import-mode=importlib

# 从已有 JSON 重建 v1.2 报告（不跑对照实验）
.venv/bin/python scripts/generate_v1_2_report.py

# ActiveUF 选择器 unit check（隔离 venv，不是 Protocol A2）
.isolated_baselines_venv/bin/python idea_A/experiments/v1.2/run_activeuf_selector_unit_check.py

# 下列原设定本阶段不重跑；产物已在磁盘上
# AMPLe partial: idea_A/baselines/AMPLe/official_reproduction/metrics.json
# PP-bandit: .venv/bin/python idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/run_official_numerical.py
# ncvxpp: PYTHONPATH=official_reproduction/mpi4py_stub python synthetic-iter/main.py
```

没有 Protocol A2 / B2 对照命令，因为适配器未实现。

## 18. Artifact Index

| 路径 | 角色 |
| --- | --- |
| reports/v1.2/00_EXISTING_EVIDENCE_AUDIT.md | 已有证据审计 |
| reports/v1.2/01_PUBLISHED_BASELINE_REGISTRY.md | 发表基线登记 |
| reports/v1.2/02_ORIGINAL_REPRODUCTION.md | 原设定复现记录 |
| reports/v1.2/03_COMPARABILITY_AUDIT.md | 可比性锁；DIRECT=0 ADAPTED=0 |
| reports/v1.2/CCFA_PUBLISHED_BASELINE_COMPARISON.md | 本文件，主交付 |
| reports/v1.2/ORIGINAL_BASELINE_REPRODUCTION.md | 原设定复现汇总 |
| reports/v1.2/BENCHMARK_COMPATIBILITY.md | 可比性汇总 |
| reports/v1.2/A_B_RESEARCH_DECISION.md | 研究决策 |
| reports/v1.2/comparison_summary.json | 机器可读对照表；INCOMPARABLE 的 performance 为空 |
| results/v1.2/idea_A/original/activeuf_selector_unit_check/ | selector_unit_check JSON/CSV/log；不是对照格 |
| idea_A/baselines/AMPLe/official_reproduction/metrics.json | AMPLe partial L2 |
| idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/metrics.json | PP-bandit PR 端点 |
| idea_A/baselines/REPRODUCTION_AUDIT.json | Idea A 基线审计 |
| idea_B/baselines/REPRODUCTION_AUDIT.json | Idea B 基线审计（含 ncvxpp） |
| idea_A/results/v1.1/TIER2_SUMMARY.json | v1.1 内部经典对照；非 CCF-A 主表 |
| idea_B/results/v1.1/TIER2_SUMMARY.json | v1.1 内部经典对照；非 CCF-A 主表 |

不提交：完整 ActiveUltraFeedback 克隆、`.venv`、`.isolated_baselines_venv`、`.cache`、v1.1 逐次 PNG、脏的 environment_manifest / smoke 报告。

