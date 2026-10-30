# v1.2 可比性审计

日期：2026-10-31。分支 `v1.2`，审计锁定提交将记录本文件。本文件写在任何新的对照实验、烟测或 10/30-seed 主矩阵之前。分类不根据尚未产生的数字改口。

依据：`00_EXISTING_EVIDENCE_AUDIT.md`、`01_PUBLISHED_BASELINE_REGISTRY.md`、`02_ORIGINAL_REPRODUCTION.md`，以及官方源码（ActiveUF `3cf700a` 的选择器签名、PP-bandit `d5daeff` 的一维 notebook）。没有新发明算法。没有把 Uncertainty Sampling 改名为 DRTS / DeltaUCB / HyPER。

标签只使用：`DIRECT` / `ADAPTED` / `DIAGNOSTIC` / `INCOMPARABLE`。预先锁定为 INCOMPARABLE 的格子，除非源码给出相反证据，否则保持 INCOMPARABLE。

## 0. 总结论（先读）

**没有任何格子被判为 DIRECT 或可辩护的 ADAPTED。**

这是可接受的科学结果，不是执行失败。Stage 4 的 3-seed smoke 与 Stage 5 的 10/30-seed 主矩阵因此都不启动。主表结论必须是 `PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`，不能用赢了均匀随机、特征值或启发式不确定度来代替。

Protocol A2：**BLOCKED**（对照格 INCOMPARABLE）。官方选择器需要 live RM UQ 张量 `(rewards, lower_bounds, upper_bounds)`；本仓库没有该分数，也不写启发式替身。即便隔离 venv 能导入 DRTS/DeltaUCB/InfoMax 并在合成数组上选出一对，A-v1.1b/c 的方差分配到“选哪一对响应”的映射在第 2 节被预先否决，因此 A-v1.1 vs ActiveUF 仍是 INCOMPARABLE。

Protocol B2：**INCOMPARABLE**。官方一维环境没有技能组，也没有可分配的审计动作。最小部分反馈层不能凭空造出 B-v1.1 所需的技能标签；给 B-v1.1 私有技能标签会破坏信息对齐。

UCB-OGD 的独立 logistic **DIAGNOSTIC 本阶段不实例化**（可选、不能进主表“击败论文”列）。CIFAR / VOC / VQA 跳过。

`selector_unit_check` 不是对照格。它只检查官方选择函数能否导入并在合成 RM UQ 数组上返回一对索引。填写在第 4 节；填写与否不改变上面的分类。

## 1. 格子一览

| Cell ID | 双方 | 标签 | 是否跑对照 |
| --- | --- | --- | --- |
| C-AMPLe-A | AMPLe 个性化向量 L2 vs Idea A 总体 RMSE | INCOMPARABLE | 否。沿用 partial 原设定 |
| C-HyPER-A | HyPER 路由 / RewardBench vs Idea A 审计泛函 | INCOMPARABLE | 否。blocked |
| C-Drago-A | Drago 全局兼容奖励识别 vs Idea A 有限预算审计 | INCOMPARABLE | 否。无官方代码 |
| C-ActiveUF-full-A | ActiveUF 全文 RM/DPO/judge vs Idea A RMSE | INCOMPARABLE | 否。CUDA/vLLM blocked |
| C-A2-Av11-ActiveUF | Protocol A2：A-v1.1b/c vs 官方 DRTS/DeltaUCB/InfoMax | INCOMPARABLE | 否。映射不成立；live RM UQ 缺失 |
| C-A2-UniformDR-ActiveUF | Protocol A2 辅助：Uniform cross-fit DR vs 官方选择器 | INCOMPARABLE | 否。同一协议不成立；辅助对照不能单独支撑发表比较 |
| C-UCBOGD-vision | UCB-OGD CIFAR/VOC/VQA/testbed vs Idea A/B | INCOMPARABLE | 否。无官方代码；跳过图像/VQA |
| C-UCBOGD-logistic | 独立论文 logistic DIAGNOSTIC | 允许 DIAGNOSTIC，本阶段 **未实例化** | 否。未跑；即使跑也不能宣称击败论文 |
| C-PPbandit-B | PP-bandit 一维 PR vs B-v1.1 技能分配 | INCOMPARABLE | 否。沿用原设定数值，不写技能对照 |
| C-B2-Bv11-ZO | Protocol B2：B-v1.1 vs 官方两层零阶 + 部分反馈 | INCOMPARABLE | 否。原环境不能承载分配决策 |
| C-ncvxpp-B | ncvxpp performative SGD vs Idea B Gramian | INCOMPARABLE | 否。单进程 partial 也不是同一目标 |

DIRECT 格子数：0。ADAPTED 格子数：0。

## 2. Protocol A2：预先登记的映射，以及为何否决

本节写在任何 smoke 数字之前。目的不是事后找一个能让两边数字可比的变换。

### 2.1 两边实际做什么

A-v1.1b/c（`idea_A/methods/v1.1/adaptive_allocation.py`）：

- 动作：对有限总体每个单位给出审计纳入概率 `q_i ∈ [0,1]`，约束 `E[q]=b`，再独立 Bernoulli 抽 `A_i`。
- 可见信息：协变量 / 层、已知 `q`、**已审计**残差 `e_i = Y_i - μ_Y(X_i)`。隐藏 `Y` 不进分配。
- 目标：近似最小化分层 AIPW 方差 `Σ_h n_h σ̂_h² / q_h`。规则是教科书 Neyman：`q_h ∝ σ̂_h`。
- 输出：长度 `n` 的概率向量，不是一对索引。
- 没有 completion、没有奖励模型、没有 `(lower, upper)` 置信带。

官方 DRTS / DeltaUCB / InfoMax（`activeuf/acquisition_function/{drts,deltaucb,infomax}.py`，commit `3cf700a`）：

- 动作：对每个 prompt 从 `n_completions` 条回复里选出两个下标。顺序任意，由 oracle 决定 chosen/rejected。
- 可见信息：三个 `torch.Tensor`，形状 `(n_prompts, n_completions)`：`rewards`、`lower_bounds`、`upper_bounds`。源码 `loop/utils.py` 的 `compute_rewards_with_uncertainty_bounds` 从 **现场奖励模型**（含 UQ）得到这三列。
- 目标：生成后续 RM / DPO 训练用的偏好对。
- 输出：`list` of `(i, j)`。
- Hugging Face `combined` 成对配置没有这三列，也没有落选回复；不能重建采集过程（见 `01`）。

### 2.2 考虑过的映射，以及否决理由

这些映射在看测试结果之前全部否决。否决后不实现对应适配器。

**M1. 每个单位当成一个 prompt，再发明两条 dummy completion（审计 / 不审计），用 `μ_Y` 当 reward、残差 σ 当 bounds。**

否决。Idea A 的动作是对标量结局做 Bernoulli 纳入，不是比较两条回复。Dummy completion 不在合成世界里。残差 σ 来自已审计 `Y`，不是 RM UQ。这会把官方选择器接到伪造输入上。

**M2. `n` 个单位当成同一 prompt 的 `n` 条 completion，把 `-σ̂_{stratum(i)}` 或 AIPW 方差贡献写成 rewards/bounds，让 DRTS 选出两个单位再只审计这两个。**

否决。动作空间从 `q ∈ [0,1]^n`（期望预算 `b n`）变成“选 2 个单位”。这不是 A-v1.1，也不是 ActiveUF 的偏好对选择。官方方法在 oracle 偏好后还要再训练 RM；这里没有 RM。把 DRTS 当 2-子集抽样器是误用。

**M3. 按 `|σ_i-σ_j|` 或 `σ_i+σ_j` 给无序对打分，声称这就是 DeltaUCB / InfoMax。**

否决。这是新启发式。禁止把不确定度规则改名为 DeltaUCB / InfoMax / DRTS / HyPER。

**M4. 另造一个合成偏好对池，把“对级 AIPW 方差”当作 A-v1.1 的分配对象。**

否决。Idea A 没有偏好结局。把 Neyman 公式改写到对上是新算法。用户要求不新发明算法。

**M5. 用 HF `chosen_score` / `rejected_score` 当 rewards，用分数差伪造 bounds，做 offline replay。**

否决。Stage 2 已经写明：成对列缺少 `lower_bounds` / `upper_bounds` 和未入选回复；judge 分数不是人类偏好。伪造 bounds 就是启发式替身。指令：若选择器仍需要我们没有的 live RM UQ，Protocol A2 **BLOCKED**，不写启发式替身。

### 2.3 预注册结论

不存在从 A-v1.1b/c 方差分配到“选哪一对响应”的、能保持双方动作、可见信息与预算单位的映射。因此：

- `C-A2-Av11-ActiveUF` = **INCOMPARABLE**
- 即使官方选择器在隔离 venv 中导入成功，并在合成 `(rewards, bounds)` 上返回一对索引，该格仍是 INCOMPARABLE
- 不实现 Protocol A2 对照适配器
- Uniform cross-fit DR 只是 Idea A 内部辅助对照，不能单独支撑“超过发表方法”

## 3. 十三维逐格填写

每一格按同一 13 项：任务、数据集、动作、模型、初始化、反馈、预算、oracle、训练、评价指标、算力、超参数、种子、部署。未对齐的项写明差在哪里。

### 3.1 C-AMPLe-A — INCOMPARABLE（预锁定）

| 维 | AMPLe official partial | Idea A v1.1 |
| --- | --- | --- |
| 任务 | 用比较查询识别单纯形上的多维个性化偏好向量 | 有限总体上估计标量泛函（均值 / 加权均值） |
| 数据集 | `minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271`，RM 已打分的回复对 | 合成 Selective Feedback（MNAR / zero-overlap） |
| 动作 | 每轮从候选对里按 volume 选一对查询 | 每单位 Bernoulli 审计 `A_i ~ Bern(q_i)` |
| 模型 | 单纯形偏好 + MH 后验；volume 打分 | ridge `μ_Y` + logistic `μ_P` + AIPW |
| 初始化 | `burnin=50000`，`init=previous` | 试点均匀审计或 `q=b` |
| 反馈 | 比较查询的偏好 | 被审计单位的标量 `Y`；自然通道可 MNAR |
| 预算 | 查询轮数 `n_rounds`（本地 3；README 100）。burn-in 另计 | 审计比例 `b`（期望纳入分数） |
| oracle | 数据集里已有 RM / 比较结果 | 审计通道揭开隐藏 `Y` |
| 训练 | 后验更新偏好向量 | 折外 nuisance；不读隐藏标签 |
| 评价指标 | 到 `true_preference` 的 L2 | 总体 RMSE、bias、Wald coverage |
| 算力 | 隔离 venv CPU torch，约 98 秒（3 轮） | 主 venv numpy |
| 超参数 | `beta=2`，`true_beta=10`，`acquisition=volume` | `ridge_lambda`、`q_floor`、漂移 κ（冻结） |
| 种子 | 官方命令 `--seed=1` | v1.1 开发/测试种子协议 |
| 部署 | 无闭环策略部署 | 静态估计与序贯闭环是两条轨道 |

预锁定保持。两边数字不相减。`volume_adapted_ipw` 仍是合成世界启发式，不进入本格。

### 3.2 C-HyPER-A — INCOMPARABLE（预锁定）

| 维 | HyPER | Idea A v1.1 |
| --- | --- | --- |
| 任务 | 把实例路由到人类或 LM 标注，优化下游 RM | 有限预算审计估计 |
| 数据集 | 官方混合偏好语料 / RewardBench | 合成总体 |
| 动作 | human vs LM 路由 | 单位级审计概率 |
| 模型 | Tulu-2 13B / Llama-3.1-8B RM，TPU EasyLM | ridge / logistic |
| 初始化 | 官方代理 RM 训练 | 试点审计 |
| 反馈 | 人类标注或 LM 标注 | 审计 `Y` |
| 预算 | 人类条数与 LM 条数分开 | `b` 审计比例 |
| oracle | 人类；LM 是廉价标注器不是同一 oracle | 审计通道 |
| 训练 | 8B/13B RM、DPO，Beaker / GCS | 本地 nuisance |
| 评价指标 | RewardBench / 下游 RM | 总体 RMSE |
| 算力 | TPU v3-128、vLLM；本机 blocked | CPU |
| 超参数 | 官方 docs | 冻结 `config.yaml` |
| 种子 | 未在本机跑 | v1.1 协议 |
| 部署 | 下游 RM 服务 | 无 |

本机 blocked。Uncertainty Sampling 保持原名。

### 3.3 C-Drago-A — INCOMPARABLE（预锁定）

| 维 | Drago et al. ICML 2026 | Idea A v1.1 |
| --- | --- | --- |
| 任务 | 理论：全局兼容奖励识别 | 有限预算审计 RMSE |
| 数据集 | 论文设定 | 合成总体 |
| 动作 | 偏好查询（理论复杂度） | `q_i` |
| 模型 | 兼容奖励类 | AIPW |
| 初始化 | 理论 | 试点 |
| 反馈 | 偏好 | 标量 `Y` |
| 预算 | 查询复杂度 | `b` |
| oracle | 偏好 oracle | 审计 |
| 训练 | 无官方代码 | 本地 |
| 评价指标 | 识别 | RMSE |
| 算力 | 无本地运行 | CPU |
| 超参数 | 论文 | 冻结 |
| 种子 | 无 | v1.1 |
| 部署 | 无 | 静态 / 闭环 |

无官方仓库。不自写作者实现。

### 3.4 C-ActiveUF-full-A — INCOMPARABLE（预锁定）

| 维 | ActiveUltraFeedback 全文 | Idea A v1.1 |
| --- | --- | --- |
| 任务 | 从 prompt 池生成偏好数据，再训 RM / DPO/CPO | 总体泛函估计 |
| 数据集 | UltraFeedback / Skywork / Tulu3 prompts；LLM judge | 合成 `Y` |
| 动作 | 生成约 30 条回复，用选择器挑一对 | 审计纳入 |
| 模型 | Skywork-Reward-V2-Qwen3-4B（ENN UQ）、Qwen3-32B oracle | ridge / logistic |
| 初始化 | 官方 RM checkpoint | 试点 |
| 反馈 | LLM-as-a-Judge 偏好；卡片写明 RM score | 审计 `Y` |
| 预算 | prompt 数、标注回复数、偏好对数、judge 调用分开记 | `b` |
| oracle | 大 LM judge，不是人类 | 审计通道 |
| 训练 | RM 再训练、DPO/CPO | nuisance 拟合 |
| 评价指标 | GSM8K、IFEval、TruthfulQA、AlpacaEval、RewardBench 切片 | RMSE |
| 算力 | CUDA 13、vLLM 0.19、RTX 5090 / GH200；Mac blocked | CPU |
| 超参数 | `configs/loop.yaml` | 冻结 |
| 种子 | 官方 sweep | v1.1 |
| 部署 | 下游策略模型 | 无同等部署 |

全文响应生成、RM 训练、DPO/CPO 保持 blocked。LLM judge ≠ 人类偏好。

### 3.5 C-A2-Av11-ActiveUF — INCOMPARABLE（本阶段锁定）

计划中唯一可能的 ADAPTED 压力测试。第 2 节否决映射后，本格不能升格。

| 维 | 若强行共用“候选池” | 实际差在哪里 |
| --- | --- | --- |
| 任务 | 计划设想：已有候选池，自然观测删失，额外审计预算固定 | ActiveUF 任务是挑偏好对；A-v1.1 任务是估标量泛函。不是同一任务 |
| 数据集 | 需要带 RM UQ 的 completion 池 | Idea A 世界没有 completion。HF `combined` 不能重建 UQ。没有共享池 |
| 动作 | 两边都应输出“选哪一对” | A-v1.1 输出 `q_i`。官方输出 `(i,j)`。动作不对齐 |
| 模型 | 官方要 ENN RM UQ；A-v1.1 要 AIPW nuisance | 信息源不同。把残差写成 bounds 是伪造 |
| 初始化 | 官方依赖已训 RM；A-v1.1 依赖试点审计残差 | 初始化不对齐 |
| 反馈 | 官方：oracle 偏好后再训 RM。A-v1.1：揭开 `Y` | 反馈类型不同 |
| 预算 | 偏好对条数 vs 期望审计分数 `b` | 单位不能合并 |
| oracle | 官方选择器的 `__call__` **不读** oracle 标签；oracle 只在选完之后标对 | A-v1.1 分配也不读隐藏 `Y`。这一点本来可以对齐，但前面几维已经失败 |
| 训练 | 官方外环再训 RM。A-v1.1 拟合 `μ_Y` | 训练对象不同 |
| 评价指标 | 官方下游 benchmark vs 总体 RMSE | 不能放同一列 |
| 算力 | 选择器本身：torch CPU 可能够用；UQ 来源：需要 RM | 没有 live RM UQ 则选择器没有合法输入 |
| 超参数 | DRTS `max_iterations=30`，`beta=1`；DeltaUCB/InfoMax `beta=1` | A-v1.1 的 `q_floor`、漂移 κ。没有共同超参表面 |
| 种子 | 官方 `np.random` / torch；DRTS 有随机扰动 | v1.1 种子协议 |
| 部署 | 官方无 Idea A 意义上的策略部署 | 序贯闭环是另一条轨道 |

Protocol A2 **BLOCKED**：缺少 live RM UQ，且映射不成立。不写启发式替身。不跑 smoke。

### 3.6 C-A2-UniformDR-ActiveUF — INCOMPARABLE

Uniform cross-fit DR 是 Idea A 内部必须比较的基线，不是发表方法。没有合法的 A2 协议表面时，它也不能单独与 DRTS 比。辅助对照不能支撑“超过发表方法”。

### 3.7 C-UCBOGD-vision — INCOMPARABLE（预锁定）

| 维 | UCB-OGD 论文 Table 1 | Idea A / Idea B |
| --- | --- | --- |
| 任务 | 多分布稳健学习：每轮选一个源采样标注，OGD 更新 | 审计泛函 / 技能可观测性 |
| 数据集 | CIFAR-10、VOC2012、VQAv2、COSMOS | 合成 |
| 动作 | 选数据源 | 选单位或技能的审计 `q` |
| 模型 | CNN / SSD300 / SmolVLM | ridge / Gramian |
| 初始化 | 论文设定 | 冻结配置 |
| 反馈 | 图像或 QA 标签 | `Y` 或技能标签 |
| 预算 | 图像张数或 QA 对数 | `b` |
| oracle | 标准监督标签 | 审计 |
| 训练 | 深度模型 | 线性 / SGD |
| 评价指标 | min/mean class acc、mAP、token acc | RMSE / worst-skill MSE |
| 算力 | 论文训练；本机 blocked | CPU |
| 超参数 | 论文 | 冻结 |
| 种子 | 论文 | v1.1 |
| 部署 | 无同等 | 闭环策略 |

无官方代码。跳过 CIFAR / VOC / VQA。不发明作者实现去复现 Table 1。

### 3.8 C-UCBOGD-logistic — 允许 DIAGNOSTIC，本阶段未实例化

论文 Algorithm 1（UCB 选源 + OGD）在一个很小的多源 logistic 合成问题上，可以标成 **DIAGNOSTIC**：独立论文实现，不进主表“击败论文”列，不与 Idea A RMSE 或 Idea B worst-skill 放同一列。

本阶段选择 **不实例化**。原因：该格即使跑完也不能产生 DIRECT/ADAPTED 对照；Stage 3 的任务是只实现合法适配器。没有官方代码时，再写一套 logistic 不会改变“发表基线比较未建立”。保持 `independent_paper_based = not done`。

### 3.9 C-PPbandit-B — INCOMPARABLE（预锁定）

| 维 | PP-bandit official 1-D | B-v1.1 |
| --- | --- | --- |
| 任务 | 最小化一维 performative risk | 闭环下控制技能可观测性，主指标 worst-skill MSE |
| 数据集 | Appendix H Example 3：Bernoulli，`f`/`g` 解析风险 | Env1–3 合成技能群体 |
| 动作 | 对标量 `φ` 或 `θ` 做两点零阶扰动；内层 `LearnModel` 用 Bernoulli KL 把 `φ` 配到 `θ` | 对 batch 内个体分配技能感知的 `q_i` |
| 模型 | 1 维重参数化；`y²` 型 | 线性评分 + Gramian |
| 初始化 | `x0=0.3` 或 `linspace(0.25,0.8,10)` | Env 规定初值 |
| 反馈 | **完整**标量 PR(`x±βd`)，无删失 | 自然观测随策略删失 + 有限审计 |
| 预算 | 部署次数 / 两点查询次数 | 期望审计条数 `b` |
| oracle | 每次查询看到完整风险 | 审计揭开标签；无技能 oracle 给学习器 |
| 训练 | 外层零阶、内层零阶，不是 SGD | ridge / SGD 在已付费标签上 |
| 评价指标 | `PR(φ)`、`PR(θ)` 曲线 | late worst-skill MSE、regret、恢复时间 |
| 算力 | 主 venv numpy，约 0.81 秒 | CPU |
| 超参数 | `alpha`、`beta`、`num_iterations` | 冻结 `config.yaml` |
| 种子 | notebook 无 seed；本地脚本 seed 0 | v1.1 30 seed |
| 部署 | 每步部署诱导分布的 `θ` | 每步部署当前参数，反馈内生 |

源码没有 skill、技能组或审计动作。B-v1.1 技能分配放在该原环境上仍是 INCOMPARABLE。不把 notebook 跑通写成图 4–6 对齐。

### 3.10 C-B2-Bv11-ZO — INCOMPARABLE

计划允许：在官方两点零阶重参数化上加 **最小** 部分反馈层，保留他们的算法，不用 SGD 替换。B-v1.1 只能使用该环境实际暴露的动作和观测。

官方环境实际暴露的东西：

- 动作：选择一维 `x`，查询 `func(x±βd)` 以及内层 `LearnModel(x_target)`。
- 观测：两次完整标量风险（以及 KL）。没有个体、没有特征 `φ`、没有技能标签、没有“对哪些坐标少看一眼”的分配向量。
- `dim = 1`。两点扰动的方向是 ±1 的标量，不存在跨组分配。

若加一层独立随机删失，使某次 `func` 查询以概率 `1-b` 缺失：那是标量缺失，不是技能分配。B-v1.1 的 EqualSkill / E-optimal / worst-skill look-ahead 需要技能标签和 per-skill Gramian。原环境没有这些对象。

若为了让 B-v1.1 上场而加入技能标签，则学习器得到了发表方法得不到的私有信息，格子仍 INCOMPARABLE。

结论：原环境 **不能支持分配决策**。B2 保持 INCOMPARABLE。不发明额外信息，不写部分反馈适配器，不把零阶法改写成多技能 SGD。

### 3.11 C-ncvxpp-B — INCOMPARABLE（预锁定）

| 维 | ncvxpp synthetic-iter | Idea B |
| --- | --- | --- |
| 任务 | 非凸 performative 损失上的 SGD / SGD-GD | 可观测性控制 |
| 数据集 | 官方 `data.csv` 线性合成；标签 `{+1,-1}` 与平衡代码 `Y==0` 的交互已记录 | Env1–3 |
| 动作 | 参数 SGD 步 | 审计分配 |
| 模型 | 线性 performative | 技能线性 |
| 初始化 | 官方 | Env |
| 反馈 | performative 分布偏移后的 batch | 删失 + 审计 |
| 预算 | 迭代 × batch（本地 ~99426） | `b` |
| oracle | 监督标签 | 审计 |
| 训练 | performative SGD；官方 `mpiexec -np 10` blocked | ridge / SGD |
| 评价指标 | loss / acc | worst-skill MSE |
| 算力 | CPU 单进程 partial | CPU |
| 超参数 | `eps` ∈ {0,0.1,10,100} | 冻结 |
| 种子 | 官方 | v1.1 |
| 部署 | 决策依赖分布 | 闭环策略 |

SGD 不替换 PP-bandit 的两层零阶法。单进程不是全文复现。

## 4. selector_unit_check（不是对照格）

本节在分类锁定之后填写。它不改变第 1–3 节的 INCOMPARABLE 判定。

- 主 venv：无 torch。Stage 2 导入失败：`ModuleNotFoundError: No module named 'torch'`。本阶段没有往主 venv 安装 torch / vLLM / CUDA。
- 隔离 venv：`/Users/grantwright/Desktop/科研/10.30新课题/v0/.isolated_baselines_venv`，Python 3.12.4，`torch 2.15.0`，`cuda=false`。未导入 vLLM。
- 脚本：`idea_A/experiments/v1.2/run_activeuf_selector_unit_check.py`
- 产物：`results/v1.2/idea_A/original/activeuf_selector_unit_check/`（`config.yaml`、`metrics.json`、`metrics.csv`、`environment.json`、`run.log`）
- 输入：2 个 prompt × 4 条 completion 的合成 `rewards` / `lower_bounds` / `upper_bounds`。不是 oracle 标签，不是 Idea A 的 `Y`。三个选择器共用同一候选池；每个 prompt 预算 1 对。

导入与调用结果（隔离 venv，seed 0）：

| 选择器 | 官方类 | 官方文件 | `file_is_official` | 选出的下标 |
| --- | --- | --- | --- | --- |
| DRTS | `DoubleReverseThompsonSampling` | `activeuf/acquisition_function/drts.py` | true | `[3,0]`，`[1,0]` |
| DeltaUCB | `DeltaUCB` | `activeuf/acquisition_function/deltaucb.py` | true | `[3,0]`，`[1,0]` |
| InfoMax | `InfoMax` | `activeuf/acquisition_function/infomax.py` | true | `[2,0]`，`[2,0]` |

结论：

- **隔离 venv 的 DRTS / DeltaUCB / InfoMax 导入成功**，并能从合成分数/不确定度数组选出一对。
- 调用的是官方类文件，不是本地重写。`vllm_imported = false`。
- 选择器 `__call__` 仍然只接受 `(rewards, lower_bounds, upper_bounds)`。真实候选池上的合法输入来自 live RM UQ。本仓库没有该分数，HF `combined` 也不能重建。因此 Protocol A2 对照仍 **BLOCKED**。
- 本检查 **不是** DIRECT/ADAPTED 格子，不触发 3-seed smoke。

## 5. 适配器与实验策略（锁定）

| 对象 | 做 / 不做 |
| --- | --- |
| Protocol A2 对照适配器 | **不做**。映射不成立；无 live RM UQ |
| 把 AIPW 残差改写成 RM bounds | **不做**。启发式替身 |
| 把 Uncertainty Sampling 叫 DRTS/DeltaUCB/HyPER | **不做** |
| Protocol B2 部分反馈适配器 | **不做**。原环境无分配动作 |
| 给 PP-bandit 加技能标签 | **不做**。私有信息 |
| 用 SGD 替换两层零阶 | **不做** |
| UCB-OGD 论文 logistic DIAGNOSTIC | **本阶段不实例化** |
| CIFAR/VOC/VQA | **跳过** |
| 官方选择器 unit check | 允许；见第 4 节与 `results/v1.2/idea_A/original/activeuf_selector_unit_check/` |
| 3-seed smoke / 10/30-seed | **不跑**。没有 DIRECT 或 ADAPTED 格 |
| 五份最终对照报告 | 本阶段不写 |

## 6. 本文件锁定后的允许动作

1. 用隔离 venv 尝试导入官方选择器，把第 4 节从 pending 改成实测（导入成功或失败）。这不是对照实验。
2. 若导入成功：保存 tiny JSON，测试“调用的是官方类文件，不是重写”。
3. 提交本审计、测试、gitignore、unit-check JSON。不提交完整 ActiveUF 克隆、不提交 v1.1 plots、不提交脏的 environment_manifest / smoke 报告。不 push。
