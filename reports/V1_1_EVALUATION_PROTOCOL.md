# v1.1 评价协议

状态：冻结。冻结日期：2026-10-31。父提交：`f98d53f59ac98164693eaf14b53d99a499f19c4a`。本文件在任何 v1.1 Tier 1/2 主实验之前写入。后续实现和跑实验的代理遵守本文件。不得在看到 Tier 2 结果之后改主假设、主对照、种子、成功规则或主格子。

`idea_A/methods/v1.0/` 与 `idea_B/methods/v1.0/` 的行为必须保持。修复和新方法只放在 `methods/v1.1/` 以及协议点名的新环境目录。v1.0 运行器仍是 A-v1.0 与 B-v1.0 参照臂的定义。

本阶段没有实现这些算法，也没有跑本协议中的实验。

## 1. 主假设

所有确认性假设都使用预先指定格子上的配对种子。3 个种子的结果标记为 PRELIMINARY，不能单独支撑 PROMISING。

**Idea A**

- H_A1（静态，确认性）：在 MNAR 与 zero-overlap、`b ∈ {0.05, 0.10}`、30 个配对种子上，比较对象是 Uniform audit + 2 折 cross-fit DR。候选方法为 v1.1b 与 v1.1c。主估计量是总体均值 `E[Y]` 的 RMSE。
- H_A2（闭环，确认性比较，探索性阈值单独记录）：在 policy-drift 与 MNAR-audit 两条闭环、同样两个预算、30 个配对种子上，比较 v1.1c 与 Uniform audit + 2 折 cross-fit DR 的总体均值 RMSE。动态设定下相对该基线 RMSE 下降 10% 是探索性描述。达到 10% 不是自动成功。没有达到就在报告里把该探索性检查记为失败。

**Idea B**

- H_B1：在 Env1 与 Env2、3 个技能、`b = 0.05`、30 个配对种子上，v1.1c 的 worst-skill MSE 和累计 regret 低于 E-optimal，并且低于 D-optimal。
- H_B2：在 Env3 的可恢复情形上，比较恢复时间。10 个种子的结果是预算与环境稳健性证据；它不把“只赢 Uniform”写成确认性成功。
- 特征值（`λ_min`、条件数、`logdet`）是次要解释。只抬特征值不是学习成功。

Uniform 随机、Uniform IPW、observed-only 和 v1.0 Neyman 是参照臂。赢了这些参照臂不等于赢了主对照。

## 2. 主指标与主对照

### Idea A

主指标，全部要报：

1. 总体均值的 RMSE：`sqrt(mean_i (estimate_i - truth_i)^2)`，`truth` 是该种子有限总体的 `E[Y]`。
2. bias：`mean_i (estimate_i - truth_i)`。
3. CI coverage：每个种子给出名义 95% 区间，coverage 是区间盖住该有限总体真值的种子比例。
4. 审计成本：期望审计比例与实现审计比例。

主对照：Uniform audit + 2 折 cross-fit DR（A-v1.1a）。

偏好率、策略价值和分层均值可以记录，但它们不是 H_A1/H_A2 的成功指标。若报告它们，必须使用各自的 nuisance（见审计第 11 节），不能复用连续 `μ_Y`。

### Idea B

主指标：

1. worst-skill MSE：每个回合取各技能条件 MSE 的最大值，再在晚期窗口上平均。晚期窗口是最后 `max(1, horizon // 4)` 个回合，与 v1 的 `late_*` 窗口一致。
2. 累计 regret：`Σ_t max(true_mse_t - noise_std^2, 0)`，`noise_std = 0.15`，噪声底 `0.0225`。报告噪声底。
3. Env3 恢复时间：定义见第 5 节。

次要：`1/(1+MSE)`、特征值、观测表现与真实表现的差。`1/(1+MSE)` 不能作为 PROMISING 的依据。

主对照：E-optimal 与 D-optimal，两个都要比。Uniform 与修正后的 EqualSkill 是参照，不是唯一对手。

- E-optimal：在审计预算下提高条件设计 Gramian 的 `λ_min`。不把 v1.0 控制器的 smoothing 和 `mix_uniform` 算进这个对照。
- D-optimal：在同一条件 Gramian 上提高 `logdet`（或等价的预测方差准则）。同样不把 v1.0 的平滑混合算进去。

二者在 README 中标为经典对照。若 v1.1c 不优于 E-optimal，结论是 NOT PROMISING 或 INCONCLUSIVE。E-optimal 胜出不算新方法成功。

Gramian 定义保持为 `W = Σ S φ φ^T`。报告要分开已实现的观测信息、条件期望信息，以及“全体候选样本”的信息。更高的特征值单独出现时，不记为成功。

## 3. 判定

每个确认性比较使用第 6 节的配对区间。有利方向：Idea A 的 RMSE 差（候选减 Uniform cross-fit DR）为负；Idea B 的 worst-skill MSE、regret、恢复时间为负（候选更小）。

| 区间 | 该比较的标签 |
| --- | --- |
| 完全落在有利一侧 | 该指标上有改进证据 |
| 完全落在不利一侧 | 该比较为 NOT PROMISING |
| 覆盖 0 | 该比较为 INCONCLUSIVE |

PROMISING 不能表示“只打败了均匀随机”。

Idea A 的 PROMISING 同时要求：

- 确认性格子上，对主对照 Uniform cross-fit DR 的 RMSE 配对区间完全为负；
- 该格子的名义 95% 区间已经计算，coverage 已报告；
- 实现审计比例的种子均值落在 `[b - 0.01, b + 0.01]`。超出该带的方法没有在规定预算下比较。
- 动态设定的 10% 相对 RMSE 下降只作为探索性检查。没达到则该检查记失败。达到了也不自动成为 PROMISING。

Idea B 的 PROMISING 同时要求：

- 在 H_B1 的格子上，v1.1c 相对 E-optimal 和相对 D-optimal 的 worst-skill MSE 与累计 regret 配对区间都完全为负；
- 特征值可以更高，但没有上述主指标区间时不能标 PROMISING；
- 审计成本服从同一个 `b ± 0.01` 带子。

30 种子之前的格子一律写 PRELIMINARY。主结论只用 30 种子的预先指定格子。某一主比较的区间贴着 0，并且该格计算便宜时，只把该格扩到 50 个种子（种子 `0..49`）。否则不做。扩种子不能改假设。

## 4. 种子

种子是配对单位。同一个整数种子在一个格子的所有方法之间共享世界随机数。方法之间的审计随机数可以不同，但必须由该种子确定地导出，并且不依赖其他方法的结果。

| 层级 | 种子 | 用途 |
| --- | --- | --- |
| Tier 1 | `0, 1, 2` | 正确性通过之后的全核心方法烟测。标签 PRELIMINARY。 |
| Tier 2 主格子 | `0, 1, ..., 29` | 唯一的确认性样本。 |
| 10 种子稳健性 | `0, 1, ..., 9` | 只用于协议写明的预算、技能数和 Env3 格子。 |
| 便宜且区间贴 0 | `0, 1, ..., 49` | 只扩展那一个主比较格子。 |

不得在看到结果后重抽种子、丢掉种子或换成另一组 30 个数。不得跑满 mechanism × budget × method × 技能数的笛卡尔积。

## 5. 环境与指标定义

工作副本：`/Users/grantwright/Desktop/科研/10.30新课题/v0`。已记录的机器类来自 v1 manifest：Darwin 23.5.0、arm64、MacBook Air M3、16 GB。主解释器当时没有 torch。v1.1 开跑时另写一份 manifest，路径使用实际工作副本。不要把仍写着 `/Users/grantwright/Research/idea-A-B` 的未提交 manifest 当作 v1.1 环境记录。不安装 vLLM、bitsandbytes、flash-attn、CUDA、NCCL。不新下大模型。

**Idea A 静态世界（沿用 v1 模拟器）**

- `n = 2000`，`K = 8`，层概率与层效应沿用 `simulator.py`。
- 主 mechanism：MNAR、zero-overlap。其余 mechanism 只出现在 Tier 1。
- 真值是该种子的有限总体泛函，方法之间共享。
- 审计 `A ~ Bernoulli(q)`，`q` 已知，构造 `q` 时不读隐藏 `Y`。

**Idea A 闭环**

- 目录 `idea_A/benchmarks/Sequential_Closed_Loop/`。
- 每轮用已经观测的数据更新策略，下一轮自然观测概率依赖新策略。总审计预算固定。
- Static 与 Continual 分目录、分指标，不写成同一个实验。
- 两条确认性场景：policy-drift，MNAR-audit。

**Idea B**

- 主技能数 `d = 3`。`population_per_round = 40`，`horizon = 32`，`noise_std = 0.15`，`feature_noise = 0.05`，`mix = (0.42, 0.42, 0.16)`，`theta_star = (1.2, 1.05, 1.0)`，弱技能坐标 2，`serve_fraction = 0.35`。
- Env1：真闭环。初值放在多数技能已经不错的参数上，使弱技能误差还能变差。v1 从 0 起步的设定只保留为 B-v1.0 参照。
- Env2：观测支撑上分不开、未观测技能上分得开的两个潜变量。结构性不可见的坐标标成不可恢复。
- Env3：变化点是回合中点，0 起下标为 16（horizon 32）。突变由种子在任何方法运行之前写死，所有方法共用，不根据表现再改。可恢复一条，不可恢复一条。
- 恢复时间：变化点之后，worst-skill MSE 首次连续 4 个回合不高于变化前最后 8 个回合的中位数。到 horizon 仍未达到则右删失，数值记为 horizon。配对 bootstrap 使用这个右删失数值，并在报告中说明删失比例。若两个方法都有超过一半种子被删失，该恢复时间比较为 INCONCLUSIVE。
- 检测延迟：只有存在预先定义的检测器时才报告它与变化点的差。没有检测器时写“未定义”，不用恢复时间代替。
- 预先登记的一次修订：仅当 Tier 1 显示可恢复 Env3 上连 full-feedback 对照的恢复时间也被右删失时，允许在 Tier 2 之前把 Env3 的 horizon 加长一次，中点变化点按比例保持。新数字写进本文件末尾的修订节，然后锁定。不得按哪个方法恢复得更快来选择 horizon。

## 6. 统计计划

- 配对差的区间是百分位 bootstrap。重抽样单位是种子。`n_boot = 10000`，`alpha = 0.05`，bootstrap RNG 种子 `0`，分位数方法为 NumPy `linear`。实现沿用 `shared/statistical_tests/paired_bootstrap.py` 的配对方式；v1.1 对 RMSE 的泛函另算，见下一条。
- Idea A 的 RMSE 区间：对每次 bootstrap 重抽样重算 `sqrt(mean squared error)`，再取候选减基线。不得把 per-seed 绝对误差的均值差叫做 RMSE 差。v1 的 `paired_bootstrap_rmse` 就是绝对误差，v1.1 不沿用那个泛函。
- bias 区间：重抽样种子后重算平均误差之差。
- coverage：种子上的命中指示平均。同时报告 Wilson 95% 区间。coverage 是主报告量。缺少区间的方法不能凭 RMSE 标 PROMISING。
- 名义水平 95%。区间构造对一个比较中的所有方法使用同一公式族（影响函数或种子内 bootstrap 二选一，在 Tier 1 之前写进 v1.1 代码并冻结）。不得为了提高某个方法的 coverage 而在看过评价种子之后改公式。
- `n_seeds <= 3` 的区间标记 PRELIMINARY。
- 确认性家族就是第 8 节的主格子。其他格子是探索性，不另做族内校正后的成功声明。
- 右删失的恢复时间使用第 5 节的数值。不在看到结果后改连续 4 轮或“变化前 8 轮”这两个常数。

## 7. 超参数协议

评价种子上不允许调参。下面的数在 Tier 2 之前锁定。Tier 1 可以暴露实现错误；修复错误不等于在评价集上选择超参。

沿用 v1 实际跑过的值，而不是未被运行器读取的 yaml 值：

| 项 | 锁定值 | 来源 |
| --- | --- | --- |
| Idea A `n`、`K` | 2000、8 | `simulator.py` / `config.yaml` |
| Idea A ridge | `1e-3` | `fit_mu_hat` 默认 |
| A-v1.0 `exploration_mix`、`q_floor` | 0.15、`1e-6` | v1.0 `config.yaml`，仅用于不改行为的 v1.0 臂 |
| cross-fit | 2 折 | 本协议。静态数据折外预测 |
| Idea B horizon、population | 32、40 | `run_synthetic.py` 常量；Env3 加长只走第 5 节的一次修订 |
| Idea B `ridge_lambda`、`mix_uniform`、SGD 步长、lazy period | 0.5、0.1、0.15、5 | v1 落盘 run config |
| Idea B smoothing | 0.6 | 落盘 run config。`methods/v1.0/config.yaml` 里的 0.65 未被运行，v1.0 参照臂继续用 0.6 |
| 噪声底 | `0.0225` | `noise_std = 0.15` |

v1.1 新的标量（信息折扣、策略漂移阈值、v1.1b 的方差估计细节中尚未由公式固定的常数）在第一次 Tier 1 运行之前写入 `methods/v1.1/config.yaml`。写入之后不因 Tier 1 或 Tier 2 的指标高低而修改。公式本身（AIPW、`q_i ∝ 1/n_skill`、E/D 准则）按审计第 11 节实现，不在评价集上挑选。

真实数据不参与调参。MultiPref 50-prompt 子集只在合成主结果之后做一次静态识别可行性。RewardBench 仍只做 schema，不打榜。

## 8. 预先指定的实验矩阵

不跑满笛卡尔积。`b = 0.01` 与 `b = 0.20` 只留在 Tier 1 或 3 个种子的边界格，不进主结论。

### Idea A

Tier 1：种子 0、1、2。核心方法全部。MNAR 与 zero-overlap，`b = 0.05`。静态世界，加一条短闭环。先过测试再保留格子。

Tier 2，静态，种子 `0..29`：

| mechanism | budget | 方法 |
| --- | --- | --- |
| MNAR、zero-overlap | 0.05、0.10 | observed-only，Uniform IPW，Uniform cross-fit DR，Neyman+DR，v1.1b，v1.1c |

Tier 2，闭环，种子 `0..29`：

| 场景 | budget | 方法 |
| --- | --- | --- |
| policy-drift、MNAR-audit | 0.05、0.10 | Uniform DR，Neyman DR，v1.1c |

消融 A1–A8：只在主格子上各改一个组件。主格子指闭环、MNAR-audit、`b = 0.05`、种子 0、1、2。若某一项与主结论有关，再在该格的 30 个种子上补这一项，不展开整张消融表。

### Idea B

Tier 1：3 个技能，`b = 0.05`，种子 0、1、2，Env1、Env2、Env3（可恢复与不可恢复），全部对照。

Tier 2，种子 `0..29`：

| 环境 | 技能数 | budget | 方法 |
| --- | --- | --- | --- |
| Env1、Env2 | 3 | 0.05 | Uniform，EqualSkill，E-optimal，D-optimal，v1.0 controller，v1.1c，Greedy |

其余只作稳健性，种子 `0..9`：

| 设置 | 内容 |
| --- | --- |
| 预算 | `b = 0.10`，环境与方法同主格子，10 个种子 |
| 技能数 | 2 和 5，只在 Env1、`b = 0.05`，10 个种子 |
| Env3 | 10 个种子，比较恢复时间；可恢复与不可恢复都报 |

消融 B1–B9：只在 Env1、`b = 0.05`、种子 0、1、2 上各改一个组件。B6（关掉内生反馈）和 B4/B5（E/D 与全控制器）若主结论依赖它们，再扩到 30 个种子。

Greedy 若仍只审计已经被自然反馈覆盖的样本，报告必须写出观测集与 No feedback 重合，不得把这条曲线当成信息对照的胜利或失败。

## 9. 测试先行

主比较使用的格子，其关键测试必须先通过。测试至少覆盖：估计量定义、AIPW、`A = 0` 不使用 `Y`、审计纳入概率、零重叠、cross-fit 不泄漏未来、两种 nuisance 各自误设与同时误设、预算相等、EqualSkill 与 Uniform 在不均衡频率下不同、Gramian 半正定、固定策略不能标成内生坍缩、不可识别反例、结果序列化。关键测试失败的格子不得用来声称改进。

## 10. 明确不在本协议里做的事

- 不重跑 AMPLe（`0cad341`，partial）、HyPER（blocked）、Drago（无代码）、PP-bandit（`d5daeff`，complete 的 1 维 demo）、ncvxpp（`ccbeedd`，partial，无 MPI）。
- 不发明简化 HyPER，不把 PP-bandit 接到技能删失环境。
- 不把 MultiPref 或 RewardBench 的结果写进合成主结论。
- 不提交 `.venv`、`.isolated_baselines_venv`、`.cache` 或权重。

## 11. 修订节

（冻结时为空。仅允许第 5 节预先登记的 Env3 horizon 加长，或测试失败后的缺陷修复记录。修订须写日期、原因，以及“尚未看到 Tier 2”。看到 Tier 2 之后的改动不算协议内修订。）
