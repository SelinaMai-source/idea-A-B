# v1 到 v1.1 审计

审计日期：2026-10-31。工作副本：`/Users/grantwright/Desktop/科研/10.30新课题/v0`。父提交：`f98d53f59ac98164693eaf14b53d99a499f19c4a`（分支 `v1`，跟踪 `origin/v1`）。本文件只记录已在源码和已落盘 JSON 中核对过的事实，以及 v1.1 计划中尚未实现的修改。本阶段没有改 `methods/v1.0`，没有实现 v1.1 算法，没有重跑 Tier 1/2，也没有重跑官方基线。

数字来自：

- `idea_A/results/SMOKE_SUMMARY.json`（`seeds = [0, 1, 2]`，`n = 2000`，`preliminary = true`，`failures = []`）
- `idea_B/results/SMOKE_SUMMARY.json`（`horizon = 32`，`population_per_round = 40`，`n_runs = 435`，`skips = []`）
- 下文点名的 `.py` / `.yaml` / 基线 `REPRODUCTION_AUDIT.json`

主矩阵在 v1 里已经跑完：Idea A 的 `matrix.mode = full`，`n_runs_written = 84`，`n_cells_skipped = 0`，对应 7 个 mechanism × 4 个 budget × 种子 0/1/2。Idea B 的非 B2 额外预算也已写入（`skips` 为空）。这些数字是 3 个种子的初步结果。

## 1. 未提交的工作区 diff

`git status` 在审计开始时有三个已修改、未暂存文件：

- `reports/A_B_SMOKE_EXPERIMENT_REPORT.md`
- `reports/environment_manifest.json`
- `shared/hardware/environment_manifest.json`

`git diff` 只有三类改动，两份 manifest 的 hunk 相同：

| 字段 | 已提交值 | 工作区值 |
| --- | --- | --- |
| `recorded_at_utc` | `2026-10-30T16:37:01.069960+00:00` | `2026-10-30T16:39:25.084726+00:00` |
| `git.commit` / 报告里的 commit | `494fe90f8e08e40d5f2bb8a6020e25ce74646c20` | `f98d53f59ac98164693eaf14b53d99a499f19c4a` |
| `disk.free_gib` | `7.504169464111328` | `7.507244110107422` |
| `disk.free_bytes` | `8057540608` | `8060841984` |
| `disk.used_bytes` | `486327255040` | `486323953664` |

`disk.path` 仍是 `/Users/grantwright/Research/idea-A-B`。diff 没有把路径改成当前 Desktop 工作副本，也没有密钥、token 或凭证。这是 `f98d53f`（“把记录到的 commit 写回报告”）之后又做了一次环境探测：时间戳、剩余磁盘、commit 字段。它不是方法改动。这三份文件保持未暂存，不进入本阶段提交。

已提交的 smoke JSON 和 `reports/REAL_DATA_FEASIBILITY.json` 里的绝对路径同样指向 `/Users/grantwright/Research/idea-A-B`。那是历史记录，本阶段不改写。

## 2. 模型符号链接

`git status`（含 `git update-index --refresh`）没有出现 `Too many levels of symbolic links`。

`idea_A/models/Qwen2.5-0.5B-Instruct` 与 `idea_B/models/Qwen2.5-0.5B-Instruct` 是目录。权重与词表文件是相对符号链接，例如 `model.safetensors -> ../../../.cache/models/Qwen2.5-0.5B-Instruct/model.safetensors`。从该目录解析后落到仓库内 `.cache/models/Qwen2.5-0.5B-Instruct/`，`os.stat` 成功，没有 ELOOP。索引里这些链接的 mode 是 `120000`。`.gitignore` 用 `*.safetensors` 与 `**/.cache/` 忽略权重和缓存。

`idea_B/models/Qwen2.5-0.5B-Instruct/.cache/huggingface/` 里还有一组指向仓库 `.cache` 的相对链接，同样可以解析，并被 `**/.cache/` 忽略。

本阶段没有改链接，没有把权重量提交。Qwen 修订号在 `reports/REAL_DATA_FEASIBILITY.json` 中为 `7ae557604adf67be50417f59c2c2f167def9a775`；该文件记录的是 CPU `generate` 烟测（`device_actually_used = cpu`），MPS generate 被中止。

## 3. Idea A 实现（v1.0，已核对）

### 3.1 模拟器是一次性抽样

`idea_A/benchmarks/Synthetic_Selective_Feedback/simulator.py` 的 `simulate()` 画一个有限总体后返回 `World`。`pi` 在该函数内由 `pi_logit` 抽一次（约第 181–182 行），随后生成 `Y`，再生成自然观测 `S`。没有 `π → S → 用估计更新 π → 下一轮 S`。`policy_dependent` 只让初始 `pi` 进入 `_natural_propensity`，不是闭环。

`idea_A/experiments/protocol.py` 的主路径对每个世界调用一次分配和一次审计。`batched_neyman_q` 只服务消融 A4：它按时间批更新层内标准差，不更新 `pi`，也不更新自然倾向。

`Y` 在 `S` 之前生成。Uniform 分配是 `allocate_uniform`：`q_i = b`，不读 `S`。IPW / DR 的和只在 `A = 1` 上使用结果。因此同一种子、同一预算下，Uniform 估计量在各 mechanism 上应当相同。JSON 证实了这一点（见第 5 节）。

### 3.2 `dr_mean` 是标准 AIPW

`shared/metrics/estimators.py` 的 `dr_mean` 写成

```text
μ_DR = (1/n) Σ [ μ_i + (A_i / q_i) (Y_i - μ_i) ]
```

实现里先把贡献设成 `mu_hat`，只在 `A = 1` 的位置加上 `(Y - mu_hat) / q`。`A = 0` 的加数是 `mu_hat`，`Y` 不是这个加数的操作数。`idea_A/tests/test_no_oracle_leakage.py` 的 `test_ipw_snips_dr_do_not_read_unaudited_entries` 把 `A = 0` 的 `Y` 改成 `1e9`，并断言 `dr_mean` / `ipw_mean` / `snips_mean` 不变。本阶段没有重跑该测试；断言与代数一致。

`ipw_mean` 与 `snips_mean` 同样只在 `A = 1` 上使用 `Y`。`naive_ips_mean` 是另一条函数：它把调用者提供的自然倾向 `p_hat` 用在自然观测掩码上，不接收审计倾向 `q`。

### 3.3 `estimate_estimands` 复用同一个连续 `mu_hat`

错误在 `idea_A/methods/v1.0/estimators.py` 的 `estimate_estimands`（约第 134–137 行），不在 `dr_mean`。

`idea_A/experiments/protocol.py` 的 `_mu_for_dr` 调用 `fit_mu_hat`：在 `A = 1` 上对连续 `Y` 做 ridge（`ridge_lambda` 默认 `1e-3`），设计矩阵含截距、`X`、`pi` 和层哑变量。这个向量被原样传进 `estimate_estimands`。该函数用同一个 `mu_hat` 估计三件不同的事：

1. `population_mean_reward`：结果是 `Y`。这里的 `mu_hat` 与 `E[Y | X]` 同型。
2. `preference_rate`：结果是 `pref_visible`，即 `1{Y > 0}`。结果模型仍是连续 `Y` 的 ridge，没有单独的 logistic，也没有与数据生成过程匹配的 `Φ` 或 logit 变换。
3. `policy_value_target_distribution`：结果先被写成 `target_weights * Y`，残差里仍减去连续 `mu_hat`。`simulator.py` 里 `target_weights` 是已知的层密度比 `p*(stratum) / p(stratum)`，`p*` 为 `K = 8` 层上的均匀分布。与该 estimand 匹配的回归是 `w(x) μ_Y(x)`，残差应是 `w(Y - μ_Y)`。当前 DR 在 `A = 0` 时加入的是 `μ_Y`，在 `A = 1` 时加入的是 `(w Y - μ_Y) / q`。

总体均值这条 DR 通道的公式是对的。偏好率与策略价值这两条复用了错误的回归。

没有 cross-fitting。`fit_mu_hat` 与 DR 残差使用同一批 `A = 1` 样本。序贯、按历史切分的 nuisance 也不存在。

### 3.4 分配

`idea_A/methods/v1.0/algorithm.py`：

- Uniform：`q_i = b`。
- Neyman：`q_h ∝ σ_h`，再与常数 `b` 按 `exploration_mix = 0.15` 混合，并用 `fit_mean_q` 把 `Σ n_h q_h / n` 拉回预算。`σ_h` 来自当前可见标签。主实验经 `_finite_sigma_from_natural` 使用自然观测 `S`。MNAR 下 `S` 依赖未观测的 `Y`（`_natural_propensity` 为 `sigmoid(-0.55 + 1.25 Y)`，再截到 `[0.02, 0.95]`），所以这些 `σ_h` 不是总体层标准差。
- `fixed_stratified`：每层期望审计条数相等，`n_h q_h = b n / K`。这是等额分配，与 Neyman 不同。
- `config.yaml` 写明 `novelty_claim: false`。v1.0 是分层 Neyman 加探索下界，再加已知 `q` 的 IPW / Hájek / AIPW。

审计 `A` 由 `draw_audit` 按已知 `q` 独立抽取，构造 `q` 时不读隐藏的 `Y`。

### 3.5 v1 没有 estimand 的区间覆盖

仓库里 Idea A 的实验代码没有总体均值的置信区间，也没有 coverage。`shared/statistical_tests/paired_bootstrap.py` 只对种子配对差做百分位 bootstrap。因此 v1 报不出 CI coverage。这是缺指标，不是一份已经算好、只是很小的覆盖率。

## 4. Idea B 实现（v1.0，已核对）

### 4.1 Gramian

`idea_B/methods/v1.0/observability.py` 把信息矩阵定义为线性模型的观测设计 Gramian：

```text
W_T = Σ_t S_t φ_t φ_t^T
W_T^cond = Σ_t q_t φ_t φ_t^T
```

`observed_design_gramian` 计算 `Σ_i S_i φ_i φ_i^T`。这个定义在 v1.1 中保留。它不是动力系统里的抽象 observability Gramian。特征值是描述量。`algorithm.py` 的文件头写明：更好的信息单独记录，不代替总体表现。

### 4.2 `EqualSkillAllocation`

`feedback_controller.py` 约第 155–163 行：对批次里出现的每个技能赋同一个 `q = target`（`target` 即预算），再调用 `_clip_mean` 把均值缩回预算。

缺失技能在批次里没有行。出现的技能又拿到同一个 `target`，所以批次上的 `q` 已经是常数 `target`。`0 < target < 1` 时 `_clip_mean` 不再改它。因此该规则在每个批次上都与逐样本 Uniform（`q_i = b`）相同，包括某些技能没有出现的批次。计划里写的“技能都出现时与 Uniform 相同”是其中一种情形；源码比这更强，不均衡频率也分不开。

不存在 `q_i ∝ 1/n_skill(i)` 的等期望条数分配，也不存在单独的频率比例分配。

### 4.3 主表现 `1/(1+MSE)`

`algorithm.py` 的 `_performance` 是 `1 / (1 + max(MSE, 0))`。`noise_std = 0.15` 时不可约噪声底是 `0.0225`，对应的 performance 是 `1 / 1.0225 ≈ 0.977995`。regret 已按 `max(true_mse - noise_std^2, 0)` 累加，但主对比字段仍是 `late_true_performance`。

`run_closed_loop` 在 `theta0 is None` 时把 `theta` 设为 0。`B2` 的 `theta_star = (1.2, 1.05, 1.0)`，弱技能坐标是 2，真值 1.0。固定策略诊断的 `late_weak_skill_error` 均值为 1.0，与“参数停在 0”一致。从 0 起步时，早期误差没有继续变差的空间，联合坍缩判据很难触发。

### 4.4 环境

`idea_B/benchmarks/Synthetic_Closed_Loop/environments.py`：

- B1 `B1ExogenousMissing`：弱技能观测率按回合衰减，`natural_observation_prob` 不读 `theta`。`policy_affects_feedback = false`。
- B2 `B2PolicyInducedCollapse`：用当前 `θ` 的预测，取最高的 `serve_fraction = 0.35` 作为自然反馈（`logit_temperature` 默认 0，所以是硬 top-k）。参数由观测标签更新，再成为下一轮策略。这是真闭环。
- `B3Recoverable` 继承 B2，只改了名字、`collapse_type` 和描述，没有改动力学。它不是“回合中点的奖励或任务混合突变”。
- `B3Unrecoverable` 把弱技能坐标从 `φ` 里投影掉，审计也不能恢复该坐标。
- 没有检测延迟或恢复时间的定义。

实验运行器 `idea_B/experiments/run_synthetic.py` 把 `population_per_round` 设为 40（环境类默认 48 被覆盖），`horizon = 32`，`d = 3`。`METHODS` 含 Uniform、Greedy、Lazy、SGD、Fixed schedule、Uncertainty、No feedback、Oracle 和 ObservabilityController。没有名为 E-optimal 或 D-optimal 的方法。

控制器默认 `information_metric = lambda_min`。`_skill_scores_from_gramian` 把最弱特征向量或 `logdet` 代理打成分数，再与均匀混合（`mix_uniform = 0.1`）、平滑，然后缩放到预算。这是控制器内部的启发式。它不是“在预算约束下最大化 `λ_min`”或“最大化 `logdet`”的经典设计，不能当作 v1.1 的 E-optimal / D-optimal 对照。

### 4.5 两个会让对照失效的实现事实

Greedy 与 B2 的自然观测使用同一个分数 `φθ`，都取 top-k。`serve_fraction = 0.35`，烟测预算最大为 0.20，因此 Greedy 抽中的审计集落在已经被服务的集合里。`observed = natural ∨ audit` 与不审计相同。JSON 里 B2 的四个预算上，Greedy 与 `NoFeedbackController` 的 `late_true_performance`、`late_lambda_min`、`late_weak_skill_error`、`cumulative_regret` 均值逐项相等，只有 `mean_feedback_cost` 不同。

`FixedExplorationSchedule` 与 `UniformExploration` 都 `return np.full(n, target)`。B2、`b = 0.05` 上两者的晚期表现、`λ_min`、弱技能误差、regret 和成本均值相同。

### 4.6 配置没有被运行器读入

`idea_B/methods/v1.0/config.yaml` 写 `smoothing: 0.65`。`run_closed_loop` 的默认值是 `0.6`。`run_synthetic.py` 的 `main` 调用 `_load_yaml(DEFAULT_CONFIG)` 后把返回值丢掉。已落盘的 `idea_B/results/synthetic_B2_ObservabilityController_budget0p05_seed0/config.yaml` 记录 `smoothing: 0.6`。v1 实际跑的是 0.6。

## 5. 数值发现

### 5.1 Idea A，MNAR，`b = 0.05`，总体均值，3 个种子

`per_method[*].mnar["b=0.05"].population_mean_reward.rmse_across_seeds`：

| 方法 | `rmse_across_seeds` | 四舍五入到计划中的四位 |
| --- | --- | --- |
| `uniform_dr` | 0.030812864196970262 | 0.0308 |
| `uniform_ipw` | 0.1143845065322542 | 0.1144 |
| `neyman_ipw` | 0.18018378912160418 | 0.1802 |
| `neyman_dr` | 0.11410395154500332 | 0.1141 |
| `observed_only` | 1.1902011608832788 | 1.190 |

`observed_only` 的 bias 均值是 1.189695728152607（标准差 0.042477248910611165）。`key_findings` 里该 bias 在 `b = 0.01, 0.05, 0.10, 0.20` 上相同，因为这条臂的 `q = 0`，预算不进入估计。

`uniform_dr`、`uniform_ipw`、`uniform_snips` 的总体均值 RMSE 在全部 7 个 mechanism 上各只有一个唯一值（`b = 0.05` 下 `uniform_dr` 都是 0.0308128642）。`neyman_dr` 与 `neyman_ipw` 在 7 个 mechanism 上各有 7 个不同值。这与第 3.1 节的一次性生成一致：Uniform 不读 `S`，Neyman 的 `σ_h` 读 `S`。

同一格子上，偏好率 RMSE：`uniform_dr` 0.11193034740880668，`uniform_ipw` 0.046963638416687145。DR 在这个二元 estimand 上差于 IPW，与“用连续 `μ_Y` 当 `P(Y>0)` 的回归”一致。策略价值 RMSE：`uniform_dr` 0.07729010939368364，`uniform_ipw` 0.12418167085800168。这一格上 DR 的 RMSE 更小，所以策略价值的这一次比较本身显示不出复用 `mu_hat` 的代数错误；错误由第 3.3 节的公式确定。

`full_feedback_oracle` 在该格的总体均值 RMSE 为 0。真值管道在全反馈下对上了总体均值。

Neyman IPW 减 Uniform IPW 的配对差（`key_findings.neyman_vs_uniform_rmse`，种子 0/1/2）：

| 格子 | 平均差（Neyman − Uniform） | 95% 百分位区间 | 区间是否含 0 |
| --- | --- | --- | --- |
| MNAR `b=0.05` | 0.0421636652086439 | [−0.06854158561273307, 0.16174829508689662] | 含 0 |
| zero-overlap `b=0.05` | 0.08123445238730907 | [0.02568117843499962, 0.15685934050465664] | 不含 0，Neyman 更差 |
| MNAR `b=0.10` | 0.03898473521677975 | [−0.1416057703403097, 0.26792226660089813] | 含 0 |

`paired_bootstrap_rmse` 与 `paired_bootstrap_abs_bias` 在这三格上的均值和区间完全相同（bootstrap 种子分别是 0 和 1，`n_boot = 10000`）。原因见第 6 节：对单个标量估计，代码里的 per-seed RMSE 就是绝对误差。

MNAR `b = 0.05` 上 Neyman DR 的总体 RMSE（0.1141）高于 Uniform DR（0.0308）。v1 没有给出“Neyman 优于 Uniform DR”的证据。主对照应是正确的 Uniform cross-fit DR。

### 5.2 Idea B，B2，`b = 0.05`，3 个种子

`controller_vs_uniform`（预算 0.05，配对种子 0/1/2，`n_boot = 5000`）：

- `late_lambda_min` 配对差均值 1.2303219467145972，区间 [1.1027045249030571, 1.3078270982853857]，不含 0。控制器的 `λ_min` 更高。
- `late_true_performance` 配对差均值 0.0005526395784082316，区间 [−7.730945227613084e-05, 0.0011408386436214668]，含 0。
- JSON 字段 `beats_uniform_on_both = true`。它只表示两个均值差为正。performance 的区间含 0。`SMOKE_SUMMARY.json` 自己的 `note` 写着：更好的 `W_T` 不被当成更好的 reward。

B2、`b = 0.05` 的晚期真实 performance 均值：控制器 0.978371，Uniform 0.977818，Oracle 0.978610。它们贴着 `1/(1+0.0225) ≈ 0.978` 的噪声底。同一格的晚期弱技能绝对参数误差：控制器 0.0195，Uniform 0.0533，Oracle 0.0075，No feedback / Greedy 0.1734。累计 regret：控制器 0.0976，Uniform 0.1152，Greedy 与 No feedback 0.2466，Lazy 4.9525。

`endogenous_collapse` 与 `endogenous_collapse_joint_criterion` 都是 `false`。上表所列方法在 B2 `b = 0.05` 的 `endogenous_collapse_flags` 均为 `[false, false, false]`。B1 的 note 写明它是外生缺失诊断。固定策略、固定抽样的 3 次运行 `late_weak_skill_error` 均值为 1.0，note 要求不得标成内生坍缩。

B3 recoverable 上，控制器与 Uniform 的晚期 performance、`λ_min`、弱技能误差与 B2 `b = 0.05` 的对应均值一致（控制器 performance 0.9783709068747761，`λ_min` 1.762170680286225）。这与“`B3Recoverable` 不改变 B2 动力学”一致。

B3 unrecoverable 上，Oracle 的晚期弱技能误差仍为 0.23933681616427394，`λ_min` 为 0.08794583536507604，performance 约 0.838，与控制器、Uniform 同量级。全反馈没有识别被投影掉的坐标。

`idea_B/results/SMOKE_SUMMARY.json` 记录 `pytest`：`idea_B/tests` 收集 18、通过 18、失败 0。本阶段没有重跑。当前测试文件中的测试函数数量与 18 一致。

## 6. 已确认的缺陷

1. **偏好率与策略价值共用连续 `mu_hat`。** 位置：`estimate_estimands`。总体均值的 `dr_mean` 代数不受这处影响。v1.0 文件保持不动，修正只进 `idea_A/methods/v1.1/`。
2. **没有 cross-fit。** 同一 `A = 1` 样本既拟合 ridge，又进入 DR 残差。
3. **主模拟没有策略闭环。** `simulate()` 一次抽完全体和 `pi`。Uniform 估计量跨 mechanism 完全相同，是这个设计的结果。
4. **`EqualSkillAllocation` 与逐样本 Uniform 相同**，包括技能频率不均、且有技能未出现的批次。
5. **主表现使用 `1/(1+MSE)`。** B2 上各方法的晚期 performance 挤在噪声底附近，`λ_min` 的差距没有变成 performance 的配对区间证据。
6. **`B3Recoverable` 不是恢复实验。** 没有中点突变，没有恢复时间。
7. **Greedy 在 B2 的烟测预算下不增加观测。** 学习曲线与 No feedback 重合，成本不为 0。
8. **Fixed schedule 与 Uniform 是同一分配。**
9. **`smoothing` 的 yaml 值 0.65 没有进入运行。** 落盘配置是 0.6。
10. **控制器的 `lambda_min` / `logdet` 分数不是 E-optimal / D-optimal 对照。**

`dr_mean` 在 `A = 0` 时不把 `Y` 算进和式，这一点保持为正确实现，v1.1 应继续复用 `shared/metrics/estimators.py`。

## 7. 统计问题

- 所有主结论目前来自 3 个种子。`paired_bootstrap_ci` 在 `n_seeds <= 3` 时把 `preliminary` 设为 true，note 写明这不是确认性优越声明。3 个点的 bootstrap 支撑集大小是 `3^3 = 27`，`n_boot = 10000` 不会增加信息。
- `errors.py` 的 `rmse(估计, 真值)` 在估计是标量时等于绝对误差。`run_synthetic.py` 把 per-seed 的 `population_mean_reward_rmse` 送进名为 RMSE 的 bootstrap，再把 `|bias|` 送进另一个 bootstrap。二者是同一个均值绝对误差泛函。单元格上的 `rmse_across_seeds = sqrt(mean((estimate - truth)^2))` 才是 RMSE，而且没有被这个配对区间直接估计。MNAR `b = 0.05` 上，两个 RMSE 之差约为 0.1802 − 0.1144 = 0.0658，而配对绝对误差差的均值是 0.0422。
- 没有多重比较校正。mechanism × budget × estimand 是一张大表。
- 没有总体均值的区间，因而没有 coverage。
- Neyman 的 `σ_h` 来自 MNAR 自然样本。
- Idea B 的 `beats_uniform_on_both` 使用均值符号。performance 区间含 0。只提高特征值不能当成学习成功。
- 坍缩标签要求观测表现稳定、`λ_min` 变差、弱技能误差变差三者同时成立，并且要求策略影响反馈、参数在动、抽样在动。从 `theta = 0` 开始时弱技能误差容易下降，联合判据在烟测里没有一次为真。这不能被写成“已经排除了内生坍缩”；它说明判据和初值使该事件难以出现。

## 8. 未完成或不可当作主结论的实验

- 种子数停在 3。没有 30 种子的确认性格子。
- Idea A 消融 A1–A5 各 39 次运行。`ablation_skip_note`：全部 mechanism 在 `b = 0.05`，外加 MNAR 与 zero-overlap 的全部预算，种子 0/1/2。其余 budget × mechanism 被跳过。
- Idea B 消融合计 `total_runs = 33`，不是完整因子设计。
- 没有 2-fold cross-fit DR，没有 estimand 专用的结果模型，没有序贯闭环环境 `Sequential_Closed_Loop`。
- 没有等技能期望条数、频率比例分配、E-optimal、D-optimal，也没有 Env3 的中点突变与恢复时间。
- 真实数据停在可行性。`reports/REAL_DATA_FEASIBILITY.json`：MultiPref 修订 `12910233a0238a997ebe425656e9dfed7b0ff031`，子集 50 个 prompt id、99 条比较（train/val/test prompts = 35/8/7）。RewardBench v1 修订 `168d848cdbbea9764fae4a544dc9ca1e6cca4931`，schema 样本 50 条，`scoring.ran = false`，`leaderboard_comparison = not_run`。Qwen 只做了 CPU 生成烟测。没有 LoRA，没有把合成矩阵重跑。
- 官方基线见下一节。本阶段不重跑。

## 9. 基线状态（读审计 JSON，未重跑）

`idea_A/baselines/REPRODUCTION_AUDIT.json`：

| 基线 | 上游提交 | `local_status` |
| --- | --- | --- |
| AMPLe（ACL 2025） | `0cad341f6a9e77063ef451c27732bf5c1534c797` | `partial`。命令把 `n_rounds` 设为 3，`burnin` 仍为 50000，acquisition 仍为 volume。`--dims 0 1` 因数据集 HTTP 401 被省略。 |
| HyPER（ACL 2025） | `4d96b27b989828b0b69a430f68d25b6b36e9015a` | `blocked`。预声明需要 TPU v3-128、EasyLM、vLLM、Beaker、GCS 和未初始化的 submodule。没有简化版 HyPER。 |
| Drago et al.，ICML 2026（`Online_Compatible_Reward_ICML2026`，论文名 Online Compatible Reward Identification from Preference Feedback） | `null`，`official_code = false` | `blocked`。无官方代码。 |

`idea_B/baselines/REPRODUCTION_AUDIT.json`：

| 基线 | 上游提交 | `local_status` |
| --- | --- | --- |
| PP-bandit（ICML 2024） | `d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5` | `complete`。官方 1 维再参数化数值实验。`multi_run_PR_phi_final_mean = -0.2883591540722531`。与 Idea B 的技能删失不可比。适配到 B2 的端口在主报告中为 BLOCKED，本阶段不重做。 |
| ncvxpp / Performative SGD（NeurIPS 2024） | `ccbeeddfbf9237aa06dca3a1579529faabd8fe99` | `partial`。`mpiexec -np 10` 因没有 mpi4py、mpicc、mpiexec 而 blocked。单进程 stub（rank 0，size 1）不是 10 replica 复现。 |

这些结果都标了 NOT DIRECTLY COMPARABLE 或同等限制。v1.1 不把它们放进 Idea A/B 的主榜，不发明简化 HyPER，不把 1 维 PP-bandit 接到技能删失环境。

## 10. 可复用与必须替换

可复用，v1.1 直接调用或保留行为：

- `shared/metrics/estimators.py` 的 `ipw_mean`、`snips_mean`、`dr_mean`、`observed_only_mean`。`dr_mean` 继续作为“审计缺失、`q` 已知、`A ⊥ Y | X, history`”下的 AIPW。
- `simulator.py` 的有限总体、mechanism、`World.observe`（只在 `S=1` 或 `A=1` 时露出 `Y`）和由全样本 `Y` 计算的真值。
- `intervention.draw_audit` 与预算记账。
- 种子配对：同一 `seed` 上的方法比较。
- Idea B 的 B1（外生缺失）、B2（策略决定谁被服务）、B3 unrecoverable（结构上不可见的坐标）、固定策略诊断，以及 Gramian 定义 `W = Σ S φφ^T`。
- `classify_collapse` 里“固定策略且固定抽样不是内生坍缩”的规则。
- 已有测试所锁定的不变量：不读未审计标签、全反馈恢复总体均值、预算均值、Gramian 半正定结构、B2 依赖策略、B1 不依赖策略。
- 基线审计 JSON 与 MultiPref 50-prompt 子集。RewardBench 仍只做 schema。Qwen 权重留在 `.cache`。

必须换掉或只能作为 v1.0 参照、不能当成 v1.1 方法：

- `estimate_estimands` 的共享 `mu_hat`。v1.1 为 `E[Y]`、`P(Y>0)`、`E[w(X)Y]` 使用不同的 nuisance。
- 用同一批审计样本拟合又估计的 DR。静态数据改为 2 折 cross-fit。
- 把一次性 `simulate()` 写成闭环。新环境放在 `idea_A/benchmarks/Sequential_Closed_Loop/`。
- `EqualSkillAllocation` 的“每个出现技能赋同一个 `q`”。v1.1 的等技能规则是 `q_i ∝ 1/n_skill(i)`，并受 `[0,1]` 与总预算约束；另设频率比例分配。三者在频率不均时必须能被测试分开。
- 把 `1/(1+MSE)` 当主指标。它降为次要描述。主指标改为 worst-skill MSE、超出 `noise_std^2` 的 excess MSE、累计 regret、恢复时间。
- 把 `B3Recoverable` 当成 Env3。Env3 是预先写死的中点突变。
- 把控制器内部的 `lambda_min` / `logdet` 分数当成 E-optimal / D-optimal。
- 把 Greedy 在 B2 上的曲线当成“花了预算的信息对照”，除非新环境使审计集不再被自然服务集包含。
- 任何只优于 Uniform 随机或 Uniform IPW 的声明。

`idea_A/methods/v1.0/` 与 `idea_B/methods/v1.0/` 的行为保持不变。修复、新估计量、新分配和新环境只进 `methods/v1.1/` 与新的 benchmark 目录。

## 11. 计划中的精确修改（尚未实现）

以下是 v1.1 的实现规格，供后续代理使用。本文件提交时这些路径还不存在。

Idea A，新代码在 `idea_A/methods/v1.1/`：

- `μ_Y(x) = E[Y|X]`：审计样本上的 ridge。
- `μ_P(x) = P(Y>0|X)`：单独的 logistic。禁止把连续 `μ_Y` 直接当概率，除非另写并测试与数据生成过程匹配的 `Φ` 或 logit 变换。
- `μ_V(x) = E[w(X)Y|X]`：`w` 是已知层密度比时使用 `w(x) μ_Y(x)`，不把 `μ_Y` 放进加权结果的 DR 残差。
- 静态数据：2 折 cross-fit，折外预测，不用未观测标签。序贯数据：用过去轮拟合、估计当前轮。普通随机 CV 在自适应抽样下会泄漏未来，协议里要写明。
- 已知审计 `q` 才是 MNAR 下的识别通道。`P(O=1) = 1-(1-p_S)(1-q)` 只在 `p_S` 不依赖未观测 `Y` 时使用。有审计不等于把 MNAR 变成 MAR。DR 的双重稳健性只针对“审计缺失、`q` 已知、`A ⊥ Y | X, history`”。
- A-v1.0：原 Neyman+IPW，行为不变。
- A-v1.1a：均匀审计 + cross-fit DR。这是必须比较的基线。
- A-v1.1b：只用审计残差估计层方差，按 AIPW 方差贡献分配 `q_h`，使 `E[q] = b`。方差公式写入 README，并说明它是分层 HT/AIPW 方差的可计算近似。
- A-v1.1c：在 v1.1b 上对 `π_t` 相对 `π_{t-1}` 变化大的层提高审计。全提案是 v1.1c。若 v1.1b 在主指标上不差，报告写明漂移项没有额外价值。
- 新环境每轮用已观测数据更新策略，下一轮自然观测概率依赖新策略，总审计预算固定。Static Estimation 与 Continual Adaptation 分目录、分指标。

Idea B，新代码在 `idea_B/methods/v1.1/`：

- 修正后的 EqualSkill（等技能期望条数）、频率比例、逐样本 Uniform 三者分开。
- E-optimal（抬 `λ_min`）和 D-optimal（`logdet` / 预测方差）作为经典对照，README 写明不是新算法。
- B-Env1：真闭环，初值放在多数技能已经不错的参数上，使弱技能误差有变差空间。观测损失可以下降，同时弱技能 MSE 上升。报告噪声底 `noise_std^2`。
- B-Env2：两个潜变量在当前策略的观测支撑上观测分布相同或接近，在未观测技能上不同。无干预时参数误差下界为正；允许的审计能分开则识别恢复；结构性看不到的坐标标成不可恢复。用显式似然或矩条件。
- B-Env3：回合中点的奖励或任务混合按种子预先写死，不根据算法表现再改。可恢复与不可恢复各一条。指标含检测延迟、恢复时间、excess MSE、worst-skill、regret、成本。
- B-v1.0 原样；B-v1.1a 为修正后的等技能期望条数；B-v1.1b 为 E-optimal；B-v1.1c 在信息分配上加信息折扣和“预期能降低 worst-skill MSE”的一步收益。主提案是 v1.1c。若它不优于 E-optimal，结论是 NOT PROMISING 或 INCONCLUSIVE。E-optimal 的胜利不算成新方法。

评价设计、种子和成功规则写在 `reports/V1_1_EVALUATION_PROTOCOL.md`，并且在任何主实验之前冻结。
