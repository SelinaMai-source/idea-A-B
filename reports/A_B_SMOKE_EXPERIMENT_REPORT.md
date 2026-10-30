# Idea A / Idea B 可行性烟测报告 v1

本文件由 `scripts/generate_report.py` 从磁盘 JSON/CSV 生成。报告中的数字均来自下列产物，没有手填。

- 生成时间（UTC）：`2026-10-30T16:37:01.143137+00:00`
- 仓库：`/Users/grantwright/Research/idea-A-B`
- 分支：`v1`
- commit：`494fe90f8e08e40d5f2bb8a6020e25ce74646c20`
- remote：`git@github.com:SelinaMai-source/idea-A-B.git`
- 缺失的主 JSON：无（所列主源均在磁盘上）
- `n_seeds=3`：n_seeds=3 只是初步证据，不是 confirmatory 结论。

**判定规则预先固定，不因结果好坏改口。** Idea A 只有在 MNAR 或零重叠下审计 IPW 相对 observed-only / naive IPS 明显降偏，**并且**同等预算下 Neyman 相对 Uniform 的 RMSE 改进被配对区间支持时，才标 PROMISING。Idea B 只有在 B2 中观测表现稳定而 λ_min 与弱技能误差同时恶化，**并且**控制器在信息与长期真实表现上均以配对区间击败同预算 Uniform 时，才标 PROMISING。

Idea A v1.0 的分配规则是经典分层 Neyman 分配加探索下界，估计用已知审计倾向的 IPW / Hájek / AIPW。**这不是新算法。**

## 1. Executive Summary

- **Idea A 状态：`INCONCLUSIVE`**。名称：Adaptive Identification under Endogenous Feedback Censoring。
- **Idea B 状态：`INCONCLUSIVE`**。名称：Feedback-Induced Observability Collapse。
- Idea A 是否出现假设效应（内生删失导致偏差、审计通道可识别）：**部分支持**。MNAR `b=0.05` 下 observed-only 总体均值 bias mean = `1.1897`，审计 Neyman IPW bias mean = `-0.098014`，Uniform IPW bias mean = `-0.104575`，naive IPS bias mean = `0.590411`。识别-via-audit 成立。
- Idea A 是否在同等预算下击败 Uniform Audit：**不支持**。MNAR `b=0.05` Neyman−Uniform RMSE 差 mean = `0.0421637`，配对 CI = [`-0.0685416`, `0.161748`]，包含 0。zero-overlap `b=0.05` 的配对 CI 排除 0，但方向是 Neyman **更差**。
- Idea B 假设的内生可观性坍缩（观测表现稳定 **且** λ_min 与弱技能误差同时恶化）：**未满足联合判据**。`B2.endogenous_collapse=false`；`observed_performance_stable_while_lambda_min_and_weak_skill_error_worsen=false`；`any_seed_weak_skill_error_worsened=false`。
- Idea B 控制器相对同预算 Uniform：信息量 λ_min 的配对 CI = [`1.1027`, `1.30783`]（排除 0，控制器更高）；长期真实表现配对 CI = [`-7.73095e-05`, `0.00114084`]（**包含 0**）。JSON 字段 `beats_uniform_on_both=true` 只反映均值符号，**不能**覆盖配对区间规则。
- 是否成功复现并在同一协议上击败 CCF-A 论文：**否，且不得声称**。AMPLe local_status=`partial`；HyPER=`blocked`；PP-bandit official=`complete` 但是 1-D 数值 demo，NOT DIRECTLY COMPARABLE；ncvxpp=`partial`。
- 关键限制：MacBook Air M3 / 16GB / 无 NVIDIA GPU；`n_seeds=3`；主 venv 无 torch；HyPER / Drago 无官方可运行代码；PP-bandit 适配到 B2 被标 BLOCKED。
- 初步建议：两个想法目前都是 **`INCONCLUSIVE`**。Idea A 的可跟进方向是「已知审计通道在 MNAR 下的识别」，而不是把 Neyman 分配说成新算法。Idea B 需要能同时恶化弱技能误差的闭环设定，以及真实表现上排除 0 的配对区间。

## 2. Hardware and Software Environment

环境清单来自本次运行的 `shared.hardware.probe.write_environment_manifests`，以及基线审计 JSON 中的 machine 字段。probe 里的 `experiments_run` 仍为脚手架布尔值（测试锁定为 false）；实验是否跑过以 SMOKE_SUMMARY 为准。

| 字段 | 值 |
| --- | --- |
| OS | macOS-14.5-arm64-arm-64bit / Darwin 23.5.0 |
| arch | arm64 |
| 审计记录的机型 | MacBook Air M3 16GB |
| RAM GiB | 16 |
| 磁盘 free GiB（probe） | 7.50417 |
| 审计 disk_free_after_experiments | 9.7Gi |
| Python | 3.12.4 (v3.12.4:8e8a4baf65, Jun  6 2024, 17:33:18) [Clang 13.0.0 (clang-1300.0.29.30)] |
| 主解释器 | /Users/grantwright/Research/idea-A-B/.venv/bin/python |
| 主 venv torch | installed=False; import_error=ModuleNotFoundError: No module named 'torch' |
| 隔离 venv | /Users/grantwright/Research/idea-A-B/.isolated_baselines_venv/bin/python |
| numpy | 2.5.4 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| pandas | 3.1.0 |
| matplotlib | 3.11.2 |
| pytest | 9.2.0 |
| transformers（主 venv） | None |
| 隔离 venv transformers | 5.22.1 |
| 隔离 venv torch | 2.15.0 |
| Qwen revision | 7ae557604adf67be50417f59c2c2f167def9a775 |
| MultiPref revision | 12910233a0238a997ebe425656e9dfed7b0ff031 |
| RewardBench revision | 168d848cdbbea9764fae4a544dc9ca1e6cca4931 |
| git branch（probe） | v1 |
| git commit（probe） | 494fe90f8e08e40d5f2bb8a6020e25ce74646c20 |
| 不安装 | vLLM / bitsandbytes / flash-attn / CUDA / NCCL |

- 清单路径：`reports/environment_manifest.json` 与 `shared/hardware/environment_manifest.json`
- Qwen 推理设备：preferred=`mps`，实际=`cpu`，generate_ok=`true`，latency_s=`0.261256`，peak ru_maxrss_bytes=`3424256000`
- 合成矩阵**不是** LLM 后训练。Qwen smoke 只做 load/generate。

## 3. Literature and Baseline Audit

状态标签区分：published result / faithful local reproduction / modified（partial） / common-protocol / blocked。官方数字不得进入 Idea A/B 排行榜。

| Paper | Venue | Official source | Official code | Upstream commit | Local status | Local path | Blocker / modification | Local reproduced numbers | Comparability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Comparison-based Active Preference Learning for Multi-dimensional Personalization (AMPLe) | ACL 2025 Main (Long) | https://github.com/ml-postech/AMPLe | true | 0cad341f6a9e77063ef451c27732bf5c1534c797 | partial | idea_A/baselines/AMPLe/official_reproduction/ | Full README run is n_rounds=100 with burnin=50000; only n_rounds was shortened to 3. README --dims 0 1 would request minhyeonoh/feedback_assistant_harmlesshelpful_271 (HTTP 401 / n | AMPLe L2 t0=0.468885, t3=0.364745 | NOT DIRECTLY COMPARABLE to Idea A Neyman-audit IPW. AMPLe optimizes multi-dimensional preference identification via volume/random acquisition on RM-scored pairs. Different target, budget, and information. |
| Hybrid Preferences: Learning to Route Instances for Human vs. AI Feedback (HyPER) | ACL 2025 Main (Long) | https://github.com/allenai/hybrid-preferences | true | 4d96b27b989828b0b69a430f68d25b6b36e9015a | blocked | idea_A/baselines/HyPER/REPRODUCTION_STATUS.md | Pre-declared. Official pipeline needs Google Cloud TPU v3-128, EasyLM RM/DPO training of Tulu-2 13B / Llama-3.1-8B, vLLM (>=0.4.1; forbidden), Beaker, GCS, and submodules open-inst | 见审计 JSON | NOT DIRECTLY COMPARABLE. HyPER routes human vs LM annotations to maximize downstream RM / RewardBench scores. |
| Online Compatible Reward Identification from Preference Feedback | ICML 2026 | https://proceedings.mlr.press/v306/drago26a.html | false | NA | blocked | idea_A/baselines/Online_Compatible_Reward_ICML2026/THEORY_COMPARATOR.md | No official code found. PMLR/ICML poster/author PDF have no GitHub badge. github.com/simone-drago has only a github.io page. github.com/marcomussi/SleepingRL is a different ICML 20 | 见审计 JSON | NOT DIRECTLY COMPARABLE. Theory-only contrast: globally compatible reward identification vs Idea A finite-budget audit estimation. |
| Performative Prediction with Bandit Feedback: Learning through Reparameterization | ICML 2024 | https://github.com/UCSC-REAL/PP-bandit-feedback | true | d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5 | complete | idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/ | none | PP-bandit PR_phi mean=-0.288359 ± 0.0071396 | Official numerical experiment is a 1-D reparameterized zeroth-order performative-risk demo (Appendix H / Figures 4-6). Comparable only as a performative-prediction algorithmic primitive, not as Idea B observability-controller numbers. License: MIT (REAL@UCSC). |
| Stochastic Optimization Schemes for Performative Prediction with Nonconvex Loss | NeurIPS 2024 | https://github.com/Qiang-CU/nips2024-ncvxpp | true | ccbeeddfbf9237aa06dca3a1579529faabd8fe99 | partial | idea_B/baselines/Performative_SGD_NeurIPS2024/official_reproduction/ | Official full run `cd synthetic-iter && mpiexec -np 10 python main.py` is BLOCKED: mpi4py missing, mpicc/mpiexec not on PATH. A single-process subset of the original synthetic-iter | eps0-batch5-res0.npy loss_last=0.328135 test_acc_last=1 eps0.1-batch5-res0.npy loss_last=0.3458 test_acc_last=1 eps10-batch5-res0.npy loss_last=0.355371 test_acc_last=1 eps100-batch5-res0.npy loss_last=0.945008 test_acc_last=0 | NOT a full paper reproduction. Synthetic-iter SGD-GD is a performative linear-model experiment; not the same target as Idea B observability Gramian / closed-loop collapse. |

逐项技术差异与协议标签（来自 `reports/COMPARABILITY_MATRIX.json`）：

| method_id | paper/source | vs | comparability | protocol_label | local_status | leaderboard |
| --- | --- | --- | --- | --- | --- | --- |
| AMPLe_official | ACL 2025 Main (Long) | Idea A | NOT DIRECTLY COMPARABLE | partial | partial | official_reproductions_not_on_idea_leaderboards |
| volume_adapted_ipw | Mathematical analogue of AMPLe volume / max-variance, implemented on Idea A's si | Idea A | partial | adapted_to_common_protocol | complete_on_requested_slice | idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap |
| HyPER | ACL 2025 Main (Long) | Idea A | NOT DIRECTLY COMPARABLE | blocked | blocked | null |
| Drago_ICML2026 | ICML 2026 | Idea A | NOT DIRECTLY COMPARABLE | blocked | blocked | null |
| uniform_ipw | Generic statistical baseline (uniform audit + known-q IPW) | Idea A | Direct | generic_control | complete_on_requested_slice | idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap |
| neyman_ipw | Classical stratified Neyman allocation + exploration floor + known-q IPW. Textbo | Idea A | Direct | generic_control | complete_on_requested_slice | idea_A_synthetic_selective_feedback_b0.05_mnar_zero_overlap |
| IPS_SNIPS_DR_Uncertainty_observed_only_naive_IPS | Generic statistical estimators / allocations in Idea A smoke | Idea A | Direct | generic_control | complete_in_smoke | idea_A_synthetic_smoke_full_matrix (separate from AMPLe/HyPER) |
| PP-bandit_official | ICML 2024 | Idea B | NOT DIRECTLY COMPARABLE | official_reproduction | complete | official_reproductions_not_on_idea_leaderboards |
| pp_bandit_zeroth_order_reparam_adapted | Attempted port of PP-bandit two-point ZO + y^2 reparameterization | Idea B | NOT DIRECTLY COMPARABLE | adapted-skipped | BLOCKED | null |
| ncvxpp_Performative_SGD | NeurIPS 2024 | Idea B | NOT DIRECTLY COMPARABLE | partial | partial | official_reproductions_not_on_idea_leaderboards |
| ObservabilityController | Idea B v1.0 method (observed-design Gramian allocation) | Idea B | Direct | idea_B_v1.0 | complete_on_requested_slice | idea_B_B2_budget0.05_common_protocol |
| UniformExploration | Generic control (uniform audit probability = budget) | Idea B | Direct | generic_control | complete_on_requested_slice | idea_B_B2_budget0.05_common_protocol |
| GreedyDeployment | Generic control (audit the highest predicted scores) | Idea B | Direct | generic_control | complete_on_requested_slice | idea_B_B2_budget0.05_common_protocol |
| LazyDeployment_StandardOnlineSGD | Generic controls in Idea B smoke | Idea B | Direct | generic_control | complete_in_smoke | idea_B_synthetic_smoke (separate from official PP/SGD tables) |

- AMPLe 优化的是多维 preference 向量的 volume/random acquisition，不是有限总体泛函在内生删失下的审计识别。
- HyPER 优化的是 human vs AI 标注路由后的奖励模型 / RewardBench 分数。本机 TPU/vLLM/13B 训练被预先标 blocked，没有发明简化版 HyPER。
- Drago et al., ICML 2026 无官方代码，只作理论对照。
- PP-bandit 官方数值实验是 1-D reparameterized zeroth-order performative-risk demo，不是 B2 的技能删失。
- ncvxpp 官方 `mpiexec -np 10` 被 mpi4py/mpicc 阻断；单进程 stub 是 partial，不得冒充 10 replica 全文复现，也不得把 Greedy/Lazy 标成官方 SGD-GD。
- 共同协议上的 `volume_adapted_ipw` 是数学类比，不是 AMPLe official。

## 4. Idea A — Formal Problem and Method

实现位置：`idea_A/methods/v1.0/{algorithm.py,estimators.py,identification.py,intervention.py}`，配置 `idea_A/methods/v1.0/config.yaml`。

v1.0 allocation is classical stratified Neyman allocation plus an exploration floor, then IPW/Hájek/AIPW with known audit propensities. This is not a novel algorithm.

有限总体 `i=1..n`，`n=2000`，离散层 `h∈{1..8}`。事实结果 `Y_i` 在模拟器中生成，但算法只在 `O_i = S_i ∨ A_i` 时可读。

- 自然观测 `S_i` 由机制生成：MCAR / MAR / MNAR / policy-dependent / low-overlap / zero-overlap / nonstationary。MNAR 与零重叠下自然倾向不可识别，记为 identification failure，不是估计器报错。
- 干预：独立审计 `A_i ~ Bern(q_i)`，`q_i` 已知并写入日志。预算约束 `E[A] = b`，`b∈{0.01,0.05,0.10,0.20}`。
- 目标量：有限总体均值 `E[Y]`、偏好率 `P(Y>0)`、层条件均值 `E[Y|h]`、在均匀层分布 `p*` 下的策略价值。
- 估计器：observed-only、审计 IPW（Horvitz–Thompson）、SNIPS/Hájek、AIPW/DR（只在 `A=1` 读残差）、naive IPS（把自然倾向当已知，这是错误对照）。
- 分配：对可见标签估计层标准差 `σ_h`，Neyman `q_h ∝ σ_h`，再与均匀预算做 `exploration_mix=0.15` 混合，并用 `q_floor=1e-6`。**经典两阶段抽样。**

```
for each simulated world (mechanism, seed):
    draw X, stratum, Y  # hidden from the method
    draw natural S according to the mechanism
    estimate σ_h from currently visible labels only
    set q_h = Neyman(σ_h) mixed with uniform budget b
    draw A_i ~ Bern(q_i); O = S OR A
    estimate functionals with IPW/SNIPS/AIPW using known q
    compare every method on this same world (paired)
```

已知理论限制：零审计重叠且自然通道也看不见时，该子总体不可识别；倾向误指定会破坏 IPW；Neyman 用的 `σ_h` 来自可见标签，在 MNAR 下本身偏倚，所以分配改进不是定理保证。

## 5. Idea A — Experiments

- 数据集：合成 Selective Feedback。`n=2000`，seeds=`[0, 1, 2]`，`n_seeds=3`，preliminary=`true`。
- 矩阵：`Full matrix executed: all mechanisms × all budgets × seeds 0,1,2.`；写入 run 数 `84`，跳过 `0`。
- 划分：有限总体一次生成，无 train/val/test 泄漏式划分；配对方法共享同一 `Y`。
- 超参数：见 `idea_A/methods/v1.0/config.yaml`（`exploration_mix=0.15`，`ridge_lambda=0.001`，`n_batches_adaptive_budget=4` 仅用于 A4）。
- 命令：`PYTHONPATH=. .venv/bin/python -m idea_A.experiments.run_synthetic`
- 失败 run：`[]`

### 5.1 MNAR / zero-overlap，`b=0.05`，全部 method

**MNAR, b=0.05**

| method | pop-mean bias mean±std | pop-mean RMSE | preference-rate bias | realized audit |
| --- | --- | --- | --- | --- |
| fixed_stratified_ipw | -0.0770197 ± 0.177816 (n=3) | 0.164351 | -0.0336467 ± 0.0587338 (n=3) | 0.0508333 ± 0.00425245 (n=3) |
| full_feedback_oracle | 0 ± 0 (n=3) | 0 | 0 ± 0 (n=3) | 1 ± 0 (n=3) |
| naive_ips | 0.590411 ± 0.0334496 (n=3) | 0.591042 | 0.193565 ± 0.00324629 (n=3) | 0 ± 0 (n=3) |
| neyman_dr | -0.0756592 ± 0.104609 (n=3) | 0.114104 | 0.00271858 ± 0.128148 (n=3) | 0.048 ± 0.0025 (n=3) |
| neyman_ipw | -0.098014 ± 0.185173 (n=3) | 0.180184 | -0.0196362 ± 0.07984 (n=3) | 0.048 ± 0.0025 (n=3) |
| neyman_snips | -0.0935301 ± 0.181313 (n=3) | 0.175112 | -0.0125911 ± 0.0558388 (n=3) | 0.048 ± 0.0025 (n=3) |
| observed_only | 1.1897 ± 0.0424772 (n=3) | 1.1902 | 0.306905 ± 0.0125525 (n=3) | 0 ± 0 (n=3) |
| uncertainty_ipw | 0.264973 ± 0.0812032 (n=3) | 0.273143 | 0.10453 ± 0.0437639 (n=3) | 0.0585 ± 0.0047697 (n=3) |
| uniform_dr | -0.011897 ± 0.0348115 (n=3) | 0.0308129 | 0.0765113 ± 0.100058 (n=3) | 0.0481667 ± 0.00332916 (n=3) |
| uniform_ipw | -0.104575 ± 0.0567611 (n=3) | 0.114385 | -0.0161667 ± 0.0540031 (n=3) | 0.0481667 ± 0.00332916 (n=3) |
| uniform_snips | -0.0923965 ± 0.0687947 (n=3) | 0.108131 | 0.00208048 ± 0.0298622 (n=3) | 0.0481667 ± 0.00332916 (n=3) |

**zero-overlap, b=0.05**

| method | pop-mean bias mean±std | pop-mean RMSE | preference-rate bias | realized audit |
| --- | --- | --- | --- | --- |
| fixed_stratified_ipw | -0.0770197 ± 0.177816 (n=3) | 0.164351 | -0.0336467 ± 0.0587338 (n=3) | 0.0508333 ± 0.00425245 (n=3) |
| full_feedback_oracle | 0 ± 0 (n=3) | 0 | 0 ± 0 (n=3) | 1 ± 0 (n=3) |
| naive_ips | -0.354369 ± 0.0303507 (n=3) | 0.355234 | -0.134659 ± 0.0113715 (n=3) | 0 ± 0 (n=3) |
| neyman_dr | -0.0497932 ± 0.184065 (n=3) | 0.158323 | -0.0124966 ± 0.108506 (n=3) | 0.0493333 ± 0.00225462 (n=3) |
| neyman_ipw | -0.0506152 ± 0.24657 (n=3) | 0.207589 | -0.0133187 ± 0.091616 (n=3) | 0.0493333 ± 0.00225462 (n=3) |
| neyman_snips | -0.0496979 ± 0.232866 (n=3) | 0.196522 | -0.00614917 ± 0.0626505 (n=3) | 0.0493333 ± 0.00225462 (n=3) |
| observed_only | -0.359015 ± 0.0296244 (n=3) | 0.359829 | -0.0592557 ± 0.0123259 (n=3) | 0 ± 0 (n=3) |
| uncertainty_ipw | 0.0116305 ± 0.135955 (n=3) | 0.111614 | 0.0426884 ± 0.073399 (n=3) | 0.0518333 ± 0.00125831 (n=3) |
| uniform_dr | -0.011897 ± 0.0348115 (n=3) | 0.0308129 | 0.0765113 ± 0.100058 (n=3) | 0.0481667 ± 0.00332916 (n=3) |
| uniform_ipw | -0.104575 ± 0.0567611 (n=3) | 0.114385 | -0.0161667 ± 0.0540031 (n=3) | 0.0481667 ± 0.00332916 (n=3) |
| uniform_snips | -0.0923965 ± 0.0687947 (n=3) | 0.108131 | 0.00208048 ± 0.0298622 (n=3) | 0.0481667 ± 0.00332916 (n=3) |

### 5.2 识别对照（key_findings.mnar_observed_only_vs_audit_ipw）

| budget | observed-only bias | std | Neyman IPW bias | std | Uniform IPW bias | std |
| --- | --- | --- | --- | --- | --- | --- |
| b=0.01 | 1.1897 | 0.0424772 | -0.156616 | 0.477698 | -0.0980561 | 0.189442 |
| b=0.05 | 1.1897 | 0.0424772 | -0.098014 | 0.185173 | -0.104575 | 0.0567611 |
| b=0.10 | 1.1897 | 0.0424772 | -0.0779234 | 0.201944 | -0.0554651 | 0.144993 |
| b=0.20 | 1.1897 | 0.0424772 | 0.0159943 | 0.0923098 | -0.0542255 | 0.0835291 |

### 5.3 Neyman vs Uniform 配对 RMSE（全部 mechanism × budget）

| mechanism | budget | mean RMSE(Neyman−Uniform) | CI low | CI high | CI excludes 0 | paired reading |
| --- | --- | --- | --- | --- | --- | --- |
| low_overlap | b=0.01 | 0.278138 | -0.124599 | 0.607848 | no | CI includes 0 / inconclusive |
| low_overlap | b=0.05 | 0.0738034 | -0.00904951 | 0.194151 | no | CI includes 0 / inconclusive |
| low_overlap | b=0.10 | 0.0723726 | -0.0143346 | 0.183105 | no | CI includes 0 / inconclusive |
| low_overlap | b=0.20 | 0.0201694 | -0.00641864 | 0.0338664 | no | CI includes 0 / inconclusive |
| mar | b=0.01 | 0.279912 | 0.11372 | 0.509448 | yes | Neyman worse |
| mar | b=0.05 | 0.0512984 | -0.0332684 | 0.114045 | no | CI includes 0 / inconclusive |
| mar | b=0.10 | 0.0317555 | -0.144718 | 0.223938 | no | CI includes 0 / inconclusive |
| mar | b=0.20 | 0.00515489 | -0.0623739 | 0.0651642 | no | CI includes 0 / inconclusive |
| mcar | b=0.01 | 0.175047 | -0.216403 | 0.513711 | no | CI includes 0 / inconclusive |
| mcar | b=0.05 | 0.0220192 | -0.110555 | 0.123417 | no | CI includes 0 / inconclusive |
| mcar | b=0.10 | 0.0129912 | -0.150875 | 0.204533 | no | CI includes 0 / inconclusive |
| mcar | b=0.20 | 0.0115096 | -0.0834833 | 0.0904636 | no | CI includes 0 / inconclusive |
| mnar | b=0.01 | 0.242427 | 0.0777466 | 0.474185 | yes | Neyman worse |
| mnar | b=0.05 | 0.0421637 | -0.0685416 | 0.161748 | no | CI includes 0 / inconclusive |
| mnar | b=0.10 | 0.0389847 | -0.141606 | 0.267922 | no | CI includes 0 / inconclusive |
| mnar | b=0.20 | -0.00573082 | -0.0697215 | 0.046546 | no | CI includes 0 / inconclusive |
| nonstationary | b=0.01 | 0.260002 | 0.0536591 | 0.506485 | yes | Neyman worse |
| nonstationary | b=0.05 | 0.0306816 | -0.0669035 | 0.103817 | no | CI includes 0 / inconclusive |
| nonstationary | b=0.10 | 0.0473374 | -0.106207 | 0.247884 | no | CI includes 0 / inconclusive |
| nonstationary | b=0.20 | 0.0110191 | -0.0637302 | 0.0738837 | no | CI includes 0 / inconclusive |
| policy_dependent | b=0.01 | 0.215441 | -0.0719334 | 0.510267 | no | CI includes 0 / inconclusive |
| policy_dependent | b=0.05 | 0.0347972 | -0.0550523 | 0.0882916 | no | CI includes 0 / inconclusive |
| policy_dependent | b=0.10 | 0.0154656 | -0.153492 | 0.218121 | no | CI includes 0 / inconclusive |
| policy_dependent | b=0.20 | -0.000460883 | -0.0711532 | 0.0623176 | no | CI includes 0 / inconclusive |
| zero_overlap | b=0.01 | 0.178641 | -0.200266 | 0.473329 | no | CI includes 0 / inconclusive |
| zero_overlap | b=0.05 | 0.0812345 | 0.0256812 | 0.156859 | yes | Neyman worse |
| zero_overlap | b=0.10 | 0.0513365 | -0.0707617 | 0.234464 | no | CI includes 0 / inconclusive |
| zero_overlap | b=0.20 | 0.0201109 | -0.0124524 | 0.0655048 | no | CI includes 0 / inconclusive |

### 5.4 共同协议（MNAR / zero-overlap，b=0.05；volume 适配器不是 AMPLe official）

协议说明：Adapted volume heuristic vs Uniform IPW vs Neyman IPW on the existing synthetic selective-feedback world. MNAR and zero-overlap only, b=0.05, seeds 0,1,2. Not AMPLe official.

| method | pop-mean bias mean±std | pop-mean RMSE | preference-rate bias | realized audit |
| --- | --- | --- | --- | --- |
| neyman_ipw | -0.098014 ± 0.185173 (n=3) | 0.180184 | -0.0196362 ± 0.07984 (n=3) | 0.048 ± 0.0025 (n=3) |
| uniform_ipw | -0.104575 ± 0.0567611 (n=3) | 0.114385 | -0.0161667 ± 0.0540031 (n=3) | 0.0481667 ± 0.00332916 (n=3) |
| volume_adapted_ipw | -0.0887121 ± 0.0893758 (n=3) | 0.11487 | -0.00440599 ± 0.0450567 (n=3) | 0.05 ± 0.00217945 (n=3) |

配对（来自 COMPARABILITY_MATRIX / common-protocol SUMMARY）MNAR `b=0.05`：
- mean RMSE(Neyman−Uniform) = `0.0421637`
- paired bootstrap CI = [`-0.0685416`, `0.161748`], n_boot=`10000`, preliminary=`true`

### 5.5 图

- `idea_A/results/aggregate_plots/bias_by_mechanism_b0.05.png`
- `idea_A/results/aggregate_plots/rmse_vs_budget_mnar.png`
- `idea_A/results/aggregate_plots/rmse_vs_budget_zero_overlap.png`

### 5.6 消融 run 计数

| ablation | INDEX n_runs | MNAR b=0.05 files | MNAR b=0.05 bias mean | RMSE mean | methods in files | INDEX |
| --- | --- | --- | --- | --- | --- | --- |
| A1_No_Adaptive_Allocation | 39 | 3 | -0.101295 | 0.125657 | a1_uniform_ipw,neyman_ipw_reference | idea_A/ablations/A1_No_Adaptive_Allocation/results/INDEX.json |
| A2_No_Propensity_Correction | 39 | 3 | 0.00164733 | 0.163627 | a2_neyman_unweighted,neyman_ipw_reference | idea_A/ablations/A2_No_Propensity_Correction/results/INDEX.json |
| A3_No_Anchor_Intervention | 39 | 3 | 0.890053 | 0.890053 | a3_naive_ips,a3_observed_only | idea_A/ablations/A3_No_Anchor_Intervention/results/INDEX.json |
| A4_Fixed_vs_Adaptive_Budget | 39 | 3 | -0.0150025 | 0.127909 | a4_neyman_batched,a4_neyman_oneshot | idea_A/ablations/A4_Fixed_vs_Adaptive_Budget/results/INDEX.json |
| A5_Misspecified_Propensity | 39 | 3 | -0.056994 | 0.137094 | a5_neyman_ipw_wrong_q,neyman_ipw_reference | idea_A/ablations/A5_Misspecified_Propensity/results/INDEX.json |

A 消融格子：全部机制在 `b=0.05`，以及 MNAR / zero-overlap 的全部预算，seeds 0/1/2。其余 budget×mechanism 为保持运行时间跳过。一次只改一个组件。

## 6. Idea A — Interpretation

1. **内生选择是否产生可测偏差？** 是。MNAR observed-only bias mean = `1.1897`（约 +1.19），远离 0。
2. **提出的方法能否恢复目标量？** 审计 IPW 可以大幅降偏：Neyman IPW bias mean = `-0.098014`，Uniform IPW = `-0.104575`。 naive IPS 仍为 `0.590411`。这支持「已知审计通道」而不是「把自然倾向当已知」。
3. **同等成本下是否优于 Uniform Audit？** 否。MNAR `b=0.05` 配对 RMSE CI 包含 0；多个格子 Neyman 均值更高。zero-overlap `b=0.05` CI 排除 0 且 Neyman 更差。
4. **是否优于 IPS/DR（在可比假设下）？** 主方法本身就是已知 `q` 的 IPW/DR。相对 naive IPS（错误假设）当然更好；这不是对公开 IPS 论文的击败。
5. **是否优于适用的已发表方法？** 不能这么说。AMPLe/HyPER 协议不可比；HyPER blocked；不得把 incomparable 数字排进同一榜。
6. **失败条件：** 零重叠下自然通道不可识别；倾向误指定（A5）；关掉审计锚（A3）；Neyman 用可见 `σ_h` 在 MNAR 下偏倚，低预算时方差更大。
7. **增益是否只来自更强假设或更多信息？** 识别增益来自独立审计（额外标注预算 + 已知 `q`）。分配增益未被配对区间支持。全反馈 oracle 有全部 `Y`，不能当成算法胜利。
8. **去掉已知技术后还剩什么新贡献？** v1.0 没有新分配算法。剩下的是现象：在 MNAR 合成世界里，一条已知 `q` 的审计通道能把 +1.19 量级的观测偏差压到约 −0.10。这是经典抽样在新应用设定上的可行性，不是新定理。

## 7. Idea B — Formal Problem and Method

实现位置：`idea_B/methods/v1.0/{algorithm.py,observability.py,feedback_controller.py,policy_update.py}`。

信息矩阵（来自 SMOKE_SUMMARY）：`W_T = sum_t S_t phi_t phi_t^T (linear-model observed design Gramian)`。这是线性模型的观测设计 Gram 矩阵，**不是**动力系统可观性 Gramian。条件期望版本 `W_T^{cond}=∑_t q_t φ_t φ_t^⊤`。

- horizon=`32`，population_per_round=`40`，primary_budget=`0.05`，seeds=`[0, 1, 2]`。
- 闭环：参数给 query 打分 → 环境按策略改变自然反馈分布 → 有限预算审计 → 用观测标签更新参数 → 新参数成为下一轮策略。
- B1 是外生缺失诊断，**不得**叫做内生坍缩。B2 是策略依赖反馈。B3 分 recoverable / unrecoverable。
- 控制器：从 `W_T` 读出弱方向（默认 λ_min / 最弱特征向量），把同一预算分到信息不足的技能，并在技能单纯形上平滑，避免剧烈切换。
- 指标：真实表现、观测表现、差距、技能误差、`λ_min(W_T/T)`、条件数、反馈成本。更好的 `W_T` **不自动**等于更好的真实表现。
- 固定策略 + 固定抽样不得标成 endogenous collapse。

信息度量的数学理由：线性模型 `y=φ^⊤θ+ε` 的观测设计 Gram 矩阵决定 `θ` 的最小二乘协方差形状；`λ_min` 小表示存在近不可识别方向。这是标准实验设计 / E-optimality 读出，不是新的 Gramian 定义。

## 8. Idea B — Experiments

- 合成闭环 run 数 `435`，skips=`[]`，wall_time_s=`79.1566`。
- 命令：`PYTHONPATH=. .venv/bin/python -m idea_B.experiments.run_synthetic`
- torch_installed（主实验记录）=`false`
- blockers：['Torch is not installed; NumPy/SciPy only. No LLM download.', 'Official clone directories Performative_Bandit_ICML2024 and Performative_SGD_NeurIPS2024 were not modified.']

### 8.1 Environment B1（外生缺失诊断）

B1 is an exogenous-missing diagnostic. Do not call this endogenous collapse.

| method | late true perf | late observed perf | late λ_min | late weak-skill error | feedback cost | endogenous_collapse_any |
| --- | --- | --- | --- | --- | --- | --- |
| ObservabilityController | 0.978315 ± 0.000588416 | 0.978357 ± 0.00120957 | 1.74966 ± 0.181997 | 0.0250914 ± 0.0169227 | 0.0528646 ± 0.00502273 | false |
| UniformExploration | 0.97773 ± 0.00121082 | 0.978796 ± 0.0014326 | 0.621438 ± 0.188302 | 0.0541975 ± 0.0436338 | 0.0549479 ± 0.00664445 | false |
| GreedyDeployment | 0.977373 ± 0.00133356 | 0.978421 ± 0.00128022 | 0.300928 ± 0.015846 | 0.0662661 ± 0.0457042 | 0.05 ± 8.49837e-18 | false |
| NoFeedbackController | 0.977002 ± 0.00190165 | 0.978346 ± 0.00179212 | 0.286933 ± 0.0307283 | 0.0754946 ± 0.059963 | 0 ± 0 | false |
| FullFeedbackOracle | 0.97861 ± 0.000299263 | 0.97861 ± 0.000299263 | 6.53694 ± 0.379708 | 0.00752428 ± 0.00256455 | 1 ± 0 | false |

### 8.2 Environment B2，budget=0.05

- `endogenous_collapse` = `false`
- joint criterion = `false`
- all_seeds_observed_stable = `true`
- all_seeds_lambda_min_worsened = `true`
- any_seed_weak_skill_error_worsened = `false`
- mean_final_weak_skill_observation_rate = `0`
- evidence.note：Joint endogenous-collapse label is False: observed performance stayed high and lambda_min(W_T/T) fell on all three greedy seeds, but weak-skill parameter error did not reliably increase relative to the early window because theta starts at 0 (early error is already large). Weak-skill observation rate at the end is 0 on these greedy runs. Greedy extra audits land on already-served queries, so Greedy matches No Feedback Controller at the same seed.

| method | late true perf | late observed perf | late λ_min | late weak-skill error | feedback cost | endogenous_collapse_any |
| --- | --- | --- | --- | --- | --- | --- |
| ObservabilityController | 0.978371 ± 0.000466183 | 0.978336 ± 0.00115423 | 1.76217 ± 0.202493 | 0.0194937 ± 0.0115528 | 0.0526042 ± 0.00385381 | false |
| UniformExploration | 0.977818 ± 0.000818027 | 0.97871 ± 0.00187418 | 0.531849 ± 0.313144 | 0.0533181 ± 0.043611 | 0.0549479 ± 0.00664445 | false |
| GreedyDeployment | 0.972631 ± 0.00443881 | 0.977837 ± 0.00176599 | 0.0794893 ± 0.0173963 | 0.173393 ± 0.070174 | 0.05 ± 8.49837e-18 | false |
| LazyDeployment | 0.976054 ± 0.00286848 | 0.977965 ± 0.00181574 | 0.452244 ± 0.18701 | 0.0992701 ± 0.0701844 | 0.05 ± 8.49837e-18 | false |
| StandardOnlineSGD | 0.938205 ± 0.0193467 | 0.974243 ± 0.00347974 | 0.417447 ± 0.180616 | 0.483602 ± 0.122524 | 0.0549479 ± 0.00664445 | false |
| FixedExplorationSchedule | 0.977818 ± 0.000818027 | 0.97871 ± 0.00187418 | 0.531849 ± 0.313144 | 0.0533181 ± 0.043611 | 0.0549479 ± 0.00664445 | false |
| UncertaintyExploration | 0.978349 ± 0.000506257 | 0.978543 ± 0.000962625 | 1.14657 ± 0.185644 | 0.0295732 ± 0.0271495 | 0.0559896 ± 0.0056517 | false |
| NoFeedbackController | 0.972631 ± 0.00443881 | 0.977837 ± 0.00176599 | 0.0794893 ± 0.0173963 | 0.173393 ± 0.070174 | 0 ± 0 | false |
| FullFeedbackOracle | 0.97861 ± 0.000299263 | 0.97861 ± 0.000299263 | 6.53694 ± 0.379708 | 0.00752428 ± 0.00256455 | 1 ± 0 | false |

### 8.3 Controller vs Uniform（配对，budget=0.05）

| metric | mean(controller−uniform) | CI low | CI high | n_boot | CI excludes 0 |
| --- | --- | --- | --- | --- | --- |
| late λ_min | 1.23032 | 1.1027 | 1.30783 | 5000 | yes |
| late true performance | 0.00055264 | -7.73095e-05 | 0.00114084 | 5000 | no |

JSON `beats_uniform_on_both` = `true`（按均值符号）。配对规则下信息量支持、真实表现不支持。

### 8.4 Environment B3

**recoverable**

| method | late true perf | late observed perf | late λ_min | late weak-skill error | feedback cost | endogenous_collapse_any |
| --- | --- | --- | --- | --- | --- | --- |
| ObservabilityController | 0.978371 ± 0.000466183 | 0.978336 ± 0.00115423 | 1.76217 ± 0.202493 | 0.0194937 ± 0.0115528 | 0.0526042 ± 0.00385381 | false |
| UniformExploration | 0.977818 ± 0.000818027 | 0.97871 ± 0.00187418 | 0.531849 ± 0.313144 | 0.0533181 ± 0.043611 | 0.0549479 ± 0.00664445 | false |
| GreedyDeployment | 0.972631 ± 0.00443881 | 0.977837 ± 0.00176599 | 0.0794893 ± 0.0173963 | 0.173393 ± 0.070174 | 0.05 ± 8.49837e-18 | false |
| NoFeedbackController | 0.972631 ± 0.00443881 | 0.977837 ± 0.00176599 | 0.0794893 ± 0.0173963 | 0.173393 ± 0.070174 | 0 ± 0 | false |
| FullFeedbackOracle | 0.97861 ± 0.000299263 | 0.97861 ± 0.000299263 | 6.53694 ± 0.379708 | 0.00752428 ± 0.00256455 | 1 ± 0 | false |

**unrecoverable**

| method | late true perf | late observed perf | late λ_min | late weak-skill error | feedback cost | endogenous_collapse_any |
| --- | --- | --- | --- | --- | --- | --- |
| ObservabilityController | 0.837404 ± 0.00594651 | 0.977156 ± 0.00118883 | 0.0369267 ± 0.00438674 | 0.367499 ± 0.143886 | 0.0153646 ± 0.00090211 | false |
| UniformExploration | 0.837496 ± 0.0060687 | 0.97748 ± 0.00110821 | 0.0387177 ± 0.00449916 | 0.353036 ± 0.117675 | 0.046875 ± 0.00340539 | false |
| GreedyDeployment | 0.837292 ± 0.00592175 | 0.976843 ± 0.00148665 | 0.0354458 ± 0.0038088 | 0.397043 ± 0.104872 | 0.0497396 ± 0.000451055 | false |
| NoFeedbackController | 0.837292 ± 0.00592175 | 0.976843 ± 0.00148665 | 0.0354458 ± 0.0038088 | 0.397043 ± 0.104872 | 0 ± 0 | false |
| FullFeedbackOracle | 0.837629 ± 0.00604905 | 0.978052 ± 0.000818008 | 0.0879458 ± 0.008825 | 0.239337 ± 0.0110122 | 0.83724 ± 0.00945065 | false |

### 8.5 共同协议 B2 b=0.05

n_runs=`9`；excluded：['PP-bandit adapted (skipped; see adapted_skipped)', 'PP-bandit official 1-D numerical demo (different protocol)', 'ncvxpp / Performative SGD synthetic-iter (different protocol)', 'Greedy/Lazy must not be relabeled as official SGD-GD']

| method | late true perf | late observed perf | late λ_min | late weak-skill error | feedback cost | endogenous_collapse_any |
| --- | --- | --- | --- | --- | --- | --- |
| ObservabilityController | 0.978371 ± 0.000466183 (n=3) | 0.978336 ± 0.00115423 (n=3) | 1.76217 ± 0.202493 (n=3) | 0.0194937 ± 0.0115528 (n=3) | 0.0526042 ± 0.00385381 (n=3) | NA |
| UniformExploration | 0.977818 ± 0.000818027 (n=3) | 0.97871 ± 0.00187418 (n=3) | 0.531849 ± 0.313144 (n=3) | 0.0533181 ± 0.043611 (n=3) | 0.0549479 ± 0.00664445 (n=3) | NA |
| GreedyDeployment | 0.972631 ± 0.00443881 (n=3) | 0.977837 ± 0.00176599 (n=3) | 0.0794893 ± 0.0173963 (n=3) | 0.173393 ± 0.070174 (n=3) | 0.05 ± 8.49837e-18 (n=3) | NA |

PP-bandit adapted 状态：
- `pp_bandit_zeroth_order_reparam_adapted` status=`BLOCKED` protocol=`adapted-skipped`
  Official PP-bandit core (bandit-icml24.ipynb / official_reproduction/run_official_numerical.py) is a 1-D reparameterization x = y^2 with two-point zeroth-order updates and an inner LearnModel that minimizes Bernoulli KL(x_target || y^2). That procedure queries an uncensored scalar performative-risk function. Idea B environment B2 is a 3-D linear skill model whose performative channel is skill-wise observation censoring, not a Bernoulli mean map, and labels are revealed only under a limited audit budget. A faithful LearnModel+reparameterization port is not feasible on this NumPy protocol without rewriting the paper's 1-D Bernoulli experiment. A two-point ZO-SGD with uniform audits would be a different generic optimizer and must not be relabeled as adapted PP-bandit. Official numerical demo remains in idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/ (COMPLETE, separate from this leaderboard).

### 8.6 图

- `idea_B/results/plots/method_vs_generic_baselines.png`
- `idea_B/results/plots/collapse_recovery_trajectories.png`
- `idea_B/results/plots/performance_vs_budget.png`
- 每个 closed-loop run 另有 `plots/{true_vs_observed_performance,information_eigenvalues,skill_specific_accuracy,skill_specific_observation_rates,regret_vs_rounds}.png`。

### 8.7 消融

B 消融 total_runs=`33`：B1=`6`，B2=`6`，B3=`6`，B4=`6`，B5=`9`。

| ablation | arm | n | late true perf mean | late λ_min mean | late weak-skill err mean | collapse fraction |
| --- | --- | --- | --- | --- | --- | --- |
| B1_No_Observability_Control | no_control | 3 | 0.972631 | 0.0794893 | 0.173393 | 0 |
| B1_No_Observability_Control | reference_controller | 3 | 0.978371 | 1.76217 | 0.0194937 | 0 |
| B2_No_Adaptive_Exploration | uniform_inside_controller | 3 | 0.977818 | 0.531849 | 0.0533181 | 0 |
| B2_No_Adaptive_Exploration | adaptive_controller | 3 | 0.978371 | 1.76217 | 0.0194937 | 0 |
| B3_Fixed_vs_Adaptive_Feedback | fixed_equal_skill | 3 | 0.977818 | 0.531849 | 0.0533181 | 0 |
| B3_Fixed_vs_Adaptive_Feedback | adaptive_controller | 3 | 0.978371 | 1.76217 | 0.0194937 | 0 |
| B4_No_Policy_Feedback_Loop | exogenous_B1 | 3 | 0.978315 | 1.74966 | 0.0250914 | 0 |
| B4_No_Policy_Feedback_Loop | closed_loop_B2 | 3 | 0.978371 | 1.76217 | 0.0194937 | 0 |
| B5_Observability_Metric_Variants | lambda_min | 3 | 0.978371 | 1.76217 | 0.0194937 | 0 |
| B5_Observability_Metric_Variants | trace_inverse | 3 | 0.978205 | 1.04988 | 0.0376273 | 0 |
| B5_Observability_Metric_Variants | logdet | 3 | 0.978205 | 1.04988 | 0.0376273 | 0 |

## 9. Idea B — Interpretation

1. **可观性坍缩能否内生出现？** 当前联合判据为 false。Greedy / NoFeedback 上观测表现稳定、λ_min 下降，但弱技能误差没有相对早期窗口继续变差（早期 `θ=0` 误差已经很大）。弱技能最终观测率为 0。
2. **是否不同于普通遗忘或非平稳抽样？** B1 明确是外生缺失。B4 把 B2 换成 B1 通道。当前还不能把 B2 标成已证实的策略诱发坍缩。
3. **更低信息是否预测更高估计误差？** Greedy 的 late λ_min mean = `0.0794893`， late weak-skill error mean = `0.173393`； 控制器 late λ_min mean = `1.76217`， weak-skill error mean = `0.0194937`。方向一致，但坍缩的时间动态（误差继续恶化）未满足。
4. **控制器是否阻止信息损失？** 相对 Uniform，λ_min 配对 CI 排除 0，支持信息量改善。
5. **是否改善长期学习（真实表现）？** 均值略高，但配对 CI 包含 0，**不支持** PROMISING 所需的真实表现优势。
6. **是否击败适用的已发表基线？** 否。PP-bandit / ncvxpp 不可比；adapted PP-bandit BLOCKED。Greedy 是 generic control。
7. **对预算与模型敏感是否稳健？** 主矩阵覆盖四个预算，但 `n_seeds=3`。真实表现优势未过配对门槛。
8. **能否用「只是多探索」解释？** 预算与 Uniform 对齐。信息量改善不能还原成更高花费。但若只看真实表现，差异与多探索/分配噪声无法分离。

## 10. Negative Results and Failure Cases

| 配置 | 观察到的失败 | 可能原因 | 证据 | 仍不确定 |
| --- | --- | --- | --- | --- |
| Idea A Neyman vs Uniform，MNAR b=0.05 | 配对 RMSE CI 含 0；差 mean=`0.0421637` | 可见 σ_h 在 MNAR 下偏倚；n_seeds=3 方差大 | `idea_A/results/SMOKE_SUMMARY.json` key_findings | 更大种子数是否翻转 |
| Idea A Neyman vs Uniform，zero-overlap b=0.05 | Neyman 显著更差，CI=[0.0256812, 0.156859] | 零重叠层的可见方差不能代表隐层 | 同一 paired 表 | 其他分配启发式是否同样失败 |
| Idea A naive IPS on MNAR | bias mean=`0.590411` | 自然倾向不可识别却被当已知 | per_method.naive_ips | 无 |
| HyPER official | blocked | TPU v3-128, vLLM, 8B–13B RM/DPO, Beaker/GCS, 未初始化 submodule | REPRODUCTION_AUDIT.json | 有集群时能否跑 |
| Drago ICML 2026 | blocked | 无官方代码 | THEORY_COMPARATOR.md | 作者是否稍后放码 |
| AMPLe full README | partial：n_rounds=3 而非 100；未用 --dims 0 1 | burnin=50000 太慢；2-aspect 数据集 HTTP 401 | AMPLe what_ran | 完整 100 round 曲线 |
| ncvxpp mpiexec -np 10 | blocked；单进程 partial | 无 mpi4py/mpicc；data.csv 标签 {+1,-1} 与 Y==1 过滤互作用 | npy loss/acc | 10 replica 官方曲线 |
| PP-bandit → B2 adapted | BLOCKED | 1-D Bernoulli KL reparam 无法忠实接到 3-D 技能删失 | common_protocol SUMMARY.adapted_skipped | 无忠实 port |
| Idea B B2 endogenous collapse | joint=false；weak-skill error 未继续恶化 | θ 从 0 起步，早期误差已大 | B2.evidence.note | 换初始化是否出现联合坍缩 |
| Idea B 真实表现 vs Uniform | CI 含 0，mean diff=`0.00055264` | horizon=32 太短或信号弱 | controller_vs_uniform | 更长 horizon |
| Qwen MPS generate | abort NSInvalidArgumentException | macOS MPS / torch 2.15 不匹配 | INFERENCE_SMOKE.json | MPS 是否可修 |
| RewardBench leaderboard | 未跑 | 没有兼容的小 RM；Qwen 是因果 LM | REAL_DATA_FEASIBILITY.json | 无 |

负结果全部保留。没有删除失败格子。

## 11. Ablation Analysis

Idea A 一次只改一个组件：

- **A1 No Adaptive Allocation**：Neyman → 均匀 `q=b`，估计仍是审计 IPW。检验层方差倾斜是否改变误差。INDEX n_runs 见上表。
- **A2 No Propensity Correction**：分配仍是 Neyman，估计改成审计样本未加权均值。检验 `A/q` 是否必要。
- **A3 No Anchor Intervention**：`q=0`，只留自然通道。MNAR/零重叠下这是识别失败。
- **A4 Fixed vs Adaptive Budget**：同一 Neyman+IPW，只改预算日程（一次分完 vs 分批更新 σ_h）。两者都是经典抽样日程。
- **A5 Misspecified Propensity**：真实 `q` 抽样，错误 `q̂` 做 IPW。检验「已知倾向」被破坏时的退化。

MNAR `b=0.05` 消融臂总体均值 bias mean（从各 `metrics.json` 现算，不是手填）：

- `A1_No_Adaptive_Allocation`: bias_mean=`-0.101295` rmse_mean=`0.125657` n=`6` methods=['a1_uniform_ipw', 'neyman_ipw_reference']
- `A2_No_Propensity_Correction`: bias_mean=`0.00164733` rmse_mean=`0.163627` n=`6` methods=['a2_neyman_unweighted', 'neyman_ipw_reference']
- `A3_No_Anchor_Intervention`: bias_mean=`0.890053` rmse_mean=`0.890053` n=`6` methods=['a3_naive_ips', 'a3_observed_only']
- `A4_Fixed_vs_Adaptive_Budget`: bias_mean=`-0.0150025` rmse_mean=`0.127909` n=`6` methods=['a4_neyman_batched', 'a4_neyman_oneshot']
- `A5_Misspecified_Propensity`: bias_mean=`-0.056994` rmse_mean=`0.137094` n=`6` methods=['a5_neyman_ipw_wrong_q', 'neyman_ipw_reference']

Idea B：

- **B1 No Observability Control**：关掉基于 `W_T` 的分配。
- **B2 No Adaptive Exploration**：`mix_uniform=1`，花费相同预算但不按弱方向加权。
- **B3 Fixed vs Adaptive Feedback**：技能均分 vs 最弱方向。
- **B4 No Policy Feedback Loop**：B2 闭环通道换成 B1 外生缺失。
- **B5 Observability Metric Variants**：λ_min / tr(W^{-1}) / logdet，不改 Gram 定义。

消融支持「信息分配改变 `W_T`」，但没有把 B2 联合坍缩判据从 false 改成 true，也没有把真实表现 CI 从含 0 改成排除 0。

## 12. Novelty Assessment

| 类别 | Idea A | Idea B |
| --- | --- | --- |
| 既有文献结果 | 分层 Neyman、IPW/Hájek/AIPW、两阶段抽样 | 观测设计 Gram 矩阵、E-optimality、performative prediction |
| 新观察到的实验现象 | MNAR 合成世界中 known-q 审计把 observed-only +1.19 bias 降到约 −0.10（n_seeds=3） | 控制器提高 λ_min（配对 CI 排除 0）；联合内生坍缩未出现 |
| 新算法机制 | **无。v1.0 明确不是新算法。** | 有限预算下按弱特征向量分配观测；是标准实验设计启发式，不是新最优性定理 |
| 未证明的理论猜想 | 内生删失 + 独立审计的有限样本保证 | 策略诱发可观性坍缩的一般条件 |
| 可能的新颖性 | 把审计识别接到偏好/选择性反馈，而非新分配规则 | 「观测表现稳定但信息坍缩」这一现象，当前实验尚未联合成立 |
| 与已发表论文的潜在重叠 | AMPLe/HyPER 都涉及反馈选择，但目标与信息不同 | ICML 2024 / NeurIPS 2024 performative 论文研究决策依赖分布，不研究技能级删失 Gramian |

经验证据不等于已证明的新颖性。不得声称 outperformed CCF-A Main 论文。

## 13. Limitations

- Mac compute：16GB ram、无 NVIDIA GPU、磁盘从约 12Gi 降到审计记录的 `9.7Gi`，真实数据阶段后约 `6.35506` GiB。
- 缺失官方复现：HyPER blocked；Drago 无代码；AMPLe 缩短 round；ncvxpp 非 10 replica；PP-bandit 未接到 B2。
- `n_seeds=3`，配对 bootstrap 不是 confirmatory。
- Qwen 只是 CPU generate smoke；RewardBench 无 RM 打分，是 proxy/schema 校验。
- 合成环境：线性奖励 / 线性技能、短 horizon、已知层或已知特征。
- MultiPref 只落了 50 个 prompt_id 的确定性子集，未做后训练。
- 没有新的识别或坍缩定理。
- 主 venv 无 torch；隔离 venv 才有 torch 2.15.0。

## 14. Final Recommendation

### Idea A：`INCONCLUSIVE`

识别-via-audit 被 MNAR 数字支持；分配新颖性不被配对区间支持（有时更差）。总体 **INCONCLUSIVE**，且必须写明 v1.0 是经典 Neyman。

下一步最能改变结论的实验：在 MNAR 下把 seeds 增到 ≥30，并比较 **Uniform IPW**（已是强对照）与任何真正非经典的审计设计；不要继续把 Neyman 当新方法。

### Idea B：`INCONCLUSIVE`

信息量改善有配对支持；内生坍缩联合判据失败；真实表现 CI 含 0。总体 **INCONCLUSIVE**。

下一步最能改变结论的实验：改变初始化使早期弱技能误差较小，检查弱技能误差是否随后上升；同时加长 horizon，使真实表现配对 CI 有机会排除 0。不要把 B1 外生缺失写成闭环坍缩。

对 ICML / NeurIPS / ACL Main 的投资建议：当前烟测**不足以**支持「已有可投稿的新算法」。可以继续做识别现象或可观性现象的机制研究，但必须先过预先写死的 PROMISING 门槛。

## 15. Reproduction Commands

在仓库根目录执行。主 Python：`/Users/grantwright/Research/idea-A-B/.venv/bin/python`。需要 torch / transformers / datasets 的步骤用 `.isolated_baselines_venv`。

```bash
cd /Users/grantwright/Research/idea-A-B
export PYTHONPATH="$PWD"

# Install (NumPy stack; torch 允许失败)
bash scripts/setup_mac.sh
.venv/bin/python scripts/verify_installation.py
.venv/bin/python -m shared.hardware

# Idea A synthetic + ablations
.venv/bin/python -m idea_A.experiments.run_synthetic
.venv/bin/python idea_A/ablations/A1_No_Adaptive_Allocation/run.py
.venv/bin/python idea_A/ablations/A2_No_Propensity_Correction/run.py
.venv/bin/python idea_A/ablations/A3_No_Anchor_Intervention/run.py
.venv/bin/python idea_A/ablations/A4_Fixed_vs_Adaptive_Budget/run.py
.venv/bin/python idea_A/ablations/A5_Misspecified_Propensity/run.py
.venv/bin/python -m idea_A.experiments.run_all
.venv/bin/python idea_A/experiments/common_protocol/run.py

# Idea B synthetic + ablations
.venv/bin/python -m idea_B.experiments.run_synthetic
.venv/bin/python idea_B/ablations/B1_No_Observability_Control/run.py
.venv/bin/python idea_B/ablations/B2_No_Adaptive_Exploration/run.py
.venv/bin/python idea_B/ablations/B3_Fixed_vs_Adaptive_Feedback/run.py
.venv/bin/python idea_B/ablations/B4_No_Policy_Feedback_Loop/run.py
.venv/bin/python idea_B/ablations/B5_Observability_Metric_Variants/run.py
.venv/bin/python idea_B/experiments/common_protocol/run.py

# Official baselines (isolated venv / as recorded)
# AMPLe partial: see idea_A/baselines/REPRODUCTION_AUDIT.json what_ran.command
# HyPER: BLOCKED — do not invent a simplified implementation
.venv/bin/python idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/run_official_numerical.py
# ncvxpp full mpiexec -np 10 is BLOCKED; partial single-process is recorded in REPRODUCTION_AUDIT.json

# Real-data feasibility (isolated venv)
.isolated_baselines_venv/bin/python scripts/download_benchmarks.py
.isolated_baselines_venv/bin/python scripts/download_models.py

# Tests and reports
.venv/bin/python -m pytest
.venv/bin/python scripts/generate_report.py
```

AMPLe 实际跑过的命令（partial，n_rounds=3，acquisition 仍为 volume，burnin=50000）：

```text
python main.py --seed=1 --task=assistant --rm_names harmless helpful humor --n_rounds=3 --true_preference 0.2 0.7 0.1 --true_beta=10 --beta=2.0 --jitter=0.01 --alpha=10 --init=previous --acquisition=volume --burnin=50000 --gamma=0.0 --mode=likelihood --margin=none --inputs=0
```

## 16. Complete Result Index

### 摘要 JSON

- `idea_A/results/SMOKE_SUMMARY.json` (idea_A_smoke, exists=yes)
- `idea_B/results/SMOKE_SUMMARY.json` (idea_B_smoke, exists=yes)
- `idea_A/baselines/REPRODUCTION_AUDIT.json` (idea_A_audit, exists=yes)
- `idea_B/baselines/REPRODUCTION_AUDIT.json` (idea_B_audit, exists=yes)
- `reports/COMPARABILITY_MATRIX.json` (comparability, exists=yes)
- `idea_A/experiments/common_protocol/SUMMARY.json` (idea_A_common, exists=yes)
- `idea_B/experiments/common_protocol/SUMMARY.json` (idea_B_common, exists=yes)
- `reports/REAL_DATA_FEASIBILITY.json` (real_data, exists=yes)
- `shared/hardware/environment_manifest.json` (env_shared, exists=yes)
- `reports/environment_manifest.json` (env_reports, exists=yes)
- `idea_A/models/Qwen2.5-0.5B-Instruct/INFERENCE_SMOKE.json` (inference_A, exists=yes)
- `idea_A/models/Qwen2.5-0.5B-Instruct/MODEL_MANIFEST.md` (manifest_A, exists=yes)
- `idea_A/experiments/common_protocol/SUMMARY.json`
- `idea_B/experiments/common_protocol/SUMMARY.json`

### Idea A run 目录

- 主矩阵 `84` 个目录，每个含 `config.yaml, metrics.json, metrics.csv, run.log, environment.json, plots/`。
- 根：`idea_A/results/syn_<mechanism>_b<budget>_s<seed>/`
- 消融 INDEX：`idea_A/ablations/A1_No_Adaptive_Allocation/results/INDEX.json`, `idea_A/ablations/A2_No_Propensity_Correction/results/INDEX.json`, `idea_A/ablations/A3_No_Anchor_Intervention/results/INDEX.json`, `idea_A/ablations/A4_Fixed_vs_Adaptive_Budget/results/INDEX.json`, `idea_A/ablations/A5_Misspecified_Propensity/results/INDEX.json`
- 共同协议：`idea_A/experiments/common_protocol/results/`

### Idea B run 目录

- 主矩阵 n_runs=`435`，根 `idea_B/results/synthetic_<env>_<method>_budget*_seed*/`
- 消融 INDEX：`idea_B/ablations/B1_No_Observability_Control/results/ablation_index.json`, `idea_B/ablations/B2_No_Adaptive_Exploration/results/ablation_index.json`, `idea_B/ablations/B3_Fixed_vs_Adaptive_Feedback/results/ablation_index.json`, `idea_B/ablations/B4_No_Policy_Feedback_Loop/results/ablation_index.json`, `idea_B/ablations/B5_Observability_Metric_Variants/results/ablation_index.json`
- 共同协议：`idea_B/experiments/common_protocol/results/`
- 轨迹：每个 run 的 `trajectories.json` / `trajectories.npz`

### 图

- `idea_A/results/aggregate_plots/bias_by_mechanism_b0.05.png`
- `idea_A/results/aggregate_plots/rmse_vs_budget_mnar.png`
- `idea_A/results/aggregate_plots/rmse_vs_budget_zero_overlap.png`
- `idea_B/results/plots/method_vs_generic_baselines.png`
- `idea_B/results/plots/collapse_recovery_trajectories.png`
- `idea_B/results/plots/performance_vs_budget.png`

### 模型与数据

- 共享权重缓存（gitignore）：`/Users/grantwright/Research/idea-A-B/.cache/models/Qwen2.5-0.5B-Instruct`
- Idea A 模型目录（manifest + 相对符号链接）：`/Users/grantwright/Research/idea-A-B/idea_A/models/Qwen2.5-0.5B-Instruct`
- Idea B 模型目录：`/Users/grantwright/Research/idea-A-B/idea_B/models/Qwen2.5-0.5B-Instruct`
- MODEL_MANIFEST：`idea_A/models/Qwen2.5-0.5B-Instruct/MODEL_MANIFEST.md`
- MultiPref subset：`/Users/grantwright/Research/idea-A-B/idea_A/benchmarks/MultiPref/subset.json`
- RewardBench sample：`/Users/grantwright/Research/idea-A-B/idea_A/benchmarks/RewardBench/schema_sample.json`

### Git

- branch `v1` commit `494fe90f8e08e40d5f2bb8a6020e25ce74646c20` remote `git@github.com:SelinaMai-source/idea-A-B.git`
- probe 记录的 commit：`494fe90f8e08e40d5f2bb8a6020e25ce74646c20` error=`None`

---

数字来源全部是磁盘 JSON。若某单元格在 JSON 中不存在，表中写 NA，而不是补造。
