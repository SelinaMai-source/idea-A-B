# Idea C v0.1 全文 Figure / Table 复现清单

工作副本 `/Users/grantwright/Desktop/科研/10.30新课题/v0`，分支 `idea-c-v0.1`。机器可读副本：`idea_C/scripts/v0_1_registry.py`。统一入口：`idea_C/scripts/run_v0_1.py`。对照：`idea_C/scripts/verify_paper_results.py`。

**官方代码：无。** 见 `OFFICIAL_CODE_AUDIT.md`。实现模式只有 `INDEPENDENT_REIMPLEMENTATION`。禁止 `OFFICIAL_*`。UAI **不是** CCF-A Main。e-PS **是预印本**。v0 的 30-seed 不是全文复现。未跑格子保持 `NOT_RUN`，不填 0。

本文写于任何 v0.1 数值结果之前。容差事后不放宽。

排除的非数值对象（有意不进计数）：UAI Figure 1（贡献示意图）、UAI Table 1（复杂度大 O 表）。e-PS Figure 1 是正文数值预览，已计入。AAAI 附录表因 PDF 未读到而单列 `NOT_LOCATED_IN_PDF`。

| Paper | 计入对象 | 说明 |
| --- | --- | --- |
| AAAI 2026 Zhou & Liu | 5 | Fig 1–4 + 未能打开的附录数值表 |
| SCORE ICML 2026 | 16 | Fig 1–4, Table 1, App B Fig 5–9 + Table 2–3, App E Fig 10–12, App F Fig 13 |
| UAI 2026（非 CCF-A） | 11 | Fig 2 power、Fig 2 runtime、Table 2、Fig 3、App Fig 4–10 |
| e-PS 预印本 | 11 | Fig 1（额外）、Fig 2a–d、Fig 3–4、App D Fig 5–8 |
| **合计** | **43** | 全部 `status=NOT_RUN` |

状态词只用：`OFFICIAL_FULL_MATCH`、`OFFICIAL_PARTIAL_MATCH`、`INDEPENDENT_FULL_MATCH`、`INDEPENDENT_PARTIAL_MATCH`、`FAIL_MISMATCH`、`BLOCKED_NO_CODE`、`BLOCKED_NO_DATA`、`BLOCKED_COMPUTE`、`NOT_RUN`、`INVALID_PROTOCOL`。无官方代码时禁止 `OFFICIAL_*`。

---

## 0. 预设容差（看结果前写死）

1. **确定型表。** SCORE Yeast 发现数、UAI NYC Taxi 标注窗口计数：数据顺序固定时逐方法对齐。相对差 `\|local-paper\|/\|paper\|`（paper=0 时用绝对差）**大于 5%** = `FAIL_MISMATCH`，直到给出解释。不得事后把 5% 放宽。
2. **曲线。** 无官方 CSV / 数据转储的 Figure 一律 `FIGURE_ONLY`。禁止把描点读数写成论文数字。可视化叠图 ≠ 原始数据对照。描点误差若以后记录，单独成行，不进入论文值列。
3. **FDR。** 经验 FDR 不得超过 \(\alpha + 2\) 个 Monte Carlo 标准误。缺 per-trial FDP 则不能宣称 FDR 通过。
4. **30-seed。** 只能标 `pilot`。不能标全文复现，不能标 `INDEPENDENT_FULL_MATCH`。
5. **Paper-scale。** AAAI / SCORE / e-PS 合成 500 trials；UAI 局部相关 200 trials；UAI runtime \(n=80\)。未带 `--paper-scale` 的 runner 必须拒绝 `--scale paper`。
6. **官方模式。** `--mode official` 必须报错退出：官方仓不存在。

Seeds：四篇论文均未给出随机种子。下表一律 `not stated in paper`。官方代码列一律 `no`。

---

## 1. AAAI 2026 — On the Robustness of Bandit Multiple Testing

来源：AAAI 2026 Main。PDF 本轮是 Anubis HTML（7040 字节），不是 PDF。实验设定来自 v0 `BASELINE_AUDIT.md` / 论文笔记 / config，附录表未能从 PDF 核对。方法：Uniform / Mean-based / E-process-based（Fig 3 为 Uniform vs Algorithm 3）。**500 trials。** Lévy 稳定性指数 / scale 原文未给。

| ID | Figure/Table | location | benchmark | methods | data/DGP | hyperparameters | seeds | MC reps | horizon | metrics | official code? | script | paper numeric source | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|aaai.fig1|Figure 1|body §4|aaai_robust_gaussian|Uniform, Mean-based, E-process-based|\(P_i=N(\mu_i,1)\)；null \(\mu=0\) / non-null \(\mu=1\)；Lévy-stable 污染概率 1/10，location 1000，skewness 0.5。\|H1\|∈{2,⌊log K⌋,⌊√K⌋}。HTML 摘录未印 K；config 暂用 K=50|\(\delta=0.05\)。Alg2 burn-in \(B=\lceil 4\varepsilon^{-1}\log(4K/\delta^2)\rceil\)，K=50 时 KB=22600。paper-scale horizon **必须 > KB**。Lévy α/scale 原文缺，代码 1.5/1.0 不得声称与作者一致|not stated|500|TPR vs pulls；须过 burn-in|TPR, FDR|no|`idea_C/scripts/run_aaai.py` via `run_v0_1.py`|FIGURE_ONLY|INDEPENDENT_PARTIAL_MATCH|
|aaai.fig2|Figure 2|body §4|aaai_robust_gaussian_K_sweep|同上|同 Fig 1；相对达目标时间 vs K|停在 TPR≥1−δ 且 FDR≤δ。Samples@Target 右删失，不得只对达标 seed 取均值|not stated|500|直到目标 TPR；须 > KB|Samples@Target, relative time vs K, FDR, TPR|no|`run_aaai.py`（当前只接线 K=50）|FIGURE_ONLY|NOT_RUN|
|aaai.fig3|Figure 3|body §4|aaai_heavy_tailed|Uniform, Algorithm 3|Student-t(2)，null 0 / non-null 5，Lévy 同前，K=50，\|H1\|=2|p=1.85，κ=50，λ=(ε/κ)^{1/p}，δ=0.05|not stated|500|TPR vs pulls；须 > KB|TPR, FDR|no|`run_aaai.py --config aaai_heavy_tailed.json`|FIGURE_ONLY|INDEPENDENT_PARTIAL_MATCH|
|aaai.fig4|Figure 4|body §4|aaai_multiagent|Algorithm 4 多 agent|K=10，\|H1\|=2（笔记/审计）；其余 DGP 未能从 PDF 复核|PDF 未打开|not stated|500|未能从 PDF 恢复|TPR, FDR|no|stub `run_v0_1.py`|FIGURE_ONLY；PDF 未读到|NOT_RUN|
|aaai.app_tables|Appendix numerical tables (if any)|appendix — **NOT_LOCATED_IN_PDF**|aaai_appendix_unknown|unknown|unknown|AAAI OJS 下载被换成 HTML，附录表无法从磁盘列出|not stated|unknown|unknown|unknown|no|stub|NOT_LOCATED_IN_PDF|NOT_RUN|

未能在 PDF 定位：AAAI 全部附录数值表、Figure 1 的 K、Figure 4 的完整 DGP、Lévy 稳定性指数与 scale。

---

## 2. SCORE — ICML 2026 / arXiv:2601.20386

来源：PMLR v306 + arXiv 源码 `icml2026_cr.tex`。合成实验 **500 reps，T=1000**。SCORE+ 仅在 CPQD 下有 FDR 保证；无 CPQD 标 `INVALID_PROTOCOL` / `N/A`，不用其 FDR 保证。Yeast 表是确定型发现数。

| ID | Figure/Table | location | benchmark | methods | data/DGP | hyperparameters | seeds | MC reps | horizon | metrics | official code? | script | paper numeric source | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|score.fig1|Figure 1|body §6.1 independent SAFFRON|score_gaussian_mixture_saffron|e-SAFFRON, SCORE-SAFFRON, SCORE+-SAFFRON|独立 Gaussian mixture。θ~Ber(π₁)，μ\|θ=(1−θ)δ₀+θ N(3,5)，X~N(μ,1)。e=φ(X;3,6)/φ(X;0,1)。π₁∈{0.3,0.8}|α=0.05，λ_t≡0.5，γ_t=1/(t(t+1))|not stated|500|T=1000|average power, power ratio vs e-SAFFRON|no|`run_score.py`|FIGURE_ONLY。终端数在 Table 2|INDEPENDENT_PARTIAL_MATCH|
|score.fig2|Figure 2|body §6.1 independent LORD|score_gaussian_mixture_lord|e-LORD, SCORE-LORD, SCORE+-LORD|同 Fig 1|α=0.05，ω_t≡0.05|not stated|500|T=1000|average power, ratio vs e-LORD|no|`run_score.py`|FIGURE_ONLY。终端数在 Table 2|INDEPENDENT_PARTIAL_MATCH|
|score.fig3|Figure 3|body §6.2 dependent SAFFRON|score_ar_exponential_saffron|同 Fig 1|AR exponential。η_t=1+ρ X_{t−1}，ρ=0.5；μ_t~0.5δ₃+0.5δ₂₀；条件 Exp。e_t=(1/3)exp(ηX(1−1/3))|α=0.05；RAI ω₁=0.05，φ=ψ=0.5；λ_t≡0.5|not stated|500|T=1000|power, ratio vs e-SAFFRON|no|stub（AR DGP 未接线）|FIGURE_ONLY。终端数在 Table 3|NOT_RUN|
|score.fig4|Figure 4|body §6.2 dependent LORD|score_ar_exponential_lord|同 Fig 2|同 Fig 3|α=0.05；RAI 同上|not stated|500|T=1000|power, ratio vs e-LORD|no|stub|FIGURE_ONLY|NOT_RUN|
|score.tab1|Table 1|body §6.3 Yeast|score_yeast_wildenhain|上列 6 法|Wildenhain 2016，417026 化学–基因相互作用。p→e：\((1-p+p\log p)/(p(-\log p)^2)\)|α=0.1|n/a（一次流）|1|全流|Discoveries|no|stub|**表内整数**：e-LORD 25954，SCORE-LORD 28808，SCORE+-LORD 44210，e-SAFFRON 29867，SCORE-SAFFRON 37748，SCORE+-SAFFRON 43624|BLOCKED_NO_DATA|
|score.fig5|Figure 5|App B independent LORD FDR|score_gaussian_mixture_lord_fdr|同 Fig 2|同 Fig 2|同 Fig 2|not stated|500|T=1000|FDR path|no|`run_score.py`|FIGURE_ONLY|NOT_RUN|
|score.fig6|Figure 6|App B independent SAFFRON FDR|score_gaussian_mixture_saffron_fdr|同 Fig 1|同 Fig 1|同 Fig 1|not stated|500|T=1000|FDR path|no|`run_score.py`|FIGURE_ONLY|NOT_RUN|
|score.fig7|Figure 7|App B dependent LORD FDR|score_ar_exponential_lord_fdr|同 Fig 4|同 Fig 4|同 Fig 4|not stated|500|T=1000|FDR path|no|stub|FIGURE_ONLY|NOT_RUN|
|score.fig8|Figure 8|App B dependent SAFFRON FDR|score_ar_exponential_saffron_fdr|同 Fig 3|同 Fig 3|同 Fig 3|not stated|500|T=1000|FDR path|no|stub|FIGURE_ONLY|NOT_RUN|
|score.fig9|Figure 9|App B CPQD 反例|score_cpqd_violation|LORD+SAFFRON 各族|T=1000，奇数非空、偶数 null。e=I{X≤α_t}/α_t。null Unif(0,1)；非空先 α_t/2，触发 null trap 后 (1+α_t)/2|α=0.05。SCORE+ 预期超 α；SCORE 应仍受控。应力测试，不排名|not stated|未印|T=1000|FDR path|no|stub|FIGURE_ONLY|NOT_RUN|
|score.tab2|Table 2|App B α 敏感性，independent|score_gaussian_mixture_alpha_grid|LORD+SAFFRON 各族|同 §6.1|α∈{0.05,0.10,0.15,0.20}，π₁∈{0.3,0.8}|not stated|500|T=1000|FDR(T), Power(T), ratio|no|`run_score.py`（目前只固定 α=0.05）|**表内 MC 均值**（见 registry `SCORE_TAB2_INDEP`）。例：α=0.05, π₁=0.3, LORD Power Baseline/SCORE/SCORE+ = 0.0818 / 0.0886 / 0.4244|INDEPENDENT_PARTIAL_MATCH|
|score.tab3|Table 3|App B α 敏感性，dependent|score_ar_exponential_alpha_grid|同上|同 §6.2|同上|not stated|500|T=1000|FDR(T), Power(T), ratio|no|stub|**表内 MC 均值**（`SCORE_TAB3_DEP`）。例：α=0.05, π₁=0.3, LORD Power 0.0706 / 0.0760 / 0.3239|NOT_RUN|
|score.fig10|Figure 10|App E AR(1) 12 法，条件 p 值|score_ar1_12methods_conditional|6 个 e 法 + LOND/LORD/SAFFRON/ADDIS/Alpha-investing/SupLORD|AR(1)：φ₀=0.5，φ₁=3.0，p₁=0.3，X₀~N(0,4/3)。条件 p=1−Φ(X_t−φ₀ X_{t−1})|α=0.05。附录 itemize 里 T 被注释掉；暂按正文合成 T=1000，PDF 未另印则不得假装已核|not stated|500|T=1000（待 PDF 确认）|Power(T), FDR(T)|no|stub|FIGURE_ONLY|NOT_RUN|
|score.fig11|Figure 11|App E AR(1) 随时间|score_ar1_12methods_time|同 Fig 10|同 Fig 10|α=0.05|not stated|500|T=1000 assumed|Power path, FDR path|no|stub|FIGURE_ONLY|NOT_RUN|
|score.fig12|Figure 12|App E 边际 p 值（FDR 失效）|score_ar1_marginal_pvalues|6 个经典 p 值法|同 AR(1)。边际 p=1−Φ(X_t/√(4/3))|α=0.05。负对照|not stated|500|T=1000 assumed|Power, FDR|no|stub|FIGURE_ONLY|NOT_RUN|
|score.fig13|Figure 13|App F 心梗数据|score_mi_conformal|e-LOND/LORD/SAFFRON 与 SCORE 对应；**不用 SCORE+**|Golovenkin 2020 figshare 12045261。logistic conformal e-value，共享校准|α=0.3；500 次随机划分|not stated|500|划分后测试流|FDR, Power|no|stub|FIGURE_ONLY。数据是 figshare，不是结果 CSV|BLOCKED_NO_DATA|

Table 2 / Table 3 的全部格子写在 `v0_1_registry.py`，verify 脚本按格对照。它们是 MC 均值，FDR 走 α+2 SE 规则；5% 相对差规则只强制用于 Yeast / NYC Taxi。

---

## 3. UAI 2026 — online e-closure / donation（**非 CCF-A Main**）

来源：arXiv:2603.24792 `arxiv_flat.tex`。正文声称 accompanying code，公开仓对不上，模式仍是独立实现。错误度量 **FDR 与 SupFDR 分列**。

| ID | Figure/Table | location | benchmark | methods | data/DGP | hyperparameters | seeds | MC reps | horizon | metrics | official code? | script | paper numeric source | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|uai.fig2_power|Figure 2 power 行|body §4|uai_local_dependence_power|e-LOND, Closed e-LOND, Donation e-LOND, r-LOND, Closed r-LOND, Donation r-LOND|m=200，μ₀=0，μ₁=3，L=100，Σ_{ij}=0.5^{\|i−j\|}（\|i−j\|≤L）。π₁=0.1…0.9。Gaussian LR e / 单侧 Gaussian p。γ_t=1/(t(t+1))|δ=0.1|not stated|200|m=200|Power vs π₁, FDR, SupFDR|no|`run_uai.py`（当前单点 π₁=0.3）|FIGURE_ONLY|INDEPENDENT_PARTIAL_MATCH|
|uai.fig2_runtime|Figure 2 runtime 行|body §4|uai_local_dependence_runtime|同上|同上，π₁=0.3，m 至 3000|n=80；单线程。论文机 Ryzen 9 9950X。Closed 在 m=3000 ~1 小时。M3 上单格外推 >8h 或内存不够 → 该格 `BLOCKED_COMPUTE`；donation/e-LOND 仍按原规模。不砍 closure 搜索空间|not stated|80|m≤3000|mean wall-clock（log）|no|`run_uai.py --m`|FIGURE_ONLY。正文：closed 从 ms 到约 1 小时，donation 仍在 ms|INDEPENDENT_PARTIAL_MATCH|
|uai.tab2|Table 2|body NYC taxi|uai_nyc_taxi|e-LOND, Donation e-LOND, Closed e-LOND|Lavin NYC taxi；Zhang e-GAI 风格 STL 残差密度比 e-value。标注窗口不是精确真发现|固定残差流上确定型计数|n/a|1|taxi 序列长度（表未印单一整数）|落在标注异常窗内的发现数|no|stub|**表内整数**：e-LOND=8，d-eLOND=16，Closed e-LOND=10|NOT_RUN|
|uai.fig3|Figure 3|body NYC taxi 图|uai_nyc_taxi_plot|e-LOND, Donation e-LOND|同 Table 2|可视化叠发现|n/a|1|taxi 序列|图|no|stub|FIGURE_ONLY。计数归 Table 2|NOT_RUN|
|uai.fig4|Figure 4|app alt-γ closed e-LOND power|uai_alt_gamma_power|e-LOND, Closed e-LOND, alt-γ Closed e-LOND|同 §4，μ∈{3,4}|δ=0.1|not stated|200|m=200|Power|no|stub|FIGURE_ONLY|NOT_RUN|
|uai.fig5|Figure 5|app alt-γ FDR|uai_alt_gamma_fdr|同上|同 Fig 4|δ=0.1，应 ≤0.1|not stated|200|m=200|FDR, SupFDR|no|stub|FIGURE_ONLY|NOT_RUN|
|uai.fig6|Figure 6|app μ=4 power|uai_local_dependence_mu4|同 Fig 2 六法|同 Fig 2，μ₁=4|δ=0.1|not stated|200|m=200|Power vs π₁|no|`run_uai.py`（DGP 现为 μ=3，μ=4 尚未覆盖）|FIGURE_ONLY|NOT_RUN|
|uai.fig7|Figure 7|app ARC online e-BH power|uai_arc_ebh_power|online e-BH, donation online e-BH|ARC + 主文 local-dep；μ∈{3,4}|δ=0.1|not stated|200|m=200|Power|no|stub|FIGURE_ONLY|NOT_RUN|
|uai.fig8|Figure 8|app ARC FDR|uai_arc_ebh_fdr|同上|同 Fig 7|δ=0.1|not stated|200|m=200|FDR, SupFDR|no|stub|FIGURE_ONLY|NOT_RUN|
|uai.fig9|Figure 9|app randomized donation power|uai_randomized_donation_power|Donation e-LOND, randomized donation e-LOND|同主文 local-dep；μ∈{3,4}|δ=0.1 + U[0,1]|not stated|200|m=200|Power|no|stub|FIGURE_ONLY|NOT_RUN|
|uai.fig10|Figure 10|app randomized donation FDR|uai_randomized_donation_fdr|同上|同 Fig 9|δ=0.1|not stated|200|m=200|FDR, SupFDR|no|stub|FIGURE_ONLY|NOT_RUN|

---

## 4. e-PS — arXiv:2609.26651（**预印本**）

来源：`main.tex` + `appendix.tex`。θ_k=0.02k·B_k，B_k~Bern(0.1)，抽一次后在 500 trial 内固定。**不是** v0 P1-A 的 μ=1。Figure 2 拆成 1D/5D × Gaussian/t 四个可跑对象。OPT-1.3B 预生成流未发布：Figure 4 缺数据，不得用别的模型冒充。

| ID | Figure/Table | location | benchmark | methods | data/DGP | hyperparameters | seeds | MC reps | horizon | metrics | official code? | script | paper numeric source | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|eps.fig1|Figure 1|body intro（相对计划的额外对象）|eps_intro_multivariate_gaussian|e-PS, Uniform|5D Gaussian 预览：达目标 TPR 的平均预算 vs Uniform，以及 FDR vs budget|细节同 §6|not stated|500|预算轴|budget@target TPR, FDR|no|stub|FIGURE_ONLY|NOT_RUN|
|eps.fig2a|Figure 2(a)|body §6 1D Gaussian|eps_gaussian|Uniform, Greedy, e-PS|K=50，N(θ_k,1)，θ_k=0.02k B_k|α=0.05；v=1.1 σ̂²。Oracle 实线 / Adaptive 虚线分行|not stated|500|TPR/FDR vs budget|TPR, FDR|no|`run_eps.py`|FIGURE_ONLY|BLOCKED_COMPUTE|
|eps.fig2b|Figure 2(b)|body §6 5D Gaussian|eps_gaussian_5d|同上|N(θ_k 1_5, I_5)|同上|not stated|500|vs budget|TPR, FDR|no|stub|FIGURE_ONLY|NOT_RUN|
|eps.fig2c|Figure 2(c)|body §6 1D Student-t|eps_studentt|同上|t_{θ_k,5}（原文 noncentrality；现有代码是 location-shift，不得声称参数一致）|同上|not stated|500|vs budget|TPR, FDR|no|`run_eps.py`|FIGURE_ONLY|BLOCKED_COMPUTE|
|eps.fig2d|Figure 2(d)|body §6 5D Student-t|eps_studentt_5d|同上|多元 t，非中心参数 θ_k 1_5|同上|not stated|500|vs budget|TPR, FDR|no|stub|FIGURE_ONLY|NOT_RUN|
|eps.fig3|Figure 3|body §7.1 Jester|eps_jester|Uniform, Greedy, e-PS|K=140 笑话，评分 [−10,10]。每笑话 50/50 划分：前半标记均值是否 <0，后半有放回抽样|α=0.1；500 次重洗。betting λ_{k,ℓ}=√[2 log(2/α)/(ℓ log(ℓ+1))]|not stated|500|vs sampling budget|TPR, FDR|no|stub|FIGURE_ONLY。公开数据页存在，仓库内无 CSV|BLOCKED_NO_DATA|
|eps.fig4|Figure 4|body §7.2 OPT-1.3B watermark|eps_watermark_opt13b|同上|五池 × 1000 条 OPT-1.3B 流（200 watermark / 800 无），20 prompt，温度 [0.5,0.8] 五档，长度 200–1200。每 trial 无放回抽 K=500|α=0.1；5×100=500 实验。λ=0.25；v 来自 hold-out|not stated|500|vs token-reads|TPR, FDR|no|stub；`--run` 写 `BLOCKED_NO_DATA`|FIGURE_ONLY。预生成流未发布，禁止用别的模型代替|BLOCKED_NO_DATA|
|eps.fig5|Figure 5|App D UCB vs e-PS 合成|eps_ucb_synthetic|e-PS, UCB|同 Fig 2 四族。**200 trials**（不是 500）|其余同 Fig 2。UCB：log E / n + √(2v log n / n)|not stated|**200**|vs budget|TPR, FDR|no|stub|FIGURE_ONLY|NOT_RUN|
|eps.fig6|Figure 6|App D UCB on Jester+watermark|eps_ucb_real|e-PS, UCB|同 Fig 3–4|500 reps。watermark 半边无数据则阻塞|not stated|500|vs n / token-reads|TPR, FDR|no|stub|FIGURE_ONLY|NOT_RUN|
|eps.fig7|Figure 7|App D two-group|eps_twogroup|e-PS, Uniform, Greedy, fixed-horizon BH|K=200，p=0.8，F0=N(0,I_d)，F1=N(0.5 1_d,I_d)，d∈{1,5,10}|α=0.05；**200 trials**|not stated|200|vs sample size|TPR, FDR|no|stub|FIGURE_ONLY|NOT_RUN|
|eps.fig8|Figure 8|App D mean detection|eps_meandetection|同上|K=200。Unif([μ±0.1]) 与 Unif([μ±1])。20 个非空均值 {0.1,…,2} 每 trial 重抽。B=5，λ=0.99，e=1+(λ/B)Y I{A=k}|α=0.05；200 trials|not stated|200|vs sample size|TPR, FDR|no|stub|FIGURE_ONLY|NOT_RUN|

---

## 5. 接线与命令

只接线现有 runner，不改算法公式。

```bash
cd /Users/grantwright/Desktop/科研/10.30新课题/v0
export PYTHONPATH="$PWD"
.venv/bin/python idea_C/scripts/run_v0_1.py --help
.venv/bin/python idea_C/scripts/run_v0_1.py --list
.venv/bin/python idea_C/scripts/run_v0_1.py --paper score --figure 1 --scale smoke
.venv/bin/python idea_C/scripts/verify_paper_results.py
```

`--scale paper` 必须同时给 `--paper-scale`，否则退出码 2。`--mode official` 一律报错。`--run` 才写 `idea_C/results/v0.1/raw/<run_id>/`（失败保留）。默认不跑 200/500。checkpoint `--resume` 跳过已完成格子。updater 只追加，不改 v0 旧行。

本清单写完时所有对象 `NOT_RUN`。v0 的 30-seed 不迁入本表当复现。2026-11-02 `match-p1-push` 只把下表 status 列改成真实结果：`INDEPENDENT_PARTIAL_MATCH=7`，`BLOCKED_COMPUTE=2`，`BLOCKED_NO_DATA=4`，`NOT_RUN=30`。无 `OFFICIAL_*`，无 `INDEPENDENT_FULL_MATCH`，无编造 PASS。
