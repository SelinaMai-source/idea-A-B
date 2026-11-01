# Idea C v0 BASELINE_AUDIT

审计日期：2026-11-01。只记录论文/附录与公开代码现状。未实现算法，未跑 Monte Carlo。UAI **不是** CCF-A Main。e-PS **不是**已发表 CCF-A。

| 论文 | Venue | 官方源 | 代码状态 | v0 标签 |
| --- | --- | --- | --- | --- |
| On the Robustness of Bandit Multiple Testing (Zhou & Liu) | AAAI 2026 Main | AAAI PDF https://ojs.aaai.org/index.php/AAAI/article/download/40148/44109 ；未找到配套 GitHub | 无官方实现。`neilzxu/e_bmt` 是 NeurIPS 2021 *Bandit Multiple Testing*，**不是**这篇 | `independent_reimplementation` |
| SCORE (Kuang, Gang, Xia) | ICML 2026 Main | PMLR PDF https://raw.githubusercontent.com/mlresearch/v306/main/assets/kuang26c/kuang26c.pdf ；arXiv:2601.20386 | 论文页 / GitHub 检索无官方仓 | `independent_reimplementation` |
| Improving online FDR procedures via online analogs of e-closure and compound e-values (Xu, Fischer, Ramdas) | UAI 2026（**非 CCF-A Main**） | PMLR https://proceedings.mlr.press/v337/xu26a.html | 见下：相关官方仓已浅克隆，但**不是**这篇 UAI 在线算法的专用发布 | UAI 在线方法 `independent_reimplementation`；离线 e-Closure / AISTATS e-LOND 仅作相关代码 |
| Sample-Efficient Multiple Testing with Adaptive Data Collection / e-PS (Lin, Ma, Ren, Wei) | arXiv:2609.26651 **预印本** | https://arxiv.org/abs/2609.26651 | 无官方代码 | `independent_reimplementation`；跳过 LLM watermark |

## 1. AAAI 2026 Robust Bandit Multiple Testing

- **Official source:** AAAI proceedings PDF only. Author emails `{zzysince1999, liuweiwei863}@gmail.com`。检索未找到论文配套仓库。武汉大学 GitHub 用户 `zhouliu` 的公开仓与本工作无关。
- **Not official:** `neilzxu/e_bmt`（NeurIPS 2021 Xu, Wang, Ramdas）。禁止当 2026 鲁棒方法的官方代码。
- **Code status:** `independent_reimplementation`。对照正文 Algorithm 1–3 与 Figure 1/2/3 公式，不对照 e-BMT 2021。
- **Benchmarks (from paper, not from memory):**
  - Figure 1/2：\(P_i=N(\mu_i,1)\)，null \(\mu=0\)，non-null \(\mu=1\)。Lévy-stable 污染概率 **1/10**，location **1000**，skewness **0.5**。\(\delta=0.05\)。\(|H_1|\in\{2,\lfloor\log K\rfloor,\lfloor\sqrt{K}\rfloor\}\)。对照 Uniform / Mean-based / E-process-based。原文 **500 trials**。Figure 1 的 K 在 HTML 摘录中未给出数字；v0 config 暂用 \(K=50\) 并在表前标明。
  - Figure 3 heavy-tail：Student-t(2)，无限方差，污染同上，null mean 0 / non-null 5，\(K=50\)，\(|H_1|=2\)，\(p=1.85\)，\(\kappa=50\)，500 trials。指标以 TPR 曲线为主，不机械复制 Gaussian 表头。
  - Figure 4 multi-agent：\(K=10\)，\(|H_1|=2\)。v0 默认 `NOT_RUN`。
- **Known issues:**
  - 无参考实现，trimmed-mean 的 \(\alpha\)、e-process \(\lambda=\sqrt{\varepsilon}/(2\sigma)\) 必须从正文再对一次再写代码。
  - 污染下非鲁棒 e-process 只允许作为 invalidity stress test。
  - 500 trials 不得用 30-seed 冒充 paper-scale。

## 2. ICML 2026 SCORE

- **Official source:** PMLR v306 PDF / arXiv:2601.20386。ICML poster 页无代码链接。
- **Code status:** 未找到官方仓 → `independent_reimplementation`。必须实现完整 alpha-wealth / overshoot：\(C=(C-O)_+\)，\(O=(\alpha_j e_j-1)_+\)，以及 SCORE-SAFFRON 的 candidate 收紧。禁止简化成「拒绝后加点 α」。
- **Benchmarks:**
  - §6.1 Independent Gaussian mixture：\(T=1000\)，\(\theta_t\sim\mathrm{Ber}(\pi_1)\)，\(\mu_t\mid\theta_t=(1-\theta_t)\delta_0+\theta_t N(3,5)\)，\(X_t\mid\mu_t\sim N(\mu_t,1)\)。e-value \(e_t=\phi(X_t;3,6)/\phi(X_t;0,1)\)。\(\pi_1\in\{0.3,0.8\}\)，\(\alpha=0.05\)，LORD \(\omega_t\equiv 0.05\)，SAFFRON \(\lambda_t\equiv 0.5\)。**500 repetitions**。方法：e-LOND/LORD/SAFFRON 与 SCORE / SCORE+ 对照。
  - SCORE+ 仅在 CPQD 下使用，否则表中 `N/A`。
  - §6.2 AR exponential、真实数据：v0 默认后置 / `NOT_RUN`。
- **Known issues:**
  - 无官方代码，wealth 更新极易写错。
  - SCORE+ 回溯分母在共享校准样本等场景文中自己也不用。
  - Power 定义必须写在表前：时间点平均 power，以及相对同族 baseline 的 power ratio。

## 3. UAI 2026 e-closure / donation（非 CCF-A Main）

- **Official paper:** PMLR v337 xu26a。错误度量是 **SupFDR** 与 FDR，二者不得混用。
- **Claimed accompanying code:** 正文写有 user-friendly implementation。公开检索结果：
  - https://github.com/neilzxu/eclosure — **已浅克隆**。SHA `8bbd5a37b48abb6d79f5e1eef95d4fe6315339b5`（`release` / `v0.1.1b`，2026-03-23，3.6M）。该仓 README / citation 指向离线论文 *Bringing Closure to False Discovery Rate Control*（arXiv:2509.02517）。Python API：`closedeBH`、`closedBY`。C++ 在 `r/eclosure/src/`。`experiments.py` 含离线 `donation_ebh`，**不是** UAI 的 online donation e-LOND。
  - https://github.com/neilzxu/evalue-omt — **已浅克隆**。SHA `cda6e9962514924044143920cf1cadbec8738d52`（`publishable`，2024-03-08，4.1M）。这是 AISTATS 2024 *Online multiple testing with e-values*。README 实验名 `lag_comp`、`wor_comp`。DeepPurpose **未**初始化。
- **Code status:** 相关官方树已在磁盘且 gitignore + MANIFEST。UAI 2026 的 Closed / Donation e-LOND 仍标 `independent_reimplementation`（或日后若发现匹配仓再改标签）。Mac 若编译官方 C++ 失败，独立 Python 必须写 deviation。
- **Benchmarks (paper Section 4 / Appendix C.3):**
  - Local dependence Gaussian：\(m=200\)，\(\mu_0=0\)，\(\mu_1=3\)，lag \(L=100\)，\(\Sigma_{i,j}=0.5^{|i-j|}\)（\(|i-j|\le L\) 否则 0，需数值确认 PSD）。\(\gamma_t=(t(t+1))^{-1}\)，\(\delta=0.1\)，**200 trials**。方法：e-LOND、Closed e-LOND、Donation e-LOND 及 r-LOND 族。
  - Runtime 网格 \(m\) 至 3000、\(n=80\)：CPU 默认跳过，表中 `NOT_RUN`。
  - NYC taxi：默认 `NOT_RUN`。
- **Known issues:**
  - 仓库错配风险高：离线 e-Closure ≠ UAI 在线 e-closure。
  - `lag_comp` 属于 2024 仓，DGP（rho/lag 的实现）与 UAI 2026 的 \(\Sigma_{ij}=0.5^{|i-j|}\) 不必相同。P0 以 UAI 正文为准。
  - Closed 方法 \(O(t^2)\)，先保证小规模正确再谈大规模 runtime。

## 4. arXiv:2609.26651 e-PS（预印本）

- **Official source:** arXiv HTML/PDF only。
- **Code status:** 无官方代码 → `independent_reimplementation`。
- **Benchmarks (§6):**
  - \(K=50\)。1D Gaussian \(N(\theta_k,1)\) 与 1D Student-t(5)。\(B_k\sim\mathrm{Bern}(0.1)\)，\(\theta_k=0.02k\cdot B_k\)（一次抽好后固定）。simple-vs-simple（Oracle LR）与 simple-vs-composite（Adaptive plug-in）分开。Uniform / Greedy / e-PS。\(\alpha=0.05\)。**500 trials**。e-PS 方差代理 \(v_{k,t}=1.1\hat\sigma_{k,t}^2\)。
  - 跳过：5D、Jester、LLM watermark（OPT-1.3B）。
- **Known issues:**
  - 预印本，venue 不得升级为 CCF-A。
  - nested rejection（e-BH 随时间单调）必须测。
  - 污染协议下若仍用非鲁棒 increment，只能标 stress test / `NOT_COMPARABLE`。

## Clone policy

- 路径：`idea_C/baselines/eclosure/official/` 与 `.../evalue-omt/`。
- 浅克隆 `--depth 1`，未装 extras，未 init 子模块。
- 因带独立 `.git`，与仓库既有 ActiveUltraFeedback 策略一致：**gitignore 克隆树，只提交 MANIFEST**。
- 清单：`idea_C/baselines/eclosure/OFFICIAL_MANIFEST.md`，`EVALUE_OMT_MANIFEST.md`。
