# Idea C v0.1 协议修复（相对 v0）

日期：2026-11-02。工作副本 `v0`，分支 `idea-c-v0.1`。本文件只记录 **fix-protocol** 相对 v0 改了什么。未跑 paper-scale 200/500 Monte Carlo，未改 `idea_C/results/v0/raw/**`，未改 `v0实验版本+结果.md` 旧行，未写 `run_v0_1.py`。

实现模式：`INDEPENDENT_REIMPLEMENTATION`。官方代码审计见 `OFFICIAL_CODE_AUDIT.md`。

## 1. Samples@Target 右删失

v0 用 `mean_std_ci` 丢掉非有限值，等于只在达标 seed 上取均值；未达则整格变成 `NOT_RUN`。

v0.1：

- `summarize_samples_at_target` 对每个方法同时报：target TPR、`n_trials`、`n_reached`、`success_rate`、成功条件均值、可识别时的中位达标时间、右删失率、`max_horizon`。
- 未达标记 `CENSORED`，不删、不填 0、不当成 `NOT_RUN`。
- 成功率不同时 `can_rank_by_conditional_mean` 为假，禁止只用条件均值排名。
- 部分达标时 `Samples@Target TPR↓` 单元格写 `CENSORED`，条件均值单独列出。

历史 v0 数字行保持原样，包括 P0 pilot 的 3520.5（那是成功 seed 条件均值）。

## 2. AAAI burn-in

重读 AAAI HTML / DOI 摘录。Algorithm 2：`B(δ,ε)=⌈4ε^{-1}log(4K/δ²)⌉` 是**每臂**长度。`K=50, δ=0.05, ε=0.1` 时 `B=452`，`KB=22600`。**没有缩短 B**。

类型稿把 Update / e-BH 写在 `t>KB` 里面；实现仍每步更新（v0 已有 deviation），否则 Figure 1 在 burn-in 内 TPR 恒为 0。

新配置 horizon **严格大于 KB**，并记录 post-burn-in 各臂抽样次数与频率。预算耗尽仍停在均匀 burn-in、且还有未拒绝臂 → `INVALID_PROTOCOL`，不得进入方法排名。Uniform 不受这条约束。

| 配置 | B | KB | smoke | pilot | paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| `aaai_robust_gaussian_v0_1.json` | 452 | 22600 | 22601 | 30000 | 80000 |
| `aaai_heavy_tailed_v0_1.json` | 452 | 22600 | 22601 | 30000 | 80000 |
| `p1a_clean_post_burnin.json` | 452 | 22600 | 22601 | 30000 | （无，非 paper-scale） |
| `p1a_contaminated_post_burnin.json` | 452 | 22600 | 22601 | 30000 | （无，非 paper-scale） |

Figure 1 横轴在 HTML 摘录里没有数字。paper=80000 沿用 v0 的 paper 槽，且 `80000>22600`。Theorem 8 在 `|H1|=2` 的上界约 25246，pilot=30000 同时超过 KB 与该上界。

**保留的历史 v0 行：** P0 horizon=4000，P1-A horizon=22600=`KB`。对应旧配置未删。

## 3. Lévy α / scale

正文只写 location 1000、skewness 0.5、污染概率 1/10。附录在已读到的 HTML/DOI 摘录里仍未给出稳定性指数与尺度。

继续用 `1.5 / 1.0`，并加显式 `unspecified_in_paper=True`（`LEVY_UNSPECIFIED_IN_PAPER`）。依赖这两参数的 Figure 对照只能是 `INDEPENDENT_PARTIAL_MATCH`，不能声称与作者一致。

## 4. SCORE+

对照 ar5iv:2601.20386 Proposition 5 / Theorem 3：

- `α_1=ω_1 α`（SAFFRON 再乘 `1-λ_1`）
- `α_t=ω_t (R_{t-1}∨1) W_t`
- `W_t=α-∑_{j<t}(α_j-O_j)_+/(R_{t-1}∨1)`
- `O=(αe-1)_+`，`C=(α-O)_+`

逐步手工账与公式辅助函数一致，**没有改公式**。v0 在 `π_1=0.3` pilot 上 SCORE+-LORD Power≈0.425 vs e-LORD≈0.083（约 5 倍）看起来像回溯分母的预期激进性，不是这次手工测试抓到的实现错误。Yeast 表 1 的约 1.7 倍是另一套数据。**paper-scale 500 仍未验证，本轮不跑。**

无 `assume_cpqd`：`status='N/A'` 且 `protocol_status='INVALID_PROTOCOL'`，`fdr_guarantee=False`，不得声称 FDR 保证。
