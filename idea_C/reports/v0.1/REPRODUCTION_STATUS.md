# REPRODUCTION_STATUS (v0.1)

日期：2026-11-02。`match-p1-push`。实现模式：`INDEPENDENT_REIMPLEMENTATION`。禁止 `OFFICIAL_*`。UAI **不是** CCF-A Main。e-PS **是预印本**。30-seed 只标 pilot。P1 已跑（Overlap A/B，`ADAPTED_PROTOCOL` / `C-v0.1_RERUN`）。

| Paper | Venue | Official repo | Objects | Ran this round | Status mix |
| --- | --- | --- | --- | --- | --- |
| AAAI Zhou & Liu | AAAI 2026 Main | none | 5 | Fig1+Fig3 paper 80000×500；P1-A post-burn-in 3-seed H=22601 + 30-seed H=30000 | Fig1/Fig3 `INDEPENDENT_PARTIAL_MATCH`（FIGURE_ONLY）；Fig2/Fig4/app `NOT_RUN` |
| SCORE Kuang/Gang/Xia | ICML 2026 Main | none | 16 | Fig1–2 / Table2 α=0.05 paper 500；P1-B 3-seed + 30-seed（Closed 30-seed skipped） | Table2 α=0.05 `INDEPENDENT_PARTIAL_MATCH`；Yeast/MI `BLOCKED_NO_DATA`；AR `NOT_RUN` |
| UAI Xu/Fischer/Ramdas | UAI 2026 **非 CCF-A Main** | none (INCOMPLETE) | 11 | Fig2 power 200 @ m=200 π1=0.3；donation/e-LOND runtime m=3000 n=80 | Closed m=3000 `BLOCKED_COMPUTE`；NYC `NOT_RUN`（CSV 已下载） |
| e-PS Lin/Ma/Ren/Wei | arXiv:2609.26651 **预印本** | none | 11 | Fig2a/2c smoke 3×H=400；P1-A 共用协议里的 e-PS（adapted μ=1，不是论文 θ） | paper 500 `BLOCKED_COMPUTE`；Jester/watermark `BLOCKED_NO_DATA` |

## 对象级计数（43 checklist objects）

- `OFFICIAL_*` = 0
- `INDEPENDENT_FULL_MATCH` = 0
- `INDEPENDENT_PARTIAL_MATCH` = 7（aaai.fig1, aaai.fig3, score.fig1, score.fig2, score.tab2, uai.fig2_power, uai.fig2_runtime）
- `FAIL_*` = 0
- `BLOCKED_COMPUTE` = 2（eps.fig2a, eps.fig2c）
- `BLOCKED_NO_DATA` = 4（score.tab1, score.fig13, eps.fig3, eps.fig4）
- `NOT_RUN` = 30

## 格级计数（verify 304 rows）

`INDEPENDENT_PARTIAL_MATCH=32`（SCORE Table2 α=0.05 全部格），`BLOCKED_NO_DATA=6`，`FIGURE_ONLY=38`，`NOT_RUN=228`，`OFFICIAL_*=0`，`FAIL_*=0`。

对照图：`idea_C/results/v0.1/figures/`。总表已追加 `C-v0.1_*` / `C-v0.1_RERUN_*`，未改 v0 旧行。
