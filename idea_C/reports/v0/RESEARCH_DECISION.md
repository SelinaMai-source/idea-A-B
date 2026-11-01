# RESEARCH_DECISION

判定日：2026-11-01。数字只来自 `idea_C/results/v0/raw/*/metrics.csv` 与已对账的 `v0实验版本+结果.md`。C1/C2/C3 仍是**假想方向**，不是已实现方法。缺证据 = No-Go。 **不开 C-v1。**

30-seed 是 `pilot`，不是 200/500 paper-scale。P0 AAAI 短 horizon 不是方法排名。污染下非鲁棒 FDR=0.96 是 stress test，不是定理反例。

规范副本在本文件。仓库根下若出现 `idea_C/RESEARCH_DECISION.md`，只做指针。

---

## C1（假想：鲁棒自适应抽样 + e-BH，对齐 AAAI / e-PS 接口）— **No-Go**

缺 post-burn-in、缺 paper-scale、缺「鲁棒 increment + 自适应抽样」的合法对照。

- P0 AAAI Robust Gaussian pilot（horizon=4000，K=50）：Uniform 与 E-process TPR 都是 **0.2**、FDR=0、Samples@Target=3520.5；Mean-based TPR=**0.0333**。horizon 低于/贴着 burn-in，不能排名。
- P1-A clean horizon=**22600=KB**：代码对 \(t\le KB\) 均匀抽样，Uniform 与 E-process **同为 TPR=0.05、FDR=0**。这是协议坍缩，不是方法结论。
- P1-A clean 上 e-PS Oracle TPR=**1.0**、FDR=0.0333、Samples@Target=345.7（假设备择 μ=1，**adapted**，不是原文 0.02k）。不能据此说「已经做出鲁棒 e-PS」。
- 污染下 Greedy / e-PS 的 FDR=**0.960**、TPR=1.0，标 `NOT_COMPARABLE`。稳健 Uniform / E-process 的 FDR=0、TPR=1.0、Samples=6779.8（两行相同）。stress test ≠ 新定理。
- AAAI Fig 3 heavy-tail、paper-scale 500：`NOT_RUN`。

## C2（假想：overshoot / donation / closed 的财富再分配统一）— **No-Go**

有同族功率差，但没有 paper-scale，也没有在同一错误度量下证明「新统一规则」。

- SCORE §6.1 π₁=0.3 pilot：SCORE+-LORD Power=**0.425** vs e-LORD **0.0835**；SCORE-SAFFRON **0.122** vs e-SAFFRON **0.0988**；SCORE-LOND **0.205** vs e-LOND **0.203**。FDR 均 ≪ 0.05。这是独立复现的方向性信号，**30 ≠ 500**。
- UAI local-dep δ=0.1 pilot：Closed e-LOND Power=**0.0862**、Donation e-LOND **0.0624**、e-LOND **0.0583**，SupFDR 均 0.00667。Closed r-LOND SupFDR=**0.135>0.1**，但是独立实现 + \(E=1/P\)，不能当反例，更不能当新方法。
- P1-B 只有 smoke 3-seed：Closed e-LOND 与 Donation e-LOND Power 同为 **0.205**（Runtime 241s vs 0.41s）；SCORE+-LORD Power=**0.405**。P1-B 30-seed `NOT_RUN`（Closed 估 ~40min，CPU 不足）。
- UAI **不是** CCF-A Main。paper-scale 200/500 `NOT_RUN`。

## C3（假想：跨 multi-arm / streaming 的共同协议新方法）— **No-Go**

没有实现任何 C-v1 规则，也没有 paper-scale 共同协议。

- P1-A 只比较了已有实现；E-process 未出 burn-in，Mean-based 未完成 t(1) 初始化。
- P1-B 30-seed 缺失；NYC Taxi / Jester / watermark / multi-agent / 5D 全部 `NOT_RUN`。
- 缺证据不得声称「统一框架」。

## C-v1

**No-Go。** v0 停在忠实复现 + 两个 common protocol 的 smoke/pilot。下一步若做，必须先补：AAAI post-burn-in、e-PS/SCORE/UAI paper-scale、P1-B 30-seed，以及把 stress test 从方法表里隔开。
