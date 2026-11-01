# COMMON_PROTOCOL_AUDIT

填写日期：2026-11-01。P1-A（Overlap A）已跑 smoke 3-seed 与 pilot 30-seed。30-seed **不是** paper-scale。未跑 P1-B 本切片以外内容。

## Shared protocol

- **Arms:** \(P_i=N(\mu_i,1)\)，null \(\mu=0\)，non-null \(\mu=1\)，\(K=50\)，\(|H_1|=2\)（与 P0 AAAI config / BASELINE_AUDIT 一致，未改 K/α）。
- **Budget:** Clean 与 Contaminated、全部方法同一 horizon **22600**（Algorithm 2 \(KB=K\cdot B(\delta,\varepsilon)\)，\(B=\lceil 4\varepsilon^{-1}\log(4K/\delta^2)\rceil=452\)）。
- **Eval time:** 预算用尽（或全部拒绝后提前停）的终态拒绝集。
- **α / δ:** \(0.05\)。
- **AAAI 算法 ε:** \(0.1\)（论文稳健参数）。Clean **数据** \(\varepsilon=0\)；Contaminated **数据** \(\varepsilon=1/10\)。
- **Seeds:** smoke `[0,1,2]`；pilot `list(range(30))`。`seed_everything(seed)`。

## Methods (existing implementations only)

| 方法 | 实现 | Clean | Contaminated |
| --- | --- | --- | --- |
| Uniform | `robust_bandit` Uniform + RobustEProcess | 可对照 | 可对照（稳健） |
| Greedy | `eps` Greedy sampler + Adaptive plug-in | 可对照 | **stress test / NOT_COMPARABLE** |
| e-PS Oracle LR | `eps` sampler + OracleLR；假设备择改为 μ=1（adapted，不是 0.02k） | 可对照（adapted） | **stress test / NOT_COMPARABLE** |
| e-PS Adaptive | `eps` sampler + AdaptivePluginLR | 可对照 | **stress test / NOT_COMPARABLE** |
| AAAI Mean-based | Algorithm 1 | 可对照；t(1) 未完成 | 可对照（稳健）；t(1) 未完成 |
| AAAI E-process-based | Algorithm 2 | 可对照；未出 burn-in | 可对照（稳健）；未出 burn-in |

未发明 C-v1 方法。共享 e-BH。

## Deviations (must stay in 结果解释)

1. Lévy-stable 稳定性指数/尺度原文未给：代码 `alpha=1.5`、`scale=1.0`。
2. e-PS Oracle 假设备择改为共同协议 μ=1，不是 e-PS 论文 θ_alt=0.02k。
3. Greedy 配对 Adaptive（已有实现），不是新算法。
4. horizon=KB：`t<=KB` 全程均匀，Uniform 与 E-process-based 抽样不可分。不得宣称胜出。
5. Mean-based `t(1)≈3240`/臂，22600 只能初始化约 7 臂。
6. Algorithm 2 每步更新 e-process / e-BH（与 P0 相同 typesetting 偏差）。
7. 污染下非鲁棒 increment 只作 invalidity stress test，**不得**当作推翻定理的证据。

## Streaming (P1-B)

P1-B smoke 3-seed 已跑（Closed e-LOND 墙钟约 241s）。pilot 30-seed 因 CPU 不足保持 `NOT_RUN`，禁止填 0。FDR 与 SupFDR 仍分列。SCORE+ 只在 CPQD 下有数字。
