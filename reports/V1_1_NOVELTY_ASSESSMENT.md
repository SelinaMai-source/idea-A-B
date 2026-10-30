# v1.1 Novelty Assessment

本报告只做研究解释。所有数字仍来自 TIER2 / 审计 JSON。

预先规则下的判定：Idea A v1.1 = **NOT PROMISING**，Idea B v1.1 = **NOT PROMISING**。

## Idea A

### 最强支持证据

identification-via-audit。v1.1 静态 MNAR `b=0.05` n=`50`： observed-only RMSE=`1.21739` bias=`1.21647` coverage=`0`； Uniform DR RMSE=`0.128779` bias=`-0.00445322` coverage=`0.96`。 这支持「已知 q 的独立审计 + 正确 cross-fit DR」修复 MNAR observed-only，不支持新分配规则。

### 最强反证

自适应分配失败。静态 v1.1c vs Uniform DR RMSE diff=`0.204919` CI [`-0.0384978`, `0.442027`] `covers_zero`。 闭环 MNAR-audit diff=`0.906352` CI [`0.0565902`, `1.67815`] `entirely_positive`。 A-v1.1b/c 是经典方差分配加启发式，不是新最优性定理。

### 最近的已发表竞争者

AMPLe（ACL 2025 Main，local_status=partial）、HyPER（blocked）、Drago et al. ICML 2026（无代码）。目标、预算与信息都不同。**不能**声称 outperform。

### 可能的新颖性主张

把已知倾向的独立审计接到内生反馈删失下的有限总体识别。这是问题设定，不是新的 Neyman/AIPW 公式。v1.1 的序贯分配目前没有被主对照支持。

### 当前结果尚不支持的部分

不支持：v1.1c 优于 Uniform DR、10% 动态 RMSE 下降、新分配算法、优于 AMPLe/HyPER、可投稿定理、LLM 后训练。

## Idea B

### 最强支持证据

在 Env1 上 v1.1c 相对 E-optimal 的 cumulative regret mean_difference=`-0.0358101` CI [`-0.0647617`, `-0.00962598`] `entirely_negative`。 这只说明累计 excess MSE 路径可以更好，不是 H_B1 成功。

### 最强反证

主指标 worst-skill MSE：Env1 mean_difference=`0.000727427` CI [`6.74253e-05`, `0.0014246`] `entirely_positive`。 H_B1 要求 worst-skill **并且** regret 同时优于 E 与 D。E-optimal 是经典对照；它赢了不算新方法。

### 最近的已发表竞争者

ICML 2024 PP-bandit（complete 的 1-D demo，NOT DIRECTLY COMPARABLE）与 NeurIPS 2024 ncvxpp（partial）。不得接到技能删失环境后称为官方算法。

### 可能的新颖性主张

闭环下用 worst-skill MSE / regret 而不是饱和的 `1/(1+MSE)` 或单独的特征值来评价信息分配。当前实验没有把 B-v1.1c 变成被支持的新控制器。

### 当前结果尚不支持的部分

不支持：v1.1c 优于 E-optimal（主指标）、Env3 恢复时间优势、只抬特征值算学习成功、击败 performative 论文。

## 总评

A v1.1 `NOT PROMISING`；B v1.1 `NOT PROMISING`。 被支持的是审计识别（A）与「必须用 worst-skill 而不是特征值」（B 的评价修正）。不被支持的是两个主提案的序贯分配 / 控制器。

TIER2 A written_count=`240`；B n_runs_attempted=`610`。
