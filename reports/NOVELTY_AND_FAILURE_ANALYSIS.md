# Novelty and Failure Analysis

本报告只做研究解释，不做工程清单。所有数字仍来自同一批 JSON。

预先规则下的判定：Idea A = **INCONCLUSIVE**，Idea B = **INCONCLUSIVE**。`n_seeds=3` 为初步证据。

## Idea A

### 最强支持证据

MNAR 下 observed-only 对总体均值的 bias mean = `1.1897` （std `0.0424772`），审计 IPW 把偏差降到 Neyman `-0.098014` / Uniform `-0.104575`。 naive IPS 仍为 `0.590411`。 这说明：偏差来自自然通道的 MNAR，而不是「随便一个 IPS 公式」。识别锚是**独立审计 + 已知 q**。

### 最强反证

同等预算下 Neyman 并不优于 Uniform。MNAR `b=0.05` 配对 RMSE 差 `0.0421637`， CI [`-0.0685416`, `0.161748`] 含 0。 zero-overlap `b=0.05` 的区间排除 0，但方向是 Neyman 更差。 因此「自适应分配」不是被支持的新颖点；被支持的是审计通道本身。

### 最近的已发表竞争者

AMPLe（ACL 2025 Main）用 volume acquisition 做多维偏好识别；HyPER（ACL 2025 Main）做人机标注路由；Drago et al.（ICML 2026）做 globally compatible reward identification。目标、预算与信息都不同。本机 AMPLe 仅 partial，HyPER/Drago blocked。**不能**声称 outperform 这些论文。

### 可能的新颖性主张

把「已知倾向的独立审计」当作内生反馈删失下偏好/奖励泛函的识别锚。这是应用层面的问题设定，不是新的 Neyman 公式。

### 该差异为何可能重要

若生产系统只能看到自选反馈，observed-only 会给出系统性偏差（本合成 MNAR 约 +1.19）。一条小预算、倾向已知的审计通道在原理上可以修复识别。这对数据治理有意义。

### 当前结果尚不支持的部分

不支持：新分配算法、优于 Uniform 的 RMSE、优于 AMPLe/HyPER、可投稿的理论定理、LLM 后训练结果。v1.0 必须继续被称为经典 Neyman。

### 还需要的工作

理论：MNAR 自然通道 + 独立审计下的有限样本保证。实验：更多种子；真实 MultiPref 上只把 audit 用在 human/expert 标签，不把三类标注合成单一真值；不要把 volume 适配器叫做 AMPLe。

## Idea B

### 最强支持证据

同预算下控制器的 late λ_min 高于 Uniform：mean difference `1.23032`， CI [`1.1027`, `1.30783`] 排除 0。 Greedy/NoFeedback 的弱技能最终观测率为 `0`。 这表明按 `W_T` 弱方向分配，确实能改变看见了哪些技能。

### 最强反证

`endogenous_collapse=false`，联合判据 `false`。 观测表现稳定、λ_min 下降，但弱技能误差没有继续恶化。 真实表现配对 CI [`-7.73095e-05`, `0.00114084`] 含 0。 JSON `beats_uniform_on_both=true` 不能覆盖这一区间规则。

### 最近的已发表竞争者

ICML 2024 PP-bandit（官方数值实验本机 complete，但是 1-D reparameterized ZO demo）与 NeurIPS 2024 ncvxpp（partial）。它们研究 performative risk，不研究技能级观测 Gramian。适配 PP-bandit 到 B2 被标 BLOCKED，以免伪造「官方算法接到新环境」。

### 可能的新颖性主张

闭环策略使某些技能的反馈消失，观测指标仍可看起来很好，而 `λ_min(W_T/T)` 暴露不可识别方向。若该现象稳定出现，它不同于普通 catastrophic forgetting。

### 该差异为何可能重要

若只监控观测准确率，系统可能在弱技能上停止学习却仍显示「表现稳定」。信息矩阵提供了一个与奖励不同的诊断。

### 当前结果尚不支持的部分

不支持：已证实的内生坍缩、控制器改善长期真实表现、击败 performative 论文、把 B1 叫做策略诱发坍缩、任意矩阵都叫 observability Gramian。

### 还需要的工作

换初始化与更长 horizon，使弱技能误差有「先低后高」的空间；把信息改善与真实表现分开检验；理论需要策略依赖观测算子的不可识别条件，而不是再发明一个 Gram 矩阵名字。

## 总评

两个想法目前都停在 INCONCLUSIVE。A 的识别现象比分配规则更值得跟；B 的信息诊断比「已经打赢长期奖励」更值得跟。都还不够成为 CCF-A Main 投稿主张。
