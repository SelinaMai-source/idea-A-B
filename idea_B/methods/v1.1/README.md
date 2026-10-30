# Idea B v1.1 — 反馈诱导的可观测性坍缩（方法修正）

源码在本目录 `idea_B/methods/v1.1/`。Python 导入名是 `idea_B.methods.v1_1`（目录名含点，不能当包名）。`idea_B.methods` 仍然指向 **v1.0**，行为不变。

## 分配规则（频率不均时必须能被测试分开）

- **Uniform**：逐样本 `q_i = b`。
- **B-v1.1a / EqualSkill**：每个出现技能的**期望审计条数**相等。`q_i ∝ 1/n_{skill(i)}`，再截到 `[0,1]`，最后把总期望审计数拉回预算。稀有技能的逐样本 `q` 更高。这与 v1.0 的 EqualSkill（对出现技能赋同一个 `q=b`，因而与 Uniform 相同）不同。
- **Proportional-to-frequency**：`q_i ∝ n_{skill(i)}`，再拉回预算。常见技能的逐样本 `q` 更高。第三条规则。

## 控制器

| 名称 | 作用 | 新颖性 |
| --- | --- | --- |
| B-v1.0 | 原样包装 v1.0 `ObservabilityController`（含 smoothing=0.6 与 mix_uniform=0.1） | 参照臂 |
| B-v1.1a | 修正后的等技能期望条数 | 正确性修复 |
| B-v1.1b | E-optimal：在预算下提高条件设计 Gramian 的 `λ_min` | **经典对照，不是新算法** |
| D-optimal | 提高 `logdet` / 预测方差 | **经典对照，不是新算法** |
| B-v1.1c | 折扣陈旧信息 + 一步预期降低 worst-skill MSE | 主提案 |
| Uniform / Greedy / Lazy / Oracle | 通用对照 | 无 |

B-v1.1c 若在冻结协议上不优于 E-optimal，结论是 NOT PROMISING 或 INCONCLUSIVE。E-optimal 赢了不算新方法成功。

## 信息量

保留 `W = Σ S φ φ^T`。报告时分开：

1. 已实现观测信息（realized `S`）
2. 条件期望信息 `Σ q φ φ^T`
3. 全体候选样本信息 `Σ φ φ^T`（不删失）

另报滑动窗口的 `λ_min`、条件数、数值秩、技能方向信息。更高的特征值单独出现时**不是**学习成功。

## 主指标

不可约噪声底 `noise_std^2 = 0.0225`（`noise_std = 0.15`）必须报告。

主指标：excess MSE（相对噪声底）、worst-skill MSE、population-weighted MSE、parameter error、cumulative regret、Env3 recovery time。`1/(1+MSE)` 只作次要描述。

## 环境（新目录，不改 v1 `Synthetic_Closed_Loop`）

- `idea_B/benchmarks/B_Env1_Endogenous_Selection/`：真闭环。初值让多数技能已经拟合得很好，弱技能误差还有变差空间。
- `idea_B/benchmarks/B_Env2_Observational_Equivalence/`：两个潜变量在当前策略观测支撑上似然相同或接近，在未观测技能上不同。识别下界用显式高斯似然和 Fisher 信息，不用图。
- `idea_B/benchmarks/B_Env3_Recovery_After_Latent_Change/`：回合中点（horizon 32 时 `t=16`）按种子预先写死，与算法表现无关。可恢复与不可恢复各一条。

## 文件

- `observability.py`：三种 Gramian、滑动窗口、秩、技能信息。
- `information_controller.py`：分配规则与控制器。
- `feedback_dynamics.py`：观测合成、主指标、恢复时间、CUSUM 检测、坍缩判定。
- `policy_update.py`：ridge / SGD。
- `identification_test.py`：Env2 的似然与矩条件。
- `algorithm.py`：闭环 runner。
- `config.yaml`：冻结超参。写入之后不因评价集指标高低而修改。
