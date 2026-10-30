# Idea B v1.0 — Feedback-Induced Observability Collapse

控制器估计技能方向上的信息，找出当前观测很少的方向，把有限的探索/审计预算分过去，再更新线性模型参数。分配在技能空间上做平滑，避免每一轮剧烈切换。

## 信息量

线性模型的**观测设计 Gram 矩阵**

\[
W_T=\sum_t S_t\phi_t\phi_t^\top
\]

以及条件期望版本 \(W_T^{\mathrm{cond}}=\sum_t q_t\phi_t\phi_t^\top\)，其中 \(q_t=\mathbb{E}[S_t\mid\phi_t,\text{policy}_t]\)。不要把任意矩阵叫做可观性 Gramian。

## 文件

- `observability.py`：构造 \(W_T\)、特征值、\(\lambda_{\min}\)、条件数、技能方向信息。
- `feedback_controller.py`：弱方向识别与预算分配；也实现通用对照的分配规则。
- `policy_update.py`：ridge 与 SGD。
- `algorithm.py`：闭环 runner 与坍缩判定。
- `config.yaml`：默认短时域烟测配置。

## 科学约束

- 至少 B2 是真闭环：策略 → 反馈分布 → 观测数据 → 更新 → 策略。
- 外生缺失（B1）和“固定策略 + 固定抽样”不得标成内生坍缩。
- 信息更好不自动等于真实表现更好，两项分开记录并检验。
