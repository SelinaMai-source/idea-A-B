# v1.2 Benchmark 兼容性

本文件由 `scripts/generate_v1_2_report.py` 从磁盘 JSON 与已锁定审计写出。数字均来自下列产物，没有手填对照 RMSE，也没有发明 10/30-seed CCF-A head-to-head。

- 生成时间（UTC）：`2026-10-30T20:17:05.317452+00:00`
- 仓库：`/Users/grantwright/Desktop/科研/10.30新课题/v0`
- 分支：`v1.2`
- commit：`fd00badccc8f1e4a8012b91c2b9d473914f4f3c5`
- remote：`git@github.com:SelinaMai-source/idea-A-B.git`
- 缺失的主源：无
- 可比性锁：`reports/v1.2/03_COMPARABILITY_AUDIT.md`
- 主判定：`PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`
- scientifically valid DIRECT 对照数：`0`；可辩护 ADAPTED 对照数：`0`

本文件建立在 `03_COMPARABILITY_AUDIT.md` 的锁上，不改口。

## 锁

- `PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`
- DIRECT = `0`，ADAPTED = `0`
- Protocol A2 = `BLOCKED_INCOMPARABLE`
- Protocol B2 = `INCOMPARABLE`

## 目标兼容性

| Cell | 发表目标 | 我们的目标 | 判定 |
| --- | --- | --- | --- |
| C-AMPLe-A | 单纯形偏好向量 | 标量泛函 RMSE | INCOMPARABLE |
| C-HyPER-A | human/LM 路由 + RewardBench | 审计泛函 | INCOMPARABLE |
| C-Drago-A | 全局兼容奖励识别 | 有限预算审计 | INCOMPARABLE |
| C-ActiveUF-full-A | RM/DPO/judge 偏好数据 | 总体 RMSE | INCOMPARABLE |
| C-A2-Av11-ActiveUF | 选回复对 | Bernoulli 审计概率 | INCOMPARABLE |
| C-UCBOGD-vision | 多源稳健视觉/VQA | 审计 / 技能 MSE | INCOMPARABLE |
| C-PPbandit-B | 一维 performative risk | worst-skill MSE | INCOMPARABLE |
| C-B2-Bv11-ZO | 一维两点零阶 | 技能分配 | INCOMPARABLE |
| C-ncvxpp-B | performative SGD loss/acc | observability Gramian | INCOMPARABLE |

## 信息访问

- ActiveUF 选择器需要 live RM UQ 张量。本仓库没有。HF `combined` 成对列缺少 `lower_bounds` / `upper_bounds` 与落选回复，offline replay 不成立。judge 分数不是人类偏好。
- A-v1.1 看见协变量、已知 `q`、已审计残差；不看见隐藏 `Y`。
- PP-bandit 看见完整标量 PR；没有个体特征或技能标签。
- 若给 B-v1.1 技能标签而官方方法没有，信息不对齐。

## 预算会计

审计比例 `b`、偏好对、查询轮数、图像张数、SGD 迭代不是同一列。burn-in 50000 不并进 `b`。

## 适配细节

- Protocol A2 适配器：**未实现**。映射 M1–M5 否决。
- Protocol B2 部分反馈适配器：**未实现**。原环境无分配动作。
- `volume_adapted_ipw` 仍是合成启发式，不叫 AMPLe。
- Uncertainty Sampling 保持原名。
- selector_unit_check 允许，但是 DIAGNOSTIC，不升格 ADAPTED。
- UCB-OGD logistic DIAGNOSTIC：**未实例化**。即使实例化也不能宣称击败论文。

完整 13 维表见 `03_COMPARABILITY_AUDIT.md` 第 3 节。

