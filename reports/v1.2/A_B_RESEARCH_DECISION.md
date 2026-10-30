# v1.2 Idea A / Idea B 研究决策

本文件由 `scripts/generate_v1_2_report.py` 从磁盘 JSON 与已锁定审计写出。数字均来自下列产物，没有手填对照 RMSE，也没有发明 10/30-seed CCF-A head-to-head。

- 生成时间（UTC）：`2026-10-30T20:17:05.351328+00:00`
- 仓库：`/Users/grantwright/Desktop/科研/10.30新课题/v0`
- 分支：`v1.2`
- commit：`fd00badccc8f1e4a8012b91c2b9d473914f4f3c5`
- remote：`git@github.com:SelinaMai-source/idea-A-B.git`
- 缺失的主源：无
- 可比性锁：`reports/v1.2/03_COMPARABILITY_AUDIT.md`
- 主判定：`PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`
- scientifically valid DIRECT 对照数：`0`；可辩护 ADAPTED 对照数：`0`

PROMISING 不能表示只打败均匀随机。本文件也不把内部经典对照改写成发表基线胜利。

总判定：`PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`。v1.3 算法迭代 justified = `false`。

## Idea A

### Current evidence / 当前证据

- 已发表对照：Protocol A2 **BLOCKED / INCOMPARABLE**。无 DIRECT，无可辩护 ADAPTED。
- 选择器 unit check 只说明隔离 venv 能调用官方 DRTS / DeltaUCB / InfoMax。
- v1.1 内部：静态 MNAR `b=0.05` n=`50`，A-v1.1c RMSE `0.3336984041151296` vs Uniform DR `0.1287791571117902`，`interval_side=covers_zero`。闭环 n=`30`：`1.074943643456929` vs `0.1685912838364942`，`interval_side=entirely_positive`。判定 **NOT PROMISING**。
- identification-via-audit 在 v1.1 静态格上仍成立（observed-only 差于 Uniform DR）。那是审计通道，不是主动选择器胜利。

### Strongest published competitor / 最强已发表竞争者

名义上最接近的是 ActiveUltraFeedback 的 **DRTS** 与 **DeltaUCB**（ICML 2026 Main）。InfoMax 是同一仓库里的已有 dueling-bandit 对照，不是该文新算法。AMPLe / HyPER 目标更远。

### Strongest valid experimental result / 最强合法实验结果

相对这些发表方法：**没有合法实验结果。** 最强原设定产物是 AMPLe 3 轮 volume 的 L2（见 ORIGINAL 报告），以及 selector_unit_check 的合成下标；两者都不是 A-v1.1 vs 发表方法。

### Most serious limitation / 最严重限制

没有从 `q_i` 方差分配到“选哪一对响应”的、保持动作 / 信息 / 预算的映射；也没有 live RM UQ。本机无法跑 ActiveUF 全文。

### Whether we exceed the published method / 是否超过发表方法

**NO valid comparison。** 不能回答是否超过 DRTS / DeltaUCB / AMPLe / HyPER。

### Whether a meaningful research gap remains / 是否仍有研究缺口

识别意义上的 MNAR 审计通道在 v1.1 内部仍可见，但 **相对 CCF-A 主动学习论文的缺口尚未被合法实验定位。** 在对照路径出现之前，不能把缺口写成“我们补上了 DRTS 做不到的内生删失”。

### Whether to invest in a new method version / 是否投资 v1.3

**否。** 合法对照路径当前不存在（本 Mac 无 RM UQ / 无已通过审计的 skill-free 映射）。v1.1 对经典对照已是 NOT PROMISING，不支持再发明分配规则。

### The one most valuable next experiment / 下一步最有价值的实验

不是新算法。若未来有带官方 UQ 的 completion 池，先重新打开 Stage 3：映射若仍不成立，对照继续 BLOCKED。在那之前不跑 10/30-seed。

## Idea B

### Current evidence / 当前证据

- 已发表对照：Protocol B2 **INCOMPARABLE**。PP-bandit 原设定数值可沿用，但与技能分配不可比。
- v1.1 内部 Env1 n=`30`：B-v1.1c vs E-optimal worst-skill `interval_side=entirely_positive`，diff `0.0007274265296445326`。判定 **NOT PROMISING**。

### Strongest published competitor / 最强已发表竞争者

名义上最接近的是 **PP-bandit** 两层零阶重参数化（ICML 2024）。ncvxpp 的 performative SGD 是另一目标，且全文 MPI blocked。

### Strongest valid experimental result / 最强合法实验结果

相对发表方法：**没有合法对照结果。** 最强原设定产物是 PP-bandit 1-D `final_PR_phi = -0.2816498886156836` 等端点，不能与 worst-skill MSE 做差。

### Most serious limitation / 最严重限制

官方一维环境没有技能组，也没有可分配的反馈动作。最小缺失层只能造成标量缺失，不能承载 EqualSkill / E-optimal / worst-skill look-ahead。

### Whether we exceed the published method / 是否超过发表方法

**NO valid comparison。**

### Whether a meaningful research gap remains / 是否仍有研究缺口

闭环可观测性崩溃是 v1.1 合成现象，尚未在发表 performative benchmark 的合法扩展上定位。在不加私有技能标签的前提下，当前环境 **撑不起** 该缺口的 head-to-head。

### Whether to invest in a new method version / 是否投资 v1.3

**否。** 没有 skill-free 的合法对照路径。E-optimal 已是更强的内部主指标基线。

### The one most valuable next experiment / 下一步最有价值的实验

不要给 PP-bandit 私加技能标签。若存在作者提供的、本身带部分反馈动作的官方环境，再重新做 Stage 3。否则不进入 v1.3 算法。

## 联合建议

两个 idea 都不进入 v1.3 算法迭代。需要的是合法发表对照路径，而不是新的分配启发式。

