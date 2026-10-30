# v1.2 原设定复现记录

日期：2026-10-31。本阶段沿用 v1 已经跑完或已经判定 blocked 的原设定产物，只补两处缺口：ActiveUltraFeedback 的选择函数能否在主 venv 导入，以及 UCB-OGD 是否有官方代码。没有重跑 AMPLe burn-in、PP-bandit notebook、ncvxpp。

主 venv：`/Users/grantwright/Desktop/科研/10.30新课题/v0/.venv/bin/python`，Python 3.12.4，numpy 2.5.4。`import torch` 的结果是 `ModuleNotFoundError: No module named 'torch'`。没有执行 pip，没有安装 vLLM 或 CUDA torch。

## 1. 沿用且本阶段不重跑

| 对象 | 沿用产物 | 本阶段动作 |
| --- | --- | --- |
| AMPLe `0cad341` | `idea_A/baselines/AMPLe/official_reproduction/metrics.json` | 不重跑。状态保持 partial |
| PP-bandit `d5daeff` | `idea_B/baselines/Performative_Bandit_ICML2024/official_reproduction/metrics.json` | 不重跑。对照论文图注，不宣称对齐 |
| ncvxpp `ccbeedd` | `idea_B/baselines/REPRODUCTION_AUDIT.json` 的 `what_ran` | 不重跑。`mpiexec -np 10` 仍 blocked |
| HyPER `4d96b27` | `idea_A/baselines/HyPER/REPRODUCTION_STATUS.md` | 不重跑。状态保持 blocked |
| Drago ICML 2026 | `idea_A/baselines/Online_Compatible_Reward_ICML2026/THEORY_COMPARATOR.md` | 不新写实现。状态保持 blocked |

可复用端点数字已抄在 `00_EXISTING_EVIDENCE_AUDIT.md` 第 4 节，来源是上述 JSON。

## 2. AMPLe：partial，L2 与总体 RMSE 不可比

本地命令仍是审计 JSON 里的那条：`acquisition=volume`，`burnin=50000`，`n_rounds=3`，`true_preference 0.2 0.7 0.1`，数据集 `minhyeonoh/feedback_assistant_harmlesshelpfulhumor_271`。README 示例的 `n_rounds=100` 没有跑。

保存的误差是偏好向量的 L2：

- `l2_to_true_t0 = 0.4688851731916284`
- `l2_to_true_t3 = 0.3647449275776112`

这是个性化向量到设定真值的距离，经过 3 轮 volume 查询。它不是论文 100 轮曲线的复现，也不是 Idea A 有限总体泛函的 RMSE。两者 **INCOMPARABLE**，不相减。

## 3. PP-bandit：数值脚本已跑完，论文图没有端点表

官方仓库 README 写明 `bandit-icml24.ipynb` 对应 Appendix H 的 Figure 4–6。本地 `run_official_numerical.py` 转换的是这些单元，并额外设置 `numpy` seed 0。notebook 源码本身没有 seed，所以这次 JSON 是“同一算法、固定种子的一次执行”，不是对论文图像素的核对。

论文 Appendix H 在可提取文本里只有图注，没有曲线端点：

- Figure 4：`θ ∈ [0.2, 0.4]`
- Figure 5：`θ ∈ [0.4, 0.8]`
- Figure 6：`θ ∈ [0.4, 0.8]`
- Figure 7：无重参数化的零阶法（橙）对比重参数化后的零阶法（蓝），Example 3

本地 `metrics.json` 的端点是：

- 单次 500 步，`x0=0.3`：`final_PR_phi = -0.28164988861568363`，`final_PR_theta = -0.2962457875065718`，`x_opt_phi = [0.332899095829306]`，`x_opt_theta = [0.6616285844084332]`
- 10 个初值各 100 步：`PR_phi` 均值 `-0.2883591540722531`（std `0.0071395956688855965`），`PR_theta` 均值 `-0.291834361799254`（std `0.007228858237631779`）

论文文本里没有这些数，也没有可以做差的发表端点。结论：数值运行完成；**不声称 notebook 复现了图 4–6 的数值**。

环境核对：notebook 实现的是一维 Bernoulli performative risk，内层用 Bernoulli KL 匹配 `φ`。源码中没有 skill、技能组或审计动作。B-v1.1 的技能分配若放在这个原环境上，**INCOMPARABLE**。

## 4. ncvxpp：单进程 partial

已有运行是 `PYTHONPATH=official_reproduction/mpi4py_stub python synthetic-iter/main.py`，`eps` 为 0、0.1、10、100，batch 5，最后迭代 99426。损失与准确率见 `00` 第 4 节。`eps=100` 的 `test_acc_last = 0.0`，其余三个设定的 `test_acc_last = 1.0`。审计记录的类别过滤（`data.csv` 为 `{+1,-1}`，平衡代码查找 `Y==0`）保持原说明。

官方多进程命令 `cd synthetic-iter && mpiexec -np 10 python main.py` 仍然 blocked。上游 README 的另一入口 `cd spam-iter && sh run.sh` 依赖 torch，本阶段没有跑。NeurIPS 论文图没有在本阶段数字化，所以这些 npy 终点不写成论文表的复现。单进程结果保持 `partial`。

## 5. HyPER 与 Drago

HyPER 仍 blocked：TPU v3-128、EasyLM、8B/13B RM、vLLM、Beaker、GCS，以及未初始化的 submodule。没有新的简化脚本。Idea A 的 Uncertainty Sampling 与 Idea B 的 Uncertainty-based Exploration 继续用原名。

Drago 等 ICML 2026 仍无官方代码。本阶段没有实现论文算法。

## 6. ActiveUltraFeedback 选择函数

### 克隆

```text
git clone --depth 1 --recurse-submodules=no \
  https://github.com/lasgroup/ActiveUltraFeedback.git \
  idea_A/baselines/ActiveUltraFeedback
```

- Commit：`3cf700a85bbe344b1b472890a488671e2916a8b4`
- 日期：2026-05-02 13:10:41 +0200
- 浅克隆：是
- 磁盘：约 1.9M
- submodule：`.gitmodules` 有三项，`git submodule status` 为空，`resources/` 不存在
- 父仓库只提交旁边的 `idea_A/baselines/ActiveUltraFeedback_MANIFEST.md`。克隆目录自己的 `.git` 还在，源码不进本次提交

### 读到的选择函数

| 名称 | 文件 | 类 | `__call__` 输入 |
| --- | --- | --- | --- |
| DRTS | `activeuf/acquisition_function/drts.py` | `DoubleReverseThompsonSampling` | `rewards`、`lower_bounds`、`upper_bounds`，注解为 `torch.Tensor` |
| DeltaUCB | `activeuf/acquisition_function/deltaucb.py` | `DeltaUCB` | 同上 |
| InfoMax | `activeuf/acquisition_function/infomax.py` | `InfoMax` | 同上 |

DRTS 用 `(upper-lower)/2` 当标准差，再对奖励做均匀 `[-1,1]` 扰动；第二步把奖励取负后再采样。实现里对张量调用 `.cpu()`。DeltaUCB 取 `upper_i - lower_j` 的最大间隔。InfoMax 对上下界差做 sigmoid 后再比置信间隔。三份文件的模块首行都是 `import torch`。`base.py` 只有 ABC，不导入 torch。

包的 `activeuf/acquisition_function/__init__.py` 会先导入 `dts.py`，而 `dts.py` 也是 `import torch`。因此 `import activeuf.acquisition_function.drts` 在进入 `drts.py` 之前就失败。

### 导入结果

直接按文件加载（避免只看到包初始化的间接失败）：

| 文件 | 结果 |
| --- | --- |
| `base.py` | 导入成功 |
| `drts.py` | `ModuleNotFoundError: No module named 'torch'` |
| `deltaucb.py` | `ModuleNotFoundError: No module named 'torch'` |
| `infomax.py` | `ModuleNotFoundError: No module named 'torch'` |

经包导入时，三个名字的第一条异常都是：

```text
File ".../activeuf/acquisition_function/__init__.py", line 1, in <module>
    from activeuf.acquisition_function.dts import DoubleThompsonSampling
File ".../activeuf/acquisition_function/dts.py", line 2, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

DRTS、DeltaUCB、InfoMax **都没有导入成功**。它们不是纯 numpy 函数，本阶段没有构造合成分数数组，也没有 `selector_unit_check`。这不是论文 RM/DPO 结果的复现。

全文流水线保持 blocked：README 要求 CUDA 13 与 vLLM 0.19，并在 RTX 5090 / GH200 上测试。响应生成、奖励模型训练、DPO/CPO 都没有跑。Hugging Face `combined` 的 parquet 没有下载。数据卡说明见 `01`：标注是 LLM judge，成对列里没有选择器所需的上下界，offline replay 本阶段不成立。

## 7. UCB-OGD 代码检索

论文：Zang et al., ICML 2025，https://proceedings.mlr.press/v267/zang25a.html 。PDF 文本使用 https://www.ee.columbia.edu/~jghaderi/icml25.pdf 的可提取文本。

检索与结果：

| 查询 | 结果 |
| --- | --- |
| PMLR HTML | 没有作者代码链接。出现的 GitHub 只有 `mlresearch/v267` 的会议页 |
| 哥伦比亚 PDF 文本 | 没有 `github.com` 代码地址 |
| GitHub 仓库搜索，引号查询 `"UCB-OGD"` | `total_count = 0` |
| GitHub 仓库搜索，`"ucb_ogd" OR "UCB_OGD"` | `total_count = 0` |
| 仓库搜索，论文标题短语 | `total_count = 0` |
| 作者 Chengbo Zang 的公开账号 `harrypotter1501` | 6 个公开仓库：`boosted-dyadic-kernel`、`domestic-terminal`、`intro-numerical-methods`、`movie-sentiment`、`stl-tutorial`、`vasculature-breast-dce-mri`。没有本篇 |
| `user:mkturkcan` 加 `UCB-OGD` | `total_count = 0` |
| `Javad-Ghaderi` 的公开仓库 | `CSrankings`、profile、`jghaderi`。没有本篇 |
| 未加引号的 `UCB-OGD` 仓库搜索 | 命中 `jacobvolz8/Online-Learning`，说明文字是课程作业里的 EWA、OGD、EXP3、UCB。这不是 Zang et al. 的官方实现，不登记为作者仓库 |

`independent_paper_based = not done`。本阶段不写多源 logistic，也不写 CIFAR / VOC / VQA 训练。这四类实验（外加 COSMOS testbed）在本机 blocked：没有官方脚本，也没有 Table 1 所需的 CNN、SSD300 或 SmolVLM 训练环境。论文 Table 1 的数字只作为发表结果记在登记表里。

## 8. 本阶段的复现结论

| 对象 | 结论 |
| --- | --- |
| AMPLe | partial 可沿用。个性化 L2 与总体 RMSE INCOMPARABLE |
| PP-bandit | 一维数值运行可沿用。未对齐论文图的端点。技能分配 INCOMPARABLE |
| ncvxpp | 单进程 partial 可沿用。全文 MPI blocked |
| HyPER | blocked |
| Drago | blocked，无代码 |
| ActiveUF 选择函数 | 导入 blocked，错误为 `No module named 'torch'`。没有 unit check |
| ActiveUF 全文 | blocked（CUDA 13 / vLLM 0.19 / 大 RM 与 judge） |
| UCB-OGD | 无官方仓库。`independent_paper_based = not done`。CIFAR、VOC、VQA blocked |

没有新算法，没有适配器，没有 Idea A/B 对比实验。
