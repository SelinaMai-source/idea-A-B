# v1.2 原设定基线复现

本文件由 `scripts/generate_v1_2_report.py` 从磁盘 JSON 与已锁定审计写出。数字均来自下列产物，没有手填对照 RMSE，也没有发明 10/30-seed CCF-A head-to-head。

- 生成时间（UTC）：`2026-10-30T20:17:05.282444+00:00`
- 仓库：`/Users/grantwright/Desktop/科研/10.30新课题/v0`
- 分支：`v1.2`
- commit：`fd00badccc8f1e4a8012b91c2b9d473914f4f3c5`
- remote：`git@github.com:SelinaMai-source/idea-A-B.git`
- 缺失的主源：无
- 可比性锁：`reports/v1.2/03_COMPARABILITY_AUDIT.md`
- 主判定：`PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`
- scientifically valid DIRECT 对照数：`0`；可辩护 ADAPTED 对照数：`0`

本文件主要沿用 `02_ORIGINAL_REPRODUCTION.md` 与已有 AMPLe / PP-bandit / ncvxpp JSON。没有新的全文复现。

## 总览

| 对象 | 官方设定 | 本地做了什么 | 保真度 | 论文数字能否对照 |
| --- | --- | --- | --- | --- |
| AMPLe | README n_rounds=100, burnin=50000, volume | n_rounds=3, burnin 与 volume 不变 | partial | 论文无与本地 L2 可减的端点表；L2 只描述本次 3 轮 |
| PP-bandit | bandit-icml24.ipynb，Appendix H 图 4–6 | 同一算法单元 + numpy seed 0 | 数值脚本 complete；图端点未建立 | 论文图注只有横轴区间，没有端点 |
| ncvxpp | mpiexec -np 10 python main.py | rank=0 size=1 stub，eps∈{0,0.1,10,100}，batch=5 | partial | 论文图未数字化 |
| HyPER | TPU / vLLM / 8B–13B RM | 未跑 | blocked | 无本地值 |
| Drago | 无官方代码 | 未实现 | blocked | 无 |
| ActiveUF 全文 | CUDA 13, vLLM 0.19, Skywork RM, Qwen3-32B oracle | 未跑 | blocked | 无 |
| ActiveUF 选择器 | 官方 DRTS/DeltaUCB/InfoMax | 隔离 venv unit check，合成 2×4 张量 | DIAGNOSTIC import/call only | 不是论文 RM/DPO 表 |
| UCB-OGD | Table 1 CIFAR/VOC/VQA/testbed | 无官方代码；independent_paper_based not done | 未复现 | 论文表不进对照列 |

主判定仍是 `PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`。原设定复现成功不等于 head-to-head 成立。

## AMPLe：官方 vs 本地

| 项 | 官方 / README | 本地 JSON |
| --- | --- | --- |
| commit | `0cad341f6a9e77063ef451c27732bf5c1534c797` | 同上 |
| acquisition | volume | n_rounds shortened from README 100 to 3; burnin and acquisition unchanged |
| n_rounds | 100（README 示例） | 3 |
| burnin | 50000 | 50000 |
| seed | 命令 --seed=1 | 1（见审计） |
| true_preference | 示例 0.2 0.7 0.1 | [0.2, 0.7, 0.1] |
| l2_t0 | 论文未给该次运行 | 0.4688851731916284 |
| l2_t3 | 论文未给该次运行 | 0.3647449275776112 |
| wall_sec | 未给 | 97.8 |

命令：`python main.py --seed=1 --task=assistant --rm_names harmless helpful humor --n_rounds=3 --true_preference 0.2 0.7 0.1 --true_beta=10 --beta=2.0 --jitter=0.01 --alpha=10 --init=previous --acquisition=volume --burnin=50000 --gamma=0.0 --mode=likelihood --margin=none --inputs=0`

## PP-bandit：官方 vs 本地

| 项 | 官方 notebook / 论文 | 本地 JSON |
| --- | --- | --- |
| commit | `d5daeffba8683dc9d8d0f1bedd3fa2c5f168c0c5` | 同上 |
| 环境 | 1-D Bernoulli PR，无 skill | 相同；源码无 skill |
| seed | notebook 无 seed | 0 |
| 单次 500 步 PR(φ) | 图中无端点表 | -0.2816498886156836 |
| 单次 500 步 PR(θ) | 图中无端点表 | -0.2962457875065718 |
| 10 初值 PR(φ) mean | 无 | -0.2883591540722531 (std 0.007139595668885597) |
| 10 初值 PR(θ) mean | 无 | -0.291834361799254 (std 0.007228858237631779) |
| elapsed_sec | 未给 | 0.814453125 |

保真度评估：算法单元来自官方 notebook，数值可复跑；**不能**写成“复现了图 4–6 的数值”。

## ncvxpp：官方 vs 本地

| 项 | 官方 | 本地 |
| --- | --- | --- |
| commit | `ccbeeddfbf9237aa06dca3a1579529faabd8fe99` | 同上 |
| 命令 | `cd synthetic-iter && mpiexec -np 10 python main.py` | PYTHONPATH=official_reproduction/mpi4py_stub python synthetic-iter/main.py |
| 进程数 | 10 | 1（mpi4py stub） |
| eps | 官方脚本 | [0, 0.1, 10, 100] |
| batch | 官方脚本 | 5 |
| wall_sec | 未在本机对比 | 540.486 |
| 状态 | 全文 | partial |

| npy | iter_last | loss_last | train_acc_last | test_acc_last |
| --- | --- | --- | --- | --- |
| eps0-batch5-res0.npy | 99426 | 0.3281349241733551 | 0.9004424778761062 | 1 |
| eps0.1-batch5-res0.npy | 99426 | 0.3458000421524048 | 0.9004424778761062 | 1 |
| eps10-batch5-res0.npy | 99426 | 0.3553709387779236 | 0.9004424778761062 | 1 |
| eps100-batch5-res0.npy | 99426 | 0.9450080394744873 | 0.09955752212389381 | 0 |

失败 / blocker：没有 `mpi4py`，PATH 上没有 `mpicc` / `mpiexec`。`spam-iter` 依赖 torch，未跑。

## ActiveUF 选择器 unit check

`status=ok`，`check_type=selector_unit_check`，`comparability_cell=null`，`protocol_a2=BLOCKED_INCOMPARABLE`。
DRTS pairs `[[3, 0], [1, 0]]`；DeltaUCB `[[3, 0], [1, 0]]`；InfoMax `[[2, 0], [2, 0]]`。
主 venv 导入失败：`ModuleNotFoundError: No module named 'torch'`（见 `02_ORIGINAL_REPRODUCTION.md`）。

## 其余 blocked

HyPER、Drago、UCB-OGD、ActiveUF 全文：无失败日志中的指标，因为实验未启动。原因见 `02_ORIGINAL_REPRODUCTION.md`。

结论：有 partial / complete 的 **原设定** 产物，但 `PUBLISHED BASELINE COMPARISON NOT ESTABLISHED`。

