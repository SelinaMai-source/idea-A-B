# 具身在线数据选择 v0

这个目录是复现框架，不是新方法。没有 `ours`，也没有 v1。

官方源码在各自的 `official/` 或 `benchmarks/_sources/`，已被根 `.gitignore` 排除。提交的是适配器、清单、测试和文档。`PYTHONPATH` 指到本目录：

```bash
cd data_selection
export PYTHONPATH="$PWD"
../.venv/bin/python -m pytest
```

父仓库的测试路径不含这里。

## 布局

每个选择器一个目录：`baselines/<name>/`，里面有 `README.md`、`config.yaml`、`adapter.py`、`audit.md`。五个控制组是纯 Python。CUPID、DemInf、Demo-SCORE、STRAP 会尝试导入作者入口；依赖缺失时返回 `BLOCKED`，`official_stack_invoked` 为 false。DataMIL 的官方 LIBERO 脚本使用 Octo，统一图像 DP 比较保持 `BLOCKED`。都不会改走 Random。

每个任务一个目录：`benchmarks/<task>/`。RoboMimic 和 LIBERO 的树只克隆一次，放在 `benchmarks/_sources/`。

HinFlow 在 `references/hinflow/`，不是 baseline，也不进图像 Diffusion Policy 排行榜。策略只在 `policies/diffusion_policy_image/`。

## 现在的事实

Mac arm64、无 NVIDIA、父环境无 torch。论文格子是 `NOT_RUN`。总表是 [`data_selection_v0实验版本+结果.md`](data_selection_v0实验版本+结果.md)。

`--device cuda` 只在 CUDA 实际不可用时阻断。有数据集和 CUDA 时，`scripts/run_unified_baselines.py` 会调用官方 `TrainDiffusionUnetImageWorkspace`。本机没有数据，也没有 CUDA，所以这条路径返回 `BLOCKED`，不训练。

合成在线烟测只证明状态机和指标代码，文件在 `results/synthetic/interface_smoke/`。它不是论文结果。

服务器上的第一条真实命令写在 [`docs/GPU_EXECUTION_PLAN.md`](docs/GPU_EXECUTION_PLAN.md)，对应 `scripts/run_lift_mh_all_data.sh`。不要在这台 Mac 上下载 `robomimic_image.zip`（HEAD Content-Length 84754919326）。
