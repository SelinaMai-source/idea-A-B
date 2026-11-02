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

每个选择器一个目录：`baselines/<name>/`，里面有 `README.md`、`config.yaml`、`adapter.py`、`audit.md`。五个控制组是纯 Python。CUPID、DemInf、DataMIL、Demo-SCORE、STRAP 在官方栈没跑起来时返回 `BLOCKED` 或 `SOURCE_UNAVAILABLE`，不会改走 Random。

每个任务一个目录：`benchmarks/<task>/`。RoboMimic 和 LIBERO 的树只克隆一次，放在 `benchmarks/_sources/`。

HinFlow 在 `references/hinflow/`，不是 baseline，也不进图像 Diffusion Policy 排行榜。策略只在 `policies/diffusion_policy_image/`。

## 现在的事实

Mac arm64、无 NVIDIA、父环境无 torch。论文格子是 `NOT_RUN`。总表是 [`data_selection_v0实验版本+结果.md`](data_selection_v0实验版本+结果.md)。

合成在线烟测只证明状态机和指标代码，文件在 `results/synthetic/interface_smoke/`。

服务器上的命令写在 [`docs/GPU_EXECUTION_PLAN.md`](docs/GPU_EXECUTION_PLAN.md)。不要在这台 Mac 上下载约 78.9GiB 的 `robomimic_image.zip`。
