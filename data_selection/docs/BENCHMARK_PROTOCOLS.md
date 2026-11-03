# Benchmark 协议

比较只用 RoboMimic 的 Lift / Can / Square / Transport MH，以及 LIBERO。统一策略是图像 CNN Diffusion Policy。Octo、BC-Transformer、HinFlow Transformer、纯 MLP、VLA 不进入这张表。

低维数据只保留接口，`formal_image_dp_result` 为 false。

## 变体

`diffusion_policy/config/task/lift_image.yaml` 和 `lift_image_abs.yaml` 的 `dataset_type` 都是 `ph`。这个字段不能写成 Lift MH。`benchmarks/common.py` 在路径或 HDF5 的 `env_args` 给出 `ph` 而 benchmark 要求 `MH` 时返回 `BLOCKED`。文件不存在也是 `BLOCKED`，不是成功。

各任务预期路径写在对应 `adapter.py` 的 `expected_dataset_path`，目前这些文件不在本机。

## 数据

RoboMimic 图像包和低维包都只做了 HEAD，没有下载正文。LIBERO 使用官方 `benchmark_scripts/download_libero_datasets.py` 和 `yifengzhu-hf/LIBERO-datasets`。不用 `lerobot/libero` 代替官方 HDF5。没有要求时不下载 LIBERO-90 全量。

## 评估

`load_dataset`、`validate_schema`、`make_environment`、`evaluate_policy` 在文件缺失或模拟器不可导入时失败关闭。`evaluate_policy` 没有环境对象时不填写成功率。

遗忘、BWT、`average_success` 只属于 `libero_continual`。单任务 RoboMimic 是 `N/A`。`forward_transfer` 还没有定义。
