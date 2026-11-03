# 兼容矩阵

`core/matrix.py` 把下面几列分开。缺 CUDA 只写进 `RUNTIME_READINESS`，不把 `image_dp_compatibility` 写成 `BLOCKED_NO_CUDA`。

- `SOURCE_STATUS`
- `IMPLEMENTATION_STATUS`
- `DEPENDENCY_STATUS`
- `DATA_STATUS`
- `MODEL_STATUS`
- `ORIGINAL_FIDELITY_STATUS`
- `UNIFIED_ADAPTATION_STATUS`
- `RUNTIME_READINESS`
- `EXPERIMENT_STATUS`

2026-11-03 本机 `RUNTIME_READINESS` 是 `BLOCKED_NO_CUDA`。五个已发表方法的适配器是 `IMPLEMENTED`，官方栈没有执行，`EXPERIMENT_STATUS` 是 `NOT_RUN`。控制组选择器是 `IMPLEMENTED_CPU_SELECTOR`。HinFlow 的 `image_dp_compatibility` 是 `NOT_APPLICABLE`。DemInf、DataMIL、STRAP 的图像 DP 适配还是 `NOT_YET_ADAPTED`，这不是 `NOT_APPLICABLE`。

克隆完成不等于 `READY`。当前没有 `READY` 的论文格子。可运行项只有 CPU 上的控制组接口，`device` 是 `cpu`。

生成文件：`configs/experiment_matrix.yaml`。
