# 模型保真

统一比较只允许图像 CNN Conditional U-Net Diffusion Policy，类名来自官方配置：

`diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy`

读到的训练配置 `diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`：

- `n_obs_steps` = 2
- `horizon` = 16
- `n_action_steps` = 8

`diffusion_policy/config/task/lift_image.yaml` 里 RGB 形状是 `[3, 84, 84]`，动作维是 7，`dataset_type` 的字面值是 `ph`。`transport_image.yaml` 的动作维是 14。这些是配置文件里的值。MH 的 HDF5/Zarr 没有打开，不能把 ph 配置说成 MH 数据已经核对。

本机没有 torch，也没有 CUDA。策略探针状态是 `BLOCKED_NO_CUDA`。没有把骨干换成 Octo、BC-Transformer、HinFlow Transformer、纯 MLP 或大型 VLA。

低维数据包只允许做接口说明，不能填进图像结果表。

原文复现必须用作者自己的骨干。DataMIL 的 LIBERO 路径在 README 里是 Octo。DemInf 是 JAX/OpenX。HinFlow 不是这个 U-Net。它们都不进入同一张 Image-DP 排行榜。
