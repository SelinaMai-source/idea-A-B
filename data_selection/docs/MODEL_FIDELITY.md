# 模型保真

统一骨干是官方 `DiffusionUnetImagePolicy`，工作区类是 `TrainDiffusionUnetImageWorkspace`。适配器不重写扩散更新。`optimization_step` 调用 `policy.compute_loss`、`backward`、优化器和 scheduler。

2026-11-03 从 pin `5ba07ac6661db573af695b419a7947ecb704690f` 的 `diffusion_policy/config/train_diffusion_unet_image_workspace.yaml` 读到：

- `n_obs_steps`: 2
- `horizon`: 16
- `n_action_steps`: 8
- `policy._target_`: `diffusion_policy.policy.diffusion_unet_image_policy.DiffusionUnetImagePolicy`
- `optimizer._target_`: `torch.optim.AdamW`
- `training.lr_scheduler`: cosine
- `training.device`: `cuda:0`
- 默认 task：`lift_image_abs`

`lift_image.yaml` 的图像形状是 `[3, 84, 84]`，动作形状是 `[7]`，`dataset_type` 是 `ph`。`lift_image_abs.yaml` 的动作形状是 `[10]`，`dataset_type` 也是 `ph`。MH 必须来自 `mh` 数据集路径，不能从这两份 yaml 推断。

## Checkpoint 角色

- `INITIAL_CHECKPOINT`：官方工作区自己的初始化。没有传入 checkpoint 时才是这条。
- `PRETRAINED_EVALUATION_CHECKPOINT`：`epoch=0250-test_mean_score=1.000.ckpt`。不能当从零训练的初值。
- `RETRAINED_CHECKPOINT`：同一次运行输出目录里的 checkpoint，只在 `--resume` 时使用。
- `latest.ckpt` 不自动当成初值。Lift 路径里的文件不用于 Can、Square 或 Transport。

本机没有 CUDA，也没有这些文件的正文。真实前向和反向留在带 `@pytest.mark.integration` 的测试里，Mac 上会 skip。skip 不是通过，也不是论文结果。
