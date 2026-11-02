# Benchmark 协议

四个 RoboMimic 任务、LIBERO-10、LIBERO-90、HinFlow 的 LIBERO 任务、自建任务增量，是四套协议，不共享「已经跑过」的结论。

## RoboMimic MH

任务目录：`robomimic_lift_mh`、`robomimic_can_mh`、`robomimic_square_mh`、`robomimic_transport_mh`。

官方图像实验目录已经列到：

`https://diffusion-policy.cs.columbia.edu/data/experiments/image/{lift,can,square,transport}_mh/`

每个任务目录下看得到 `diffusion_policy_cnn/`。只有 `image/lift_mh/diffusion_policy_cnn/train_0/checkpoints/` 被继续列过，里面是 `latest.ckpt` 和 `epoch=0250-test_mean_score=1.000.ckpt`。其他任务的 checkpoint 文件名没有列，状态是 `NOT_YET_CHECKED`。

配置里的图像键，以 `lift_image.yaml` 为例，是 `agentview_image` 和 `robot0_eye_in_hand_image`。数据文件本身没打开。

遗忘：`N/A`。

## LIBERO-10 与 LIBERO-90

LIBERO-10 是 DataMIL / STRAP 原文设定。LIBERO-90 在 STRAP README 里是检索库。二者都还没跑。

Hugging Face `yifengzhu-hf/LIBERO-datasets` 的 revision 是 `f13aa24a3da8c43c7225569f28c562979fa0e35a`。`usedStorage` 是 99830157042 字节。同一次响应里单个文件的 `size` 是 0，所以没有逐文件大小。HDF5 没下载。

`lerobot/libero` 的 API 返回了 HTTP 200，但保存的正文在 20000 字符处被截断，revision 和文件清单没有解析完。它是改编数据，不能和官方 HDF5 混用。

## HinFlow 原文

单独放在 `benchmarks/libero_hinflow/` 和 `references/hinflow/`。不进入 Image-DP 表。README 给出的 LIBERO 任务名是 `libero_butter`、`libero_book`、`libero_chocolate`、`libero_microwave`。ManiSkill 任务不在本阶段。子模块初始化超时，vendored 依赖可能不完整。

## 自建持续学习

`benchmarks/libero_continual/` 才记录 `A[t, j]`、遗忘和后向迁移。它不是 DataMIL 或 HinFlow 的原文设定。当前 `NOT_RUN`。
