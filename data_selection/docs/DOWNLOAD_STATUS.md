# 下载状态

2026-11-03 直接请求官方 URL，没有使用镜像。没有下载 zip 或 checkpoint 正文。`DOWNLOAD_FAILED` 只表示传输失败，和 `SOURCE_VERIFIED_DEFERRED` 不是一回事。

| 资产 | URL | 镜像 | HEAD | Content-Length | 本地大小 | sha256 | 状态 | 路径 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| robomimic_image | `https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip` | 无 | 200，`application/zip`，`Accept-Ranges: bytes` | 84754919326 | 未下载 | 空 | `SOURCE_VERIFIED_DEFERRED` | 空 |
| robomimic_lowdim | `https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip` | 无 | 200，`application/zip` | 1930261710 | 未下载 | 空 | `SOURCE_VERIFIED_DEFERRED` | 空 |
| libero_hdf5 | `https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets` | 未改用 | 先前 API 200 | 仓库 `usedStorage` 99830157042；逐文件大小仍是 0 | 未下载 | 空 | `SOURCE_VERIFIED_DEFERRED` | 空 |
| lerobot_libero | `https://huggingface.co/datasets/lerobot/libero` | 无 | 先前正文被截断 | 未作为官方 HDF5 | 未下载 | 空 | `NOT_YET_CHECKED` | 空 |
| Lift MH `latest.ckpt` | `.../lift_mh/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt` | 无 | 200 | 4627773609 | 未下载 | 空 | `SOURCE_VERIFIED_DEFERRED` | 空 |
| Lift MH `epoch=0250-test_mean_score=1.000.ckpt` | 同目录 | 无 | 200 | 4627773609 | 未下载 | 空 | `SOURCE_VERIFIED_DEFERRED` | 空 |
| 其他任务 checkpoint | `https://diffusion-policy.cs.columbia.edu/data/experiments/image/` | 无 | 目录 200 | 未逐文件 HEAD | 未下载 | 空 | `NOT_YET_CHECKED` | 空 |
| HinFlow HF | `https://huggingface.co/datasets/Zhangchen0731/HinFlow` | 无 | 先前 API 200 | `usedStorage` 28527405288 | 未下载 | 空 | `SOURCE_VERIFIED_DEFERRED` | 空 |

`image/` 下列出 `can_mh`、`can_ph`、`lift_mh`、`lift_ph`、`pusht`、`square_mh`、`square_ph`、`tool_hang_ph`、`transport_mh`、`transport_ph`。`can_mh`、`square_mh`、`transport_mh` 下面有 `diffusion_policy_cnn`、`diffusion_policy_transformer`、`ibc_dfo`、`lstm_gmm`。这些任务的 checkpoint 文件名没有打开。

Lift MH `train_0/checkpoints/` 只有 `latest.ckpt` 和 `epoch=0250-test_mean_score=1.000.ckpt`。后者是 `PRETRAINED_EVALUATION_CHECKPOINT`，不是 `INITIAL_CHECKPOINT`。`latest.ckpt` 也不当作从零训练的初值。

本机下载器对 `robomimic_image` 和 `robomimic_lowdim` 返回 `SOURCE_VERIFIED_DEFERRED`，不把部分文件标成完成。
