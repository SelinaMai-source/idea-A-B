# 下载状态

本机只做了 HTTP 元数据。没有下载 zip、HDF5 或 checkpoint。

| 资产 | URL | 状态 | 测到的大小 |
| --- | --- | --- | --- |
| RoboMimic 图像包 | https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip | SOURCE_VERIFIED_DEFERRED | HEAD `Content-Length` 84754919326 字节 |
| RoboMimic 低维包 | https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip | SOURCE_VERIFIED_DEFERRED | HEAD `Content-Length` 1930261710 字节 |
| LIBERO HDF5 | https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets | SOURCE_VERIFIED_DEFERRED | revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`；`usedStorage` 99830157042 字节；逐文件 size 为 0 |
| LeRobot LIBERO | https://huggingface.co/datasets/lerobot/libero | NOT_YET_CHECKED | API HTTP 200，正文被截断，未解析完 |
| Lift MH CNN train_0 | experiments/image/lift_mh/diffusion_policy_cnn/train_0/checkpoints/ | SOURCE_VERIFIED_DEFERRED | 目录里有 `latest.ckpt` 与 `epoch=0250-test_mean_score=1.000.ckpt`，未取文件大小 |
| 其他任务 checkpoint | experiments/image/ 下的其余任务目录 | NOT_YET_CHECKED | 只确认了任务文件夹存在 |
| HinFlow 数据与 planner | https://huggingface.co/datasets/Zhangchen0731/HinFlow | SOURCE_VERIFIED_DEFERRED | revision `63f6c6866538b32958e728d42c25d8a0e84b16b9`；`usedStorage` 28527405288 字节；逐文件大小缺失；未下载 |

hf-mirror 对 `yifengzhu-hf/LIBERO-datasets` 返回了同一个 revision 和同一个 `usedStorage`。没有假设每个文件都被镜像。

父仓库已经忽略 `*.parquet`，子目录不能把 parquet 重新放进 git。导出会写 CSV 和 JSONL。当前环境没有 pyarrow，parquet 记为未生成，不是空表冒充。

校验和：官方响应里没有提供 sha256，清单里留空，不用自造哈希。
