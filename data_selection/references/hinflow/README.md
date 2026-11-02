# HinFlow 参照

HinFlow 不是数据选择 baseline，也不进入统一 Image-DP 排行榜。README 写的是高层 planner 加 hindsight 在线模仿。训练入口是 `python -m scripts.train_hinflow_policy`。骨干不是图像 CNN Diffusion Policy。

LIBERO 任务名：`libero_butter`、`libero_book`、`libero_chocolate`、`libero_microwave`。hindsight 只留在本参照里。

官方树在 gitignore 的 `official/`。检出 SHA 以 `manifests/source_repositories.yaml` 为准。仓库根没有 LICENSE。数据页是 README 给出的 `https://huggingface.co/datasets/Zhangchen0731/HinFlow`，本机没有下载。
