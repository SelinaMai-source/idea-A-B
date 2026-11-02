# 复现状态

没有一条论文格子被跑完。图像 Diffusion Policy 训练是 `BLOCKED_NO_CUDA`。权重没有下载，是 `BLOCKED_NO_CHECKPOINT` 这一层的缺失；因为没有 CUDA，探针状态记为 `BLOCKED_NO_CUDA`。

| 协议 | 状态 |
| --- | --- |
| P0 原文复现 | NOT_RUN |
| P1 统一离线 Image-DP | NOT_RUN |
| P2 在线闭环 | 只有 `SYNTHETIC_INTERFACE_SMOKE`，论文行是 NOT_RUN |
| P3 LIBERO 任务增量 | NOT_RUN |

合成烟测在 `results/synthetic/interface_smoke/result.json`。seed 0，两轮评估的合成成功率都是 0.0，AUC 也是 0.0。这是该随机过程算出来的数，不是把未跑的 RoboMimic 或 LIBERO 填成 0。优化器步数是 0，没有更新 Diffusion Policy。

论文数字没有从 PDF 里抽取，对照表是 `NOT_EXTRACTED` / `NOT_RUN`。

控制组选择器在 CPU 上有测试。这不表示它们已经在图像策略上训练过。
