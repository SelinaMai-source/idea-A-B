# 复现状态

没有 RoboMimic 或 LIBERO 的图像 Diffusion Policy 训练结果。论文格子保持 `NOT_RUN`。

| 项目 | 状态 |
| --- | --- |
| 源码 pin | 11 个仓库 HEAD 与清单 pin 一致 |
| HinFlow 子模块 | 未完成，不能当 ready |
| 图像 / 低维数据包 | `SOURCE_VERIFIED_DEFERRED` |
| Lift MH checkpoint 正文 | 未下载 |
| 父环境 torch / CUDA | 没有 |
| 已发表方法官方栈 | 未执行，适配器 `BLOCKED` |
| DataMIL 统一 DP | `BLOCKED`，因为原文脚本使用 Octo |
| 在线 P2 | `BLOCKED`，策略、模拟器和监督都没有就绪 |
| 合成烟测 | 只在总表第 11 节，不是论文数字 |
| CPU pytest | 49 tests，0 failed，2 skipped。两个 skip 都是 CUDA / 模拟器集成测试 |

`training_optimizer_steps` 不会从合成循环报成训练步数。合成循环写 0，真实受阻路径写 `null`。
