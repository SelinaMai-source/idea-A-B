# Image CNN Diffusion Policy

统一比较只使用这一族：图像输入、CNN 编码器、Conditional U-Net 动作扩散策略。实现指向 `real-stanford/diffusion_policy` 的本地忽略目录 `official/`。

本目录不放进任何一个数据选择 baseline 里冒充该方法的原骨干。Octo、BC-Transformer、HinFlow Transformer、纯 MLP 和大型 VLA 不能替入这张表。

无 CUDA 时状态是 `BLOCKED_NO_CUDA`。无已下载权重时状态是 `BLOCKED_NO_CHECKPOINT`。低维数据只能做接口烟测，不能写成图像 Diffusion Policy 结果。horizon 与分辨率只在读到官方 task yaml 后记为已核对，读不到则为 `UNVERIFIED`。
