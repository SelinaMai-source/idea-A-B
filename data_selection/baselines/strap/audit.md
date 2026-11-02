# STRAP 审计

- 检出：`baselines/strap/official`，SHA `a106da6cb178893923498f5cc0b4a51933c47ab3`
- companion：`baselines/strap/official_robomimic_strap`，SHA `3386adb5b664e9bedf88e7d26ced986439e67193`，MIT License
- STRAP 仓库根没有 LICENSE。
- README：ICLR 2025。LIBERO-10 上取少量演示，从 LIBERO-90 检索子轨迹，编码用 DINOv2。
- 入口已看到：`strap/retrieval/retrieval.py`、`strap/embedding/encode_datasets.py`
- 未编码，未检索，未训练。子轨迹预算不能换成 episode 随机子集。
