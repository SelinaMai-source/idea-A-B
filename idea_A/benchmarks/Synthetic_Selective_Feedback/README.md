# Synthetic Selective Feedback

Idea A 的合成世界。总体 `n=2000`，离散层 `K=8`，有限总体真值由完整 `Y` 精确计算。

机制：

- `mcar`：`S` 与 `(X, Y)` 独立。
- `mar`：`S` 只依赖观测到的 `X`。
- `mnar`：`S` 依赖未观测的 `Y`。自然通道是**识别失败**，不是估计器实现错误。
- `policy_dependent`：`S` 依赖行为动作 `π`。
- `low_overlap`：部分层 `P(S=1)` 接近 0。
- `zero_overlap`：部分层 `P(S=1)=0`。自然通道在这些层上不可识别。
- `nonstationary`：`P(S=1)` 随时间下标变化。

算法只能通过 `World.observe(a)` 读取标签：当且仅当 `O = S ∨ A` 时 `Y` 可见。独立审计倾向 `q` 由分配规则写入，估计时当作已知。
