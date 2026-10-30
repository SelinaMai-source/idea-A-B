# Lazy Deployment（generic control）

不是 NeurIPS 2024 官方 `synthetic-iter` 的输出。

规则：观测分配与 Greedy Deployment 相同，但参数每隔 `lazy_period` 轮才更新。用来检验“信息变多”是否自动变成更好的部署表现。
