# demo_score

发表方法。打分需要成功与失败 rollout 监督。没有这组监督时保持 BLOCKED。

官方树在 gitignore 的 `official/`。源不可用或入口未审计时，`select()` 返回 `SOURCE_UNAVAILABLE` 或 `BLOCKED`，选中集合为空。
