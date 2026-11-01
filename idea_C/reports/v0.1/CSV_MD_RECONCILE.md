# CSV ↔ Markdown reconcile (v0.1)

日期：2026-11-02。只读对账，未改任何已有数字。未把 `NOT_RUN` 换成 0。
Data 列里未转义的 `|H1|` 按表头列数从右侧对齐后再比。

- markdown 数据行：310（`v0*`=158，`C-v0.1*`=143）
- 因 `|H1|` 多切列、已重对齐的行：67
- v0.1 raw `metrics.json`：144
- 已匹配 version 并比较的数值/token 格：480
- numeric 一致：450；token 一致：30
- mismatch：0
- raw 无对应 MD 行：1
- 「尚未运行」的 v0 行被填 0：0

**结论：已写入的 C-v0.1 数字与 CSV/JSON 均值一致，0 处 mismatch。v0 旧行未被覆盖、未把 NOT_RUN 换成 0。**

## Raw without markdown row

|run_id|version|reason|
|---|---|---|
|v01_eps.fig4_smoke_seed0|C-v0.1_INDEPENDENT_eps.fig4_smoke|stub_or_not_ingested|

