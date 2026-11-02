# 未解决

- 本机无 NVIDIA、无 torch，不能训练或评估图像 Diffusion Policy。
- `robomimic_image.zip` 有 84754919326 字节，按计划没有下载。低维包也没有下载。
- LIBERO HDF5 未下载。`usedStorage` 是仓库级数字，逐文件大小在 API 里是 0。
- `lerobot/libero` 的元数据正文被截断，revision 未记录。
- 除 Lift MH `diffusion_policy_cnn/train_0` 外，checkpoint 文件名没有逐个列出。
- MH 的 HDF5/Zarr 没打开，图像键只来自 task yaml。`lift_image.yaml` 的 `dataset_type` 是 `ph`。
- 论文表格里的成功率没有抽取。
- CUPID、DemInf、DataMIL、Demo-SCORE、STRAP 的官方入口都没有执行。
- HinFlow、robosuite、LIBERO 的第一次克隆在 180 秒超时。600 秒重试后三者都有 SHA。HinFlow 的子模块初始化失败，`third_party` 里的 robosuite / robomimic / maniskill 可能不完整。
- Demo-SCORE 与 STRAP 的仓库根目录没有 LICENSE 文件，许可证记为未核实。
- 父环境没有 pyarrow，parquet 没有生成。
- 没有实现 ours / v1。
