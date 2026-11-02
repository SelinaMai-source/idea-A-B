# 官方源码审计

克隆使用 HTTPS、depth 1。SHA 是当时的远端 HEAD。失败记为 `SOURCE_UNAVAILABLE`，没有用自写启发式顶替。

| 名称 | 状态 | SHA | 许可证文件 | 子模块 |
| --- | --- | --- | --- | --- |
| cupid | CLONED | `2941eba9427a3398d3cbe26b10c01e3ece56bd18` | LICENSE：MIT License | INITIALIZED |
| datamil | CLONED | `f8fc4abccacd1e2b070e68128a7f2bce98f4b303` | LICENSE：MIT License | NO_SUBMODULES |
| deminf | CLONED | `232151defb258b5ceb0a6b62644574b57d981b58` | LICENSE：MIT License | NO_SUBMODULES |
| demo_score | CLONED | `b82a5ece97b8975643b14703734afa8f24ab7b0a` | 无 | NO_SUBMODULES |
| diffusion_policy | CLONED | `5ba07ac6661db573af695b419a7947ecb704690f` | LICENSE：MIT License | NO_SUBMODULES |
| hinflow | CLONED | `f80eeae61bc3e1b32b03b63f06e4add3943a473f` | 无 | SUBMODULE_FAILED |
| libero | CLONED | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | LICENSE：MIT License | NO_SUBMODULES |
| robomimic | CLONED | `d309eaecc18acf4152a830a895a6984b8ac71b05` | LICENSE：MIT License | NO_SUBMODULES |
| robomimic_strap | CLONED | `3386adb5b664e9bedf88e7d26ced986439e67193` | LICENSE：MIT License | INITIALIZED |
| robosuite | CLONED | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` | LICENSE：MIT License | NO_SUBMODULES |
| strap | CLONED | `a106da6cb178893923498f5cc0b4a51933c47ab3` | 无 | NO_SUBMODULES |

## 失败与子模块

- `hinflow` 子模块 SUBMODULE_FAILED：submodule update timed out after 300s: Command '['git', '-C', '/Users/grantwright/Desktop/科研/10.30新课题/v0/data_selection/references/hinflow/official', 'submodule', 'update', '--init', '--depth', '1', '--recursive']' timed out after 300 seconds

第一次克隆 HinFlow、robosuite、LIBERO 时 180 秒超时。随后 600 秒重试得到了上面的 SHA。

入口只记录已经在检出里看到的文件。这些入口都没有被执行。

- CUPID：`eval_demonstration_scores.py`、`train_trak_diffusion.py`、`finalize_trak.py`。README 写的是 RoboMimic 与 PushT，建在 Diffusion Policy 上。
- DemInf：`scripts/quality/estimate_quality.py`。JAX/OpenX，README 要求相对动作，没有 LIBERO。
- DataMIL：`scripts/perform_libero_dm.sh`。LIBERO 路径使用 Octo，另有 MetaWorld，后者不在本阶段。
- Demo-SCORE：`demo_score/demo_score/train_trajwise_classifier.py`。需要成功和失败 rollout。仓库根没有 LICENSE。
- STRAP：`strap/retrieval/retrieval.py` 与 `strap/embedding/encode_datasets.py`。LIBERO-10/90，子轨迹，DINOv2。策略训练在 `robomimic_strap`。仓库根没有 LICENSE。
- HinFlow：`scripts/train_planner.py`、`scripts/train_hinflow_policy.py`。README 的 LIBERO 任务名是 libero_butter、libero_book、libero_chocolate、libero_microwave。仓库根没有 LICENSE。子模块超时。
- Diffusion Policy：`diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`。
- RoboMimic、robosuite、LIBERO：基准源码。LIBERO 下载脚本在 `benchmark_scripts/download_libero_datasets.py`，本次没有运行。
