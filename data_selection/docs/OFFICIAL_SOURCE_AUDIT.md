# 官方源码审计

2026-11-03 用 `tools/fetch_sources.py` 对清单里的 pin 做了 `git rev-parse HEAD`。pin 没有改成上游新 HEAD。HEAD 与 pin 一致时 `pin_verified` 为 true。官方树仍被根 `.gitignore` 排除。

| 名称 | pin | HEAD 与 pin | 子模块 | 预期文件 |
| --- | --- | --- | --- | --- |
| cupid | `2941eba9427a3398d3cbe26b10c01e3ece56bd18` | 一致 | `third_party/trak` 已检出 `ff2fdc840dfdb51607b029777f16892a657b3978` | 在 |
| datamil | `f8fc4abccacd1e2b070e68128a7f2bce98f4b303` | 一致 | 无 | 在 |
| deminf | `232151defb258b5ceb0a6b62644574b57d981b58` | 一致 | 无 | 在 |
| demo_score | `b82a5ece97b8975643b14703734afa8f24ab7b0a` | 一致 | 无 | 在 |
| diffusion_policy | `5ba07ac6661db573af695b419a7947ecb704690f` | 一致 | 无 | 在 |
| hinflow | `f80eeae61bc3e1b32b03b63f06e4add3943a473f` | 一致 | 未完成 | `.gitmodules` 在 |
| libero | `8f1084e3132a39270c3a13ebe37270a43ece2a01` | 一致 | 无 | `benchmark_scripts/download_libero_datasets.py` 在 |
| robomimic | `d309eaecc18acf4152a830a895a6984b8ac71b05` | 一致 | 无 | 在 |
| robomimic_strap | `3386adb5b664e9bedf88e7d26ced986439e67193` | 一致 | 已初始化 | 在 |
| robosuite | `5ce6643f3092639d08f7b0f90ed1c6a84f50552c` | 一致 | 无 | 在 |
| strap | `a106da6cb178893923498f5cc0b4a51933c47ab3` | 一致 | 无 | 在 |

## HinFlow 子模块

父仓库 pin 已核对。`.gitmodules` 指向：

- `third_party/co-tracker`，当前检出 `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`，与 gitlink 不一致（`git submodule status` 前缀 `+`），工作区有本地修改 `M third_party/co-tracker`。这次没有 reset。
- `third_party/maniskill`，gitlink `ac59dcadacd307e7acbbce22c708b6b1a28fb7bc`，未检出。
- `third_party/robomimic`，gitlink `5dee58f9cc1235010d0877142b54d0e82dd23986`，未检出。2026-11-03 的 `git submodule update --init --depth 1 third_party/robomimic` 在 `--no-checkout` 阶段被中止，URL 是 `git@github.com:ARISE-Initiative/robomimic.git`。中止后没有 HEAD，没有 `train.py`，`.git/modules/robomimic` 不存在。
- `third_party/robosuite`，gitlink `eafb81f54ffc104f905ee48a16bb15f059176ad3`，未检出。

HinFlow 不是数据选择 baseline，也不能标成 fully ready。

## 入口

适配器会导入作者依赖。本机没有 torch、hydra、jax，导入失败就返回 `BLOCKED`，`official_stack_invoked=false`，不用随机分数代替。

- CUPID：`eval_demonstration_scores.py`、`train_trak_diffusion.py`、`finalize_trak.py`。
- DemInf：`scripts/quality/estimate_quality.py`。不把 `estimator=random` 当成替身。
- DataMIL：`scripts/perform_libero_dm.sh` 含 `--config.pretrained_path=hf://rail-berkeley/octo-small`。Octo 分数不是图像 Diffusion Policy 分数，统一比较保持 `BLOCKED`。
- Demo-SCORE：`demo_score/demo_score/train_trajwise_classifier.py`。失败动作不能当模仿标签，分类器训练轨迹不能进入评估。
- STRAP：`strap/retrieval/retrieval.py`、`strap/embedding/encode_datasets.py`。粒度是子轨迹。不会把作者示例里的 stove-pot 过滤当成 RoboMimic 结果。
