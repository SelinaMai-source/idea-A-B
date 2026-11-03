# 未解决

- 本机 macOS arm64，父环境 `ModuleNotFoundError: No module named 'torch'`。`nvidia-smi` 不存在。图像 Diffusion Policy 没有训练。
- `robomimic_image.zip` HEAD 200，Content-Length 84754919326。没有下载。下一步在 Linux 上：`bash scripts/download_robomimic.sh image`。
- `robomimic_lowdim.zip` HEAD 200，Content-Length 1930261710。本机下载器拒绝正文。低维结果不能写成图像 DP。
- LIBERO HDF5 未下载。官方入口在 `benchmarks/_sources/libero/benchmark_scripts/download_libero_datasets.py`。不要用 `lerobot/libero` 代替。
- Lift MH `latest.ckpt` 与 `epoch=0250-test_mean_score=1.000.ckpt` 的 HEAD Content-Length 都是 4627773609。正文未下载。后者不能当 `INITIAL_CHECKPOINT`。Can / Square / Transport 的 checkpoint 文件名还没有列出来。
- `lift_image.yaml` 的 `dataset_type` 是 `ph`。本机没有 MH HDF5，所以 Lift MH 是 `BLOCKED`，不是成功。
- CUPID 卡在 `torch` / `hydra` / `click` 导入。DemInf 卡在 `jax`。Demo-SCORE 和 STRAP 卡在 `torch`。DataMIL 的 `scripts/perform_libero_dm.sh` 使用 Octo，统一 DP 适配没有实现。都没有退回随机分数。
- HinFlow `third_party/co-tracker` 有本地修改，检出 `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`，没有 reset。`third_party/maniskill`（`ac59dcadacd307e7acbbce22c708b6b1a28fb7bc`）和 `third_party/robosuite`（`eafb81f54ffc104f905ee48a16bb15f059176ad3`）没有开始检出。`third_party/robomimic`（gitlink `5dee58f9cc1235010d0877142b54d0e82dd23986`）用 `git submodule update --init --depth 1 third_party/robomimic` 重试，远程是 `git@github.com:ARISE-Initiative/robomimic.git`。约 3.5 分钟时 pack 约 9MB，工作区仍是 `--no-checkout`，不能记成已完成。
- `conda` 在 `/opt/anaconda3/bin/conda`，但 Darwin 上 `bash scripts/setup_environment.sh` 退出码 2，七套环境都没有创建。
- RTX 4090 与作者 CUDA 11.6 / PyTorch 1.12.1 的组合没有在这台机器上验证，也不写成已经兼容。
- 没有实现 ours / v1。没有论文成功率。
