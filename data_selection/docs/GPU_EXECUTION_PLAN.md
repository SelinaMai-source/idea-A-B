# Linux GPU 执行计划

本机没有跑 GPU 训练，也没有观察到 OOM。

Diffusion Policy 的 `lift_image.yaml` 注释写着：该评估配置需要 16 核、64GB 内存。图像 zip 的 `Content-Length` 是 84754919326 字节。LIBERO 数据集的 Hugging Face `usedStorage` 是 99830157042 字节，而且包含 libero_10、libero_90、libero_goal、libero_object、libero_spatial，不是只有 LIBERO-10。解压后的体积没有测量。

优先机器：1 张 RTX 4090 24GB，内存至少 64GB，系统盘之外的数据盘要能放下图像 zip 和将要使用的 LIBERO 子集。24GB 是否够训练没有在这台机器上验证。

先做检查，再下载，再训练。训练命令在预检失败时不要执行。

```bash
cd /path/to/v0
git checkout data_selection_v0
cd data_selection
export PYTHONPATH="$PWD"
export DATA_ROOT=/data/data_selection
python3 tools/inspect_environment.py --require-cuda
python3 tools/preflight.py --require-cuda
python3 tools/fetch_sources.py
bash scripts/download_robomimic.sh lowdim
bash scripts/download_robomimic.sh image
python3 tools/verify_assets.py
```

LIBERO 只在 `benchmarks/_sources/libero/benchmark_scripts/download_libero_datasets.py` 真实存在时运行。不要换一条没核对过的路径。

```bash
cd benchmarks/_sources/libero
python benchmark_scripts/download_libero_datasets.py --datasets libero_10
HF_ENDPOINT=https://hf-mirror.com python benchmark_scripts/download_libero_datasets.py --use-huggingface --datasets libero_10
```

已核对的 Lift MH 图像 CNN 权重目录：

```bash
wget --recursive --no-parent --no-host-directories --relative --reject='index.html*' \
  https://diffusion-policy.cs.columbia.edu/data/experiments/image/lift_mh/diffusion_policy_cnn/
```

`can_mh`、`square_mh`、`transport_mh` 的 checkpoint 文件名还没列目录，不要把 Lift 的 `epoch=0250-test_mean_score=1.000.ckpt` 套到那些任务上。

预检通过之后，仍然先 dry-run。dry-run 只写计划，不训练：

```bash
export PYTHONPATH="$PWD"
python3 scripts/run_unified_baselines.py \
  --method random --benchmark robomimic_lift_mh --seed 0 \
  --device cuda --dry-run --output-dir "$DATA_ROOT/plans/random_lift"
python3 scripts/run_original_baselines.py \
  --method cupid --benchmark robomimic_lift_mh --seed 0 \
  --device cuda --dry-run --output-dir "$DATA_ROOT/plans/cupid"
python3 scripts/run_online_baselines.py \
  --method random --benchmark libero_10 --seed 0 \
  --device cuda --dry-run --output-dir "$DATA_ROOT/plans/online"
python3 scripts/run_continual_learning.py \
  --method random --benchmark libero_continual --seed 0 \
  --device cuda --dry-run --output-dir "$DATA_ROOT/plans/continual"
```

论文方法在官方依赖安装并真正调用入口之前会保持 `BLOCKED`，不会退回 Random。
