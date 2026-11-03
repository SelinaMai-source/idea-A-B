# Linux GPU 执行计划

本机没有跑图像 Diffusion Policy，也没有 OOM 记录。下面的命令和仓库里的脚本一致。预检失败时训练脚本自己退出。

第一笔真实实验是 RoboMimic Lift MH、图像 CNN Diffusion Policy、All Data、seed 0。数据集必须是 MH，不能把 `lift_image.yaml` 的 `dataset_type: ph` 当成 MH。

```bash
cd /path/to/v0
git checkout data_selection_v0
cd data_selection
export PYTHONPATH="$PWD"
bash scripts/setup_environment.sh --env unified_dp
export ROBOMIMIC_LIFT_MH_DATASET=/data/robomimic/lift/mh/image.hdf5
bash scripts/run_lift_mh_all_data.sh
```

`scripts/run_lift_mh_all_data.sh` 在 Darwin 上直接退出。在 Linux 上它先跑：

```bash
python tools/preflight.py --require-cuda --require-dataset "$ROBOMIMIC_LIFT_MH_DATASET" --expect-variant mh --expect-task lift
python scripts/run_unified_baselines.py \
  --method all_data \
  --benchmark robomimic_lift_mh \
  --task lift \
  --seed 0 \
  --config configs/experiments/robomimic_lift_mh_all_data_seed0.yaml \
  --output-dir results/unified/robomimic_lift_mh/all_data/seed_0 \
  --device cuda \
  --retention-ratio 1.0
```

`--dry-run` 不训练。不要加 `--synthetic`。没有数据集或没有 CUDA 时进程退出码是 2，`paper_result` 为 false。

图像 zip 只在 Linux 上、并且磁盘够 84754919326 字节时下载：

```bash
bash scripts/download_robomimic.sh image
```

不要在 macOS 上执行这条。低维包也不是这次图像实验的结果。Lift 的 `epoch=0250-test_mean_score=1.000.ckpt` 不能当作从零训练的初值，也不能用于 Can、Square、Transport。
