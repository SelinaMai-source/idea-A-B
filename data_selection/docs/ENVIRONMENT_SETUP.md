# 环境

2026-11-03 本机：macOS arm64，Python 3.12.4，父虚拟环境 `/Users/grantwright/Desktop/科研/10.30新课题/v0/.venv`。`torch` 不可导入。`nvidia-smi` 不存在。`conda` 在 `/opt/anaconda3/bin/conda`，但 `scripts/setup_environment.sh` 在 Darwin 上拒绝创建环境，退出码 2。没有改全局 pip 配置，没有系统级安装，没有装 `torch==1.12.1+cu116`。

上游规格仍在 `environments/*.yaml`，状态是 `SPEC_RECORDED_NOT_INSTALLED`。这些 YAML 不是已安装环境。

- `unified_dp` / `robodiff`：Python 3.9，作者文件里的 CUDA toolkit 11.6 与 PyTorch 1.12.1。这和 RTX 4090 不是同一个已验证环境。4090 上要先看 `nvidia-smi` 和 `torch.cuda.get_device_capability()`，再决定 wheel。不要把另一套 CUDA wheel 写成作者环境。
- `cupid`：Python 3.9，单独环境。
- `deminf`：Python 3.11，JAX 0.4.37。不和 PyTorch 混装。
- `datamil`：Python 3.10，TensorFlow 2.14.0，JAX 0.4.20，CUDA 11.8。LIBERO 原文策略是 Octo。
- `demoscore`：Python 3.10，作者要求 demoscore / robodiff / aloha 分开。
- `strap`：Python 3.10，另加 `robomimic_strap`。
- `hinflow`：Python 3.10.14，PyTorch 2.4.0。子模块未齐，不能当已就绪。

Linux 上一次只建一个环境：

```bash
cd data_selection
bash scripts/setup_environment.sh --env unified_dp
bash scripts/setup_gpu.sh
```

PyPI 只用当次命令的 `-i https://pypi.org/simple`，失败再试 `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`。不要改 `pip.conf`。

Mac 上的 CPU 测试：

```bash
cd data_selection
../.venv/bin/python -m pytest
```
