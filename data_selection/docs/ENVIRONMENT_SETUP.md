# 环境

本机记录在 `manifests/local_environment.json`：macOS arm64，Python 3.12.4，父虚拟环境没有 torch，`nvidia-smi` 不存在。没有做系统级安装，也没有建七套 Conda。

上游规格只抄进 `environments/*.yaml`，版本来自对应 README 或 `conda_environment.yaml`，没有升级：

- 统一 DP：`robodiff`，Python 3.9，CUDA toolkit 11.6，PyTorch 1.12.1
- CUPID：同代 PyTorch 栈，README 写 Ubuntu 20.04 与 mujoco210
- DemInf：Python 3.11，JAX 0.4.37 CUDA 12
- DataMIL：Python 3.10，TensorFlow 2.14.0，JAX 0.4.20，CUDA 11.8；LIBERO 策略是 Octo
- Demo-SCORE：Python 3.10，至少 demoscore / robodiff / aloha 三个环境
- STRAP：Python 3.10，另加 `robomimic_strap`
- HinFlow：检出成功之前不写版本号

Mac 上的测试用父环境：

```bash
cd data_selection
../.venv/bin/python -m pytest
```

父 `pyproject.toml` 的测试路径仍然只有 `idea_A`、`idea_B` 和 `shared`。
