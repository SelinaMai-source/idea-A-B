#!/usr/bin/env bash
# Create isolated Linux environments from environments/*.yaml.
# On Darwin this script prints the commands and does not install CUDA stacks.
# It does not run apt, does not edit pip.conf, and does not install torch+cu116 on macOS.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="${1:-}"
if [[ "${1:-}" == "--env" ]]; then
  ENV_NAME="${2:-}"
fi

plan() {
  cat <<'EOF'
# unified image CNN Diffusion Policy (author pins, not an RTX 4090 claim)
conda create -n robodiff python=3.9 pip=22.2.2 -y
conda activate robodiff
# Author conda env requests cudatoolkit 11.6 and pytorch 1.12.1.
# Do not pip install torch==1.12.1+cu116 on Darwin.
# On an RTX 4090, record nvidia-smi and torch.cuda.get_device_capability()
# before choosing a wheel. A newer CUDA wheel is a separate env, not the author env.
pip install -i https://pypi.org/simple torch==1.12.1 torchvision==0.13.1 --index-url https://download.pytorch.org/whl/cu116 || true

# original-paper envs stay separate from robodiff
conda create -n cupid python=3.9 -y
conda create -n deminf python=3.11 -y
conda create -n datamil python=3.10 -y
conda create -n demoscore python=3.10 -y
conda create -n strap python=3.10 -y
conda create -n hinflow python=3.10 -y
EOF
}

if [[ "$(uname -s)" == "Darwin" ]]; then
  echo "setup_environment.sh refuses to create CUDA or JAX stacks on Darwin." >&2
  plan >&2
  if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not on PATH." >&2
  else
    echo "conda is present at $(command -v conda) but no environment was created." >&2
  fi
  exit 2
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not on PATH. No environment was created." >&2
  plan >&2
  exit 2
fi

echo "Linux conda was found. This script does not install until ENV is one of the named stacks." >&2
case "${ENV_NAME}" in
  unified_dp|robodiff)
    conda create -n robodiff python=3.9 pip=22.2.2 -y
    echo "Created robodiff with Python 3.9. Install PyTorch from the author file only after nvidia-smi." >&2
    ;;
  cupid_original|cupid)
    conda create -n cupid python=3.9 -y
    ;;
  deminf_original|deminf)
    conda create -n deminf python=3.11 -y
    ;;
  datamil_original|datamil)
    conda create -n datamil python=3.10 -y
    ;;
  demoscore_original|demoscore)
    conda create -n demoscore python=3.10 -y
    ;;
  strap_original|strap)
    conda create -n strap python=3.10 -y
    ;;
  hinflow_original|hinflow)
    conda create -n hinflow python=3.10 -y
    ;;
  "")
    echo "Pass --env unified_dp (or another environments/*.yaml name) to create one env." >&2
    plan >&2
    exit 2
    ;;
  *)
    echo "Unknown env ${ENV_NAME}." >&2
    exit 2
    ;;
esac
