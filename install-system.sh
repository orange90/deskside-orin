#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get install -y --no-install-recommends \
  cuda-nvcc-12-6 cuda-cudart-dev-12-6 libcublas-dev-12-6 \
  libnvinfer-bin python3-libnvinfer docker.io nvidia-container-toolkit \
  cmake v4l-utils tmux
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker
usermod -aG docker "${SUDO_USER:?Run this script via sudo}"
dpkg-query -W cuda-nvcc-12-6 libnvinfer10 docker.io nvidia-container-toolkit tmux
