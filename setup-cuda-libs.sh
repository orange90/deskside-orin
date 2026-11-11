#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runtime/cosmos-cuda/debs
cd runtime/cosmos-cuda/debs
apt-get download libcusparse-12-6=12.5.3.3-1 libcufft-12-6=11.2.6.59-1 \
  libcufile-12-6=1.11.1.6-1 libcurand-12-6=10.3.7.68-1 \
  libcudnn9-cuda-12=9.3.0.75-1 cuda-cupti-12-6=12.6.68-1 \
  libnvjitlink-12-6=12.6.68-1 cuda-nvrtc-12-6=12.6.68-1
for deb in ./*.deb; do
  dpkg-deb -x "$deb" ..
  dpkg-deb --show "$deb"
done
