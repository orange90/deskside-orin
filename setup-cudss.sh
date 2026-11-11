#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runtime/cosmos-cuda/cudss
cd runtime/cosmos-cuda/cudss
archive=libcudss-linux-aarch64-0.7.1.4_cuda12-archive.tar.xz
if [[ ! -f "$archive" ]]; then
  wget -nv -T 40 -O "$archive.part" "https://developer.download.nvidia.cn/compute/cudss/redist/libcudss/linux-aarch64/$archive"
  mv "$archive.part" "$archive"
fi
printf '%s  %s\n' f283c31b6badf4a5277014295705eac8e6e28f27de30d746719ea1cef7a750b8 "$archive" | sha256sum -c
tar -xJf "$archive" --strip-components=1
ls -l lib/libcudss.so.0
