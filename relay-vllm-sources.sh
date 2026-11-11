#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p vllm-probe
relay() {
  local name="$1" repo="$2" rev="$3" file="vllm-probe/$1.tar.gz"
  if ssh -n orin "test -s 'Documents/orin-desk-agent/runtime/vllm-vendor/$name/.revision'"; then
    return
  fi
  curl -fLsS --retry 1 --max-time 100 "https://codeload.github.com/$repo/tar.gz/$rev" -o "$file.part"
  mv "$file.part" "$file"
  ssh orin 'mkdir -p Documents/orin-desk-agent/runtime/vllm-vendor'
  scp -q "$file" "orin:Documents/orin-desk-agent/runtime/vllm-vendor/$name.tar.gz"
  ssh orin "cd Documents/orin-desk-agent/runtime/vllm-vendor && mkdir -p '$name' && tar -xzf '$name.tar.gz' --strip-components=1 -C '$name' && printf '%s\n' '$rev' > '$name/.revision'"
  printf '%s %s ready\n' "$name" "$rev"
}
relay cutlass nvidia/cutlass v4.4.2 &
relay triton triton-lang/triton v3.5.1 &
relay deepgemm deepseek-ai/DeepGEMM a6b593d2826719dcf4892609af7b84ee23aaf32a &
relay msa vllm-project/MSA 2e63ec37a0fc29bc20f39cd1a52e0f5affc33a73 &
relay flashmla vllm-project/FlashMLA b70aff3d110a2b1a037e62eac295166b5143643a &
relay qutlass IST-DASLab/qutlass 830d2c4537c7396e14a02a46fbddd18b5d107c65 &
relay flashattn vllm-project/flash-attention bb9a72e7dde0dc614ffc663e052cd6a19ce73a42 &
relay flash-cutlass NVIDIA/cutlass 62750a2b75c802660e4894434dc55e839f322277 &
status=0
for pid in $(jobs -p); do wait "$pid" || status=1; done
exit "$status"
