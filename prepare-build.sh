#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs runtime models
if [[ ! -x runtime/uv ]]; then
  uv_url=https://files.pythonhosted.org/packages/96/72/da1119f94034d7173e6c8e2cd77a6299744458f0af3cb8e3a04d5cd419d1/uv-0.13.7-py3-none-manylinux_2_17_aarch64.manylinux2014_aarch64.musllinux_1_1_aarch64.whl
  wget -nv -O runtime/uv.whl "$uv_url"
  printf '%s  %s\n' a90f5d95716ab83091f0f73a8e7ec64e557abf2a28537fcd719b2ac88e6f725a runtime/uv.whl | sha256sum -c
  python3 -m zipfile -e runtime/uv.whl runtime/uv-wheel
  install -m 755 runtime/uv-wheel/uv-0.13.7.data/scripts/uv runtime/uv
fi
if [[ ! -x .venv/bin/python ]]; then
  runtime/uv venv --python /usr/bin/python3 --system-site-packages .venv
fi
runtime/uv pip install --python .venv/bin/python -r requirements-app.txt
for row in whisper.cpp:f92e1e4d1056700c39d4e86c1140f0a088fe1ab6 llama.cpp:7d7f0287530bc8981690f361d8c1627dbae1c99a; do
  IFS=: read -r repo revision <<< "$row"
  source="runtime/$repo-src"
  if [[ -d "$source" ]]; then
    if [[ "$(cat "$source/.revision")" != "$revision" ]]; then
      printf 'Unexpected revision in %s\n' "$source" >&2
      exit 1
    fi
    continue
  fi
  wget -nv -O "$source.tar.gz" "https://codeload.github.com/ggml-org/$repo/tar.gz/$revision"
  stage="$(mktemp -d "$source.XXXXXX")"
  tar -xzf "$source.tar.gz" --strip-components=1 -C "$stage"
  printf '%s\n' "$revision" > "$stage/.revision"
  mv "$stage" "$source"
done
