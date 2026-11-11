#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
base=runtime/sensevoice-benchmark
archive=sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
runtime/uv pip install --index-url https://pypi.org/simple --python .venv/bin/python --only-binary :all: --target "$base/python" 'sherpa-onnx==1.13.8' >"$base/pip.log" 2>&1 &
pip_job=$!
wget -nv --timeout=20 --tries=3 -O "$base/model.tar.bz2" "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$archive.tar.bz2"
tar -xjf "$base/model.tar.bz2" -C "$base" "$archive/model.int8.onnx" "$archive/tokens.txt"
wait "$pip_job"
sha256sum "$base/$archive/model.int8.onnx" "$base/$archive/tokens.txt"
du -sh "$base/$archive/model.int8.onnx" "$base/python"
echo SENSEVOICE_FETCH_OK
