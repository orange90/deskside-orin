#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./cosmos-env.sh
python3 - <<'PY'
from pathlib import Path

root = Path("runtime/vllm-cosmos-src/vllm")
processor = root / "transformers_utils/processors/cosmos3_edge.py"
source = processor.read_text()
old = "from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict"
new = old + ", get_image_size"
if new not in source:
    assert source.count(old) == 1 and source.count("    get_image_size,\n") == 1
    processor.write_text(source.replace(old, new).replace("    get_image_size,\n", ""))

model = root / "model_executor/models/cosmos3_edge.py"
source = model.read_text()
old = '            ".norm_added_k.": None,'
new = '            ".k_norm_und_for_gen.": None,'
if new not in source:
    assert source.count(old) == 1
    model.write_text(source.replace(old, old + "\n" + new))
PY
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.0.0+orin.sm87.894ebb27 \
  runtime/uv pip install --python .vllm-runtime/bin/python --no-deps --no-build-isolation \
  -e runtime/vllm-cosmos-src
