#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source ./cosmos-env.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .cosmos/bin/python - <<'PY'
import json
from pathlib import Path
import torch
import torchvision
assert torch.cuda.is_available()
x = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
assert (x @ x)[0, 0].item() == 64
result = {"torch": torch.__version__, "cuda": torch.version.cuda,
          "torchvision": torchvision.__version__, "gpu": torch.cuda.get_device_name(0),
          "bf16_matmul": True}
path = Path("runtime/cosmos-cuda/gpu-ready.json")
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(result) + "\n")
tmp.replace(path)
print(json.dumps(result))
PY
