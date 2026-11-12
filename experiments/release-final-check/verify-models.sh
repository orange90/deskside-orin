#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source_root="$(realpath ../..)"
tree="$(mktemp -d "$PWD/release.XXXXXX")"
unzip -q release.zip -d "$tree"
cd "$tree"
for dir in .venv .cosmos models runtime; do
  ln -s "$source_root/$dir" "$dir"
done
test ! -e data.sqlite3 && test ! -e calibration.json
for f in ./*.sh; do bash -n "$f"; done
sha256sum -c MODEL-SHA256SUMS
export LD_LIBRARY_PATH="$PWD/runtime/dla/usr/lib/aarch64-linux-gnu/nvidia:$PWD/runtime/dla/usr/local/cuda-12.6/targets/aarch64-linux/lib:/usr/local/cuda/lib64"
.venv/bin/python - <<'PY'
import numpy as np
from pathlib import Path
from yolo_pose import YoloPose
model=YoloPose(Path.cwd())
boxes,joints=model(np.zeros((640,640,3),dtype=np.uint8))
assert boxes.shape[1:]==(5,) and joints.shape[1:]==(17,3)
assert np.isfinite(boxes).all() and np.isfinite(joints).all()
print(model.info, boxes.shape, joints.shape, flush=True)
PY
bash offline-model-smoke.sh --with-qwen
test ! -e data.sqlite3 && test ! -e calibration.json
echo ARCHIVE_RUNTIME_OK
