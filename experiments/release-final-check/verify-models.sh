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
from vision import Model
for name, shape in (("rtmdet",(640,640,3)),("rtmpose",(256,192,3))):
    result=Model(name)(np.zeros(shape,dtype=np.uint8),(0,0,0),(1,1,1))
    assert len(result)==2 and all(x.size and np.isfinite(x).all() for x in result)
    print(name, [x.shape for x in result], flush=True)
PY
bash offline-model-smoke.sh --with-qwen
test ! -e data.sqlite3 && test ! -e calibration.json
echo ARCHIVE_RUNTIME_OK
