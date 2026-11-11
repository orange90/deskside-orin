#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
fetch() {
  local repo="$1" rev="$2" source="$3" dest="$4" size="${5:-0}"
  local remote="Documents/orin-desk-agent/models/$dest"
  if ssh -n -o BatchMode=yes orin "test -s '$remote' && { test '$size' = 0 || test \$(stat -c%s '$remote') = '$size'; }"; then return; fi
  echo "Downloading $dest"
  ssh -n -o BatchMode=yes orin "mkdir -p '${remote%/*}'"
  curl -fLsS --retry 2 --connect-timeout 15 "https://huggingface.co/$repo/resolve/$rev/$source" |
    ssh -o BatchMode=yes orin "cat > '$remote.part'"
  ssh -n -o BatchMode=yes orin "test -s '$remote.part' && { test '$size' = 0 || test \$(stat -c%s '$remote.part') = '$size'; } && mv '$remote.part' '$remote'"
  echo "Ready $dest"
}
case "$1" in
  cosmos)
    rev=344d602
    curl -fsSL "https://huggingface.co/api/models/nvidia/Cosmos3-Edge/tree/$rev?recursive=true&expand=false" > cosmos-tree.json
    while IFS=$'\t' read -r f bytes; do
      fetch nvidia/Cosmos3-Edge "$rev" "$f" "cosmos/$f" "$bytes"
    done < <(python3 -c 'import json; print(*["{}\t{}".format(x["path"], x["size"]) for x in json.load(open("cosmos-tree.json")) if x["type"]=="file" and x["path"].startswith(("transformer/","vision_encoder/")) or x["type"]=="file" and "/" not in x["path"]],sep="\n")')
    ;;
  qwen)
    fetch Canfield/Qwen3.5-4B-Q4_K_M-GGUF 4bc3afea43e5c81d17b8df3ef09c8bfdb9ff56f2 Qwen3.5-4B-Q4_K_M.gguf qwen.gguf 2707513696
    echo '25082a7dd3776cc3c741c6347d3bd04523f05796607b3fbc32fa3a25dfa1418c  Documents/orin-desk-agent/models/qwen.gguf' | ssh orin 'sha256sum -c -'
    ;;
  small)
    fetch bukuroo/RTMDet-ONNX aa91dbccc283db36b2c250a1d91ba9518db553da rtmdet-tiny-coco.onnx rtmdet.onnx
    fetch ziq/rtm 6bdd0110307167f5f003e92e34099a21bd61283c rtmpose-s.onnx rtmpose.onnx
    for f in ggml-base.bin ggml-small.bin; do fetch ggerganov/whisper.cpp 5359861c739e955e79d9a303bcbc70fb988958b1 "$f" "$f"; done
    ;;
  voice)
    fetch ggml-org/whisper-vad 9ffd54a1e1ee413ddf265af9913beaf518d1639b ggml-silero-v6.2.0.bin ggml-silero-v6.2.0.bin
    ;;
esac
