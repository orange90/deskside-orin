# Qwen3.5-4B 视觉独立回放

只读复用生产 Q4_K_M 语言权重，另行加载 F16 视觉编码器，在 loopback 8083 启动独立 llama.cpp 服务。不采集新画面、不改变线上模型、不回填历史。模型离线加载；图片以 base64 发送到 Orin 自己的 127.0.0.1。

## 准备

在现有项目根目录的 `runtime/qwen35-vision-eval/` 放入本目录脚本。沿用已验证的 `runtime/llama-server`（本机报告 0.9.0-dev/build0/commit unknown，不冒充有可追溯构建号），现有 `models/qwen.gguf` 为 Canfield/Qwen3.5-4B-Q4_K_M-GGUF，SHA256 见项目 MODEL-SHA256SUMS。

视觉编码器来自 `unsloth/Qwen3.5-4B-GGUF` 固定 revision `e87f176479d0855a907a41277aca2f8ee7a09523` 的 `mmproj-F16.gguf`，大小 672423616 bytes，SHA256 `cd88edcf8d031894960bb0c9c5b9b7e1fea6ebee02b9f7ce925a00d12891f864`。来源元数据见 projector-source.json。GGUF 标明 Qwen3.5-4B、projection_dim=2560，与既有语言权重 embedding_length=2560 一致；已实测成功加载图像并推理。

输入沿用 `runtime/yolo26-eval/samples/` 中五张已有照片。脚本运行环境使用上一轮独立 `runtime/yolo26-eval/venv/bin/python`，此轮未修改任何 Python 依赖。

## 执行

在 Orin 项目目录运行，使用独立 tmux 会话：

```bash
python3 runtime/qwen35-vision-eval/audit.py before
tmux new-session -d -s orin-qwen35-vision -n server \
  'bash runtime/qwen35-vision-eval/launch.sh > runtime/qwen35-vision-eval/server.log 2>&1'
# 等待 http://127.0.0.1:8083/health 就绪，再顺序运行：
runtime/yolo26-eval/venv/bin/python runtime/qwen35-vision-eval/evaluate.py --smoke
runtime/yolo26-eval/venv/bin/python runtime/qwen35-vision-eval/evaluate.py
runtime/yolo26-eval/venv/bin/python runtime/qwen35-vision-eval/evaluate.py --compact
tmux send-keys -t '=orin-qwen35-vision:=server' C-c
python3 runtime/qwen35-vision-eval/audit.py after
```

确认实验 8083 已关闭、原 8765/8081/8082 均就绪。首次实际执行的 server 窗口名为 0，复现示例已显式命名 server，避免歧义。脚本结果采用追加写入，重新执行时应另建结果目录或备份上一轮结果，避免把重复试验混为一次统计。

## 协议

- 30W，完整照片，max image tokens=1024，context=4096，GPU offload=99，默认视觉 GPU 卸载；语言 Q4_K_M、视觉 F16。
- 不提供旧状态、人工答案、样本文件名。U02 源自旧看板截图，先遮盖原标签和底部提示文字，变换在脚本中固定并另存；其余四图仅无损重编码 PNG。
- temperature=0、seed=42、关闭思考、关闭提示缓存，所有返回 cached_tokens 均为 0。
- 首图单列。随后 5 图随机顺序重复 3 轮，每配置 15 次，计时从本机 HTTP 请求开始到完整结果；不含文件读取/预先编码，包含视觉编码、提示处理、输出与本机传输。
- 正式协议输出 presence/posture/evidence；额外 compact 同时缩短提示、改成单一 state 字段并删除解释。这是另一个实验条件，不是纯粹只改变输出长度，不能将结果变化单独归因于“没有解释”。
- 图片与逐条结果仅本机保存，不进入 GitHub。重复同一图片 3 次不能当作 15 个独立准确率样本，也不代表连续场景泛化能力。

将 results/、inputs/ 与协议复制到 Mac 本机 `outputs/orin-desk-agent/evidence/57-qwen35-vision/` 后，可运行 summarize.py、make-report.py 生成统计与 HTML（脚本按该相对路径读取）。首次运行完整协议的文件名为 trials.jsonl，后续脚本统一为 trials-explained.jsonl；汇总兼容两者。
