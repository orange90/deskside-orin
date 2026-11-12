# YOLO26m-Pose 上线记录 · 2026-11-12

用户在照片比较后明确要求将线上姿态检测切换为 YOLO26m-Pose。本次同时替换 RTMDet-tiny 人体检测和 RTMPose-s 关键点估计，因为 YOLO26m-Pose 一次推理就输出人体框与 17 个关节。坐站几何规则、相对起落动作、Cosmos 复核、遮挡/离开保护、最多 3 秒短暂未知补齐、隐私保护与历史数据库保持既有逻辑。Qwen 视觉实验没有上线。

## 实现

- `yolo_pose.py`：直接调用 TensorRT，生产环境不引入 PyTorch/Ultralytics；640×640 居中 letterbox、RGB、float32 归一化输入，FP16 引擎；输出 `(1,56,8400)`；人体置信阈值 0.28、NMS IoU 0.45。反算回工位 ROI 和完整画面坐标。引擎带 JSON 元数据头，启动验证 pose/17 个关节。
- `vision.py`：每次从当前帧联合获取框和骨骼，取消原先检测框的 0.45 秒缓存；关节按原有尺度送入坐站规则，ROI/摄像头变化清空相对运动历史。
- 看板只有一个联合视觉模型节点，显示实际引擎体积 56,185,131 bytes（53.6 MiB）和明确标注的内存规划估算；实时画面显示模型名称、耗时和骨骼。预览仍按需每秒 1 帧，视觉采样约 6 FPS。
- `launch-ready.sh` 等待新引擎；旧计划文件留在 Orin 供回滚，没有删模型或历史记录。

## 验证与边界

| 检查 | 结果 |
|---|---|
| Orin 回归测试 | 23 项全部通过，含坐站连续性、人工标签及新解码/坐标测试 |
| Mac 回归测试 | 23 项中 20 项通过、3 项因无 OpenCV 跳过；这 3 项已在 Orin 实测通过 |
| 历史照片后端一致性 | 4 张照片 × 全图/工位 ROI，共 8 组；框和高置信关节与原 Ultralytics 引擎参考输出相差不超过 1.1 像素、置信度误差 ≤0.003 |
| 原生后端照片测试 | 预热后 80 次，Vision.sample 墙钟耗时中位 51.17 ms，P95 60.50 ms；含前后处理，不含摄像/Cosmos/看板 |
| 线上约 90 秒观测 | 每 2 秒获取一次共 45 个状态样本；全部 ready、非隐私暂停、模型名均正确；由帧计数与采集时间差计算平均 6.169 FPS，达到 ≥5 FPS 目标 |
| 线上模型耗时 | 45 个采样值中位 106.68 ms、P95 111.79 ms；包含预处理、推理、后处理；期间 Cosmos 同时工作，不代表每帧全量延迟分布 |
| 数据保留 | 切换前 4037 条活动区间的 kind/state/start 均保留，end 只允许自然延长；1169 条摘要、2 条待办、配置与 1 条人工标注保留 |
| 浏览器 | 新模型节点和实际体积正确，实时骨骼显示成功，截图该帧 14/17 个高置信关节；390 px 手机无横向溢出，无脚本错误；隐私清屏/恢复脚本通过 |
| 当次硬件快照 | 整机可用 RAM 约 41.1 GiB、swap 0、最高温度约 50.5°C；不是长时间峰值，也不是单模型内存测量 |

照片回放和线上模型耗时是不同输入与负载下的测量，不能互相替代。YOLO 输出关节点，不直接输出坐站；椅背完全遮挡仍可能漏检，坐站标签仍可能来自 Cosmos。此次证明模型接入、实时速度及数据保留，不代表坐站时长误差 ≤5% 或切换 P95 ≤3 秒已经通过人工验收。历史数据没有用新模型重算。

## 权重与重建

源权重为 Ultralytics assets v8.4.0 release 的 `yolo26m-pose.pt`，49,036,866 bytes，SHA256：

```text
2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4
```

已上线的 TensorRT 10.3/CUDA 12.6/Orin 引擎 SHA256：

```text
074860e9a8e7d94c857e6b803832229caf9d518739127bcf1097e473b32a57b5
```

引擎来自前一轮本机实验；本次复用已验证字节，没有重新编译。TensorRT 引擎应在目标平台重建，重建后的序列化哈希不保证相同。`MODEL-SHA256SUMS` 记录源权重，不强制不同构建的引擎哈希。

1. Mac 源码目录执行 `bash fetch-models.sh vision`，下载到 SSH 别名 orin 的 `~/Documents/orin-desk-agent/models/` 并验证 SHA256。
2. Orin 先按《重建与运行》准备 `.venv`、TensorRT 和 CUDA。若已有本轮验证的 ONNX，`rebuild-vision.sh` 直接使用；请勿把其他来源的同名 ONNX 混入 models。
3. 缺 ONNX 时，在隔离导出环境安装 Ultralytics 8.4.210、ONNX 1.23.2 及其兼容的 torch/torchvision，设置 `ORIN_EXPORT_PYTHON`。生产 `.venv` 不安装它们。现机已有 `runtime/yolo26-eval/venv/bin/python`，已验证 torch 2.11.0、torchvision 0.26.0。导出配置为 CPU、opset 17、batch 1、640×640、static、float32、simplify=false；已有实验 ONNX 使用同参数 CUDA 导出。CPU 导出不需要 Jetson CUDA PyTorch，可在匹配的隔离环境生成 ONNX 后复制到 Orin。
4. 在 tmux 中执行 `ORIN_EXPORT_PYTHON=runtime/yolo26-eval/venv/bin/python bash rebuild-vision.sh`。脚本准备 DLA 动态库，调用 `build-yolo-engine.py`：FP16、builderOptimizationLevel=0、workspace 2 GiB、输出元数据头与引擎。日志 `logs/yolo26m-build.log`。新重建脚本基于此前验证的构建流程整理，本次上线未再次执行完整重建。
5. 运行 `.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v`，再启动/重启 vision；核对 `/api/state` 的 `camera.vision_model`、`vision_ms` 和实时画面。`bash stack.sh start` 只补启动缺失服务，不会重启已有窗口。

仅下载权重/依赖需要外网，运行期从本机文件加载。生产推理没有 Ultralytics 联网检查和自动安装行为。

## 部署、证据与回滚

本次先暂存至 `runtime/yolo26-switch`，验证后用独立 tmux 会话执行部署。检查麦克风未录音，验证旧源码哈希，在线备份 SQLite 和代码；短暂设置 `.autostart-paused` 防止 cron 启动半更新服务，只重启 `orin-agent:vision`，保留 Qwen/Cosmos 窗口。健康失败自动恢复旧代码，数据库不回滚。

本机备份目录：`backups/yolo26m-20261112-105129`。日志：`runtime/yolo26-switch/check.log`、`deploy.log`、`verified-summary.log`；采样：`online-samples.json`、`online-summary.json`。最初数据库检查脚本误将 intervals.kind 当唯一键，修正为 kind/state/start 联合键后，用同一批采样和同一备份重新核对通过；没有修改数据库。

Mac 工作区证据目录 `outputs/orin-desk-agent/evidence/58-yolo26-switch/` 保存上述日志及 live-pose.png、model-modal.png、mobile.png。照片、数据库、逐条日志、权重与旧备份均不提交 GitHub。`experiments/yolo26-switch/` 保留此次验证/单次部署源码；其中部署脚本固定了旧版本哈希，不是可反复执行的通用升级工具。

需要回滚时，在 Orin 项目目录停止 vision 窗口并暂时暂停监督，恢复该备份中的 `vision.py`、`trt_model.py`、`main.py`、`model_registry.py`、`dashboard.html`、`launch-ready.sh`、`rebuild-vision.sh`、`fetch-models.sh`、`MODEL-SHA256SUMS`，再 `bash stack.sh start`。旧 `models/rtmdet.plan`、`models/rtmpose.plan` 仍在。不要恢复备份数据库覆盖新记录。新加的 yolo_pose.py 和 build-yolo-engine.py 在旧启动链上不会调用。
