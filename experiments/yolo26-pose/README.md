# YOLO26 Pose 独立照片回放

仅在 Jetson 上用已存在照片比较候选模型，不采集新画面、不切换生产模型、不写活动数据库。结果、模型和照片保留本机。

## 环境与文件

生产项目根目录默认 `/home/orange/Documents/orin-desk-agent`，实验目录为其下 `runtime/yolo26-eval`。

本次使用独立 venv，借用既有 CUDA PyTorch/TensorRT 的只读包路径；未更新生产环境的包。已验证环境：Python 3.10.12、torch 2.11.0、torchvision 0.26.0、Ultralytics 8.4.210、TensorRT 10.3.0、CUDA 12.6、ONNX 1.23.2。模型来自 Ultralytics 官方 assets 仓库 v8.4.0 release：`yolo26s-pose.pt`、`yolo26m-pose.pt`。

- `models/`：上述两份权重；TensorRT 引擎在目标 Orin 本机构建。
- `samples/`：standing.jpg、sitting.jpg、occluded.jpg、empty.png。原图及标签依据记录在本机 sample-manifest.json；空工位图是带界面覆盖文字的截图裁剪，非原始相机帧。
- `venv/bin/python`：独立环境，能导入 ultralytics、torch、torchvision、numpy、cv2、tensorrt、onnx。
- 将本目录脚本放入实验目录。`run.sh` 借用生产项目的 cosmos-env.sh 和 DLA 库路径，实际调用的是实验 venv。

Ultralytics `sync=false`、`YOLO_OFFLINE=true`、`YOLO_AUTOINSTALL=false`，无云端推理或照片上传。

## 复现

从实验目录执行，建议置于独立 tmux 会话；各模型串行测试，以免互相争抢 GPU：

```bash
bash run.sh --model yolo26s-pose --backend pt
bash run.sh --model yolo26m-pose --backend pt
bash run.sh --model rtmdet-rtmpose --backend baseline
bash fast-batch.sh  # 构建两个 FP16 引擎，再串行完成 engine 测速和 review 回放
```

`fast-batch.sh` 的 review 回放额外需要 `samples/standing-old.jpg`（历史未知站立照片）和既有 empty.png。`--review` 只输出图片和坐标，不参与速度统计。

默认 TensorRT 优化构建尝试约 15 分钟仍未结束，已停止该实验进程；最终两个引擎均使用 `builder_optimization_level=0`、2 GiB workspace、FP16，构建档一致，不代表充分调优后的极限速度。原有模型和生产服务未停止或替换。

## 测量口径

- 30W，生产视觉、Qwen、Cosmos 服务继续运行；并不代表语音和所有模型都满载。
- 4 张图片分别全图和工位 ROI（0.25,0.04,0.78,0.9）回放，共 8 种输入，不能当作 8 张独立样本。
- 模型输入固定 640×640，batch=1；检测阈值 0.28、NMS IoU 0.45，高置信关节阈值 0.55。不同模型分数未校准，点数不等于准确率。
- 每模型预热 12 次，再重复 25 轮，共 200 次计时。PyTorch CUDA 同步，时间含预处理、推理、后处理、结果转 CPU 和现有几何坐站判定；不含文件读取、绘图、摄像、Cosmos 复核、状态稳定和看板刷新。
- 每次调用后停 50ms 让出资源；报告中位/P95 延迟，不把延迟倒数说成连续摄像吞吐。
- 基线每次清空检测缓存，确保每张图都重新推理，无跨照片相对动作、无 Cosmos；它是 RTMDet/RTMPose 组件对照，非完整生产流程准确率。
- YOLO 的关键点缩放到与 RTMPose 相同的裁剪尺度，再交给现有 leg_posture 规则。YOLO 本身只输出框和关键点，不直接输出坐站。测试 JSON 中的 away 是“未检出人”的原始映射，生产环境还有遮挡/离开保护，二者不可混同。
- 仅有整图坐/站/在位标签，没有逐关节点人工真值；不可计算可信的关键点 mAP、连续坐站时长误差或状态切换 P95。

`results/<模型>-<后端>/` 保存原始坐标/置信度、逐次耗时、统计及标骨骼图片。RSS 是含 Python/PyTorch 框架的进程峰值，torch allocated 不包含全部 TensorRT/CUDA 内存，不能冒充模型整体显存占用。
