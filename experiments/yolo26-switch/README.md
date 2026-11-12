# 2026-11-12 单次切换与验证

脚本使用这台 Orin 的固定路径。`validate.py` 依赖前轮照片及 Ultralytics 输出，照片未入库。运行前将待验源码、辅助模块与 tests 暂存至 `runtime/yolo26-switch`，并使该目录 models 指向已验证模型。`check.sh` 在应用 venv 下跑测试及 8 组照片一致性，写 validated 标记。

`deploy.py` 是针对 25dac69 原版本的单次部署，验证旧文件哈希并备份，成功后保存 deployed.json。不要在已更新的实例上重复运行或绕过哈希检查。observe.py 只采集 90 秒状态元数据，随后 verify-preserved.py 使用部署备份检查旧记录保留；单独重跑 verify-preserved.py 使用现有观测文件，不再采集。

结果汇总和回滚见 ../../docs/YOLO26m上线.md。照片、数据库和逐条运行日志保持本机。
