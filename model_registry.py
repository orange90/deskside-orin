"""Allowlisted local model choices. Changes are atomic and do not touch activity DB."""
import json
import os
from functools import lru_cache
from pathlib import Path
import store

ASR = {
    'small': {'name': 'Whisper small', 'path': 'models/ggml-small.bin', 'note': '当前基线 · 中文现场准确率仍待改善'},
    'base': {'name': 'Whisper base', 'path': 'models/ggml-base.bin', 'note': '轻量模型 · 中文准确率可能下降'},
    'turbo': {'name': 'Whisper large-v3 Turbo Q5', 'path': 'runtime/whisper-turbo-benchmark/ggml-large-v3-turbo-q5_0.bin', 'note': '实验候选 · 已做合成对照，真人未验收；可能较慢'},
}


# Conservative deployment budgets, not per-process measurements. Shared runtimes overlap.
MEMORY_BUDGET = {'small': [1.0, 2.0], 'base': [0.5, 1.0], 'turbo': [1.5, 3.0],
                 'detector': [0.2, 0.6], 'pose': [0.15, 0.5], 'qwen': [3.5, 6.0], 'cosmos': [10.0, 16.0]}


@lru_cache(maxsize=32)
def weight_bytes(root, relative):
    path = Path(root)/relative
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and p.suffix in {'.safetensors', '.bin', '.gguf', '.plan', '.onnx'})
    return None


def metadata(key, path):
    return {'weights_bytes': weight_bytes(str(store.ROOT), path), 'memory_gib': MEMORY_BUDGET[key],
            'memory_note': '规划估算，非逐模型实测。含权重与推理缓存预算，随输入长度和后端变化；共享内存不可直接相加。'}


def current():
    try:
        selected = json.loads((store.ROOT/'model-selection.json').read_text())['asr']
        if selected in ASR:
            return selected
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return next((key for key, value in ASR.items() if Path(value['path']).name == os.environ.get('ORIN_WHISPER_MODEL', 'ggml-small.bin')), 'small')


def asr_path():
    return store.ROOT / ASR[current()]['path']


def catalog():
    selected = current()
    components = [
        {'id': 'detector', 'title': '人体检测', 'model': 'RTMDet-tiny', 'engine': 'TensorRT FP16', 'description': '从工位画面定位人物，为姿态估计提供裁剪区域。', 'path': 'models/rtmdet.plan'},
        {'id': 'pose', 'title': '姿态估计', 'model': 'RTMPose-s', 'engine': 'TensorRT FP16', 'description': '结合相对关节几何、起落轨迹判断坐站；不依赖固定图像高度。', 'path': 'models/rtmpose.plan'},
        {'id': 'cosmos', 'title': '视觉复核', 'model': 'Cosmos3-Edge Reasoner', 'engine': '本机 CUDA', 'description': '对选中的帧或短片段复核姿态、生成画面摘要。隐私保护时暂停。', 'path': 'models/cosmos'},
        {'id': 'asr', 'title': '语音识别', 'model': ASR[selected]['name'], 'engine': 'whisper.cpp · CUDA', 'description': '按下录音后才采集 Orin 麦克风。中文转写交给本地任务路由。', 'path': ASR[selected]['path']},
        {'id': 'qwen', 'title': '问答与任务', 'model': 'Qwen3.5-4B', 'engine': 'llama.cpp · Q4_K_M', 'description': '回答开放问题并解析任务。坐站统计与明确的待办、提醒优先走确定性规则。', 'path': 'models/qwen.gguf'},
    ]
    for component in components:
        path = component.pop('path')
        component['installed'] = (store.ROOT/path).exists()
        component.update(metadata(selected if component['id'] == 'asr' else component['id'], path))
        component['selected'] = selected if component['id'] == 'asr' else component['id']
        if component['id'] == 'asr':
            component['options'] = [dict(id=key, name=value['name'], note=value['note'],
                                         installed=(store.ROOT/value['path']).is_file(), **metadata(key, value['path'])) for key,value in ASR.items()]
        else:
            component['options'] = [{'id': component['id'], 'name': component['model'], 'installed': component['installed'],
                                     'note': '本机目前只有此兼容模型；新增模型需先适配与验证', **metadata(component['id'], path)}]
    return {'components': components}


def select_asr(key):
    if key not in ASR or not (store.ROOT/ASR[key]['path']).is_file():
        raise ValueError('该模型未安装或不兼容')
    path = store.ROOT/'model-selection.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'asr': key}, ensure_ascii=False)+'\n')
    os.replace(temporary, path)
