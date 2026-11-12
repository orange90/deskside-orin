"""Build the pinned static FP16 engine on the target Orin (TensorRT 10.3)."""
import ast
import json
from pathlib import Path
import onnx
import tensorrt as trt

path = Path('models/yolo26m-pose.onnx')
metadata = {}
for item in onnx.load(str(path)).metadata_props:
    try:
        metadata[item.key] = ast.literal_eval(item.value)
    except (ValueError, SyntaxError):
        metadata[item.key] = item.value
assert metadata.get('task') == 'pose' and metadata.get('kpt_shape') == [17, 3]
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, '')
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
if not parser.parse(path.read_bytes()):
    raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
assert tuple(network.get_input(0).shape) == (1, 3, 640, 640)
assert tuple(network.get_output(0).shape) == (1, 56, 8400)
config = builder.create_builder_config()
config.builder_optimization_level = 0
config.set_flag(trt.BuilderFlag.FP16)
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024**3)
engine = builder.build_serialized_network(network, config)
if engine is None:
    raise RuntimeError('TensorRT build failed')
header = json.dumps(metadata).encode()
dest = path.with_suffix('.engine')
temporary = dest.with_suffix('.engine.part')
temporary.write_bytes(len(header).to_bytes(4, 'little') + header + bytes(engine))
temporary.replace(dest)
print('Built', dest, dest.stat().st_size, 'bytes', flush=True)
