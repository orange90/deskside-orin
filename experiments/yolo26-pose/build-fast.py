import os
os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
import ast,json,time
from pathlib import Path
import onnx,tensorrt as trt
from ultralytics import YOLO,settings
settings.update({'sync':False})
root=Path('/home/orange/Documents/orin-desk-agent/runtime/yolo26-eval')
for name in ['yolo26s-pose','yolo26m-pose']:
    start=time.monotonic();print('START',name,'builderOptimizationLevel=0',flush=True)
    path=root/'models'/f'{name}.onnx'
    if not path.exists():
        YOLO(root/'models'/f'{name}.pt').export(format='onnx',device=0,imgsz=640,batch=1,half=False,simplify=False,opset=17,dynamic=False)
    meta={}
    for p in onnx.load(str(path)).metadata_props:
        try:meta[p.key]=ast.literal_eval(p.value)
        except (ValueError,SyntaxError):meta[p.key]=p.value
    logger=trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger,'')
    builder=trt.Builder(logger);network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(path.read_bytes()):raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config=builder.create_builder_config();config.builder_optimization_level=0
    config.set_flag(trt.BuilderFlag.FP16);config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,2*1024**3)
    engine=builder.build_serialized_network(network,config)
    assert engine is not None
    header=json.dumps(meta).encode()
    dest=root/'models'/f'{name}.engine'
    dest.write_bytes(len(header).to_bytes(4,'little')+header+bytes(engine))
    print('FINISH',name,'seconds',round(time.monotonic()-start,2),'bytes',dest.stat().st_size,flush=True)
