import ctypes as c
import json

import numpy as np
import tensorrt as trt

CUDA = c.CDLL("libcudart.so.12")
LOGGER = trt.Logger(trt.Logger.WARNING)


def check(result):
    if result:
        raise RuntimeError(f"CUDA runtime error {result}")


def alloc(size):
    pointer = c.c_void_p()
    check(CUDA.cudaMalloc(c.byref(pointer), c.c_size_t(size)))
    return pointer


class TensorRT:
    def __init__(self, plan, shape, *, metadata_header=False, output_names=None):
        trt.init_libnvinfer_plugins(LOGGER, "")
        self.runtime = trt.Runtime(LOGGER)
        data = plan.read_bytes()
        self.metadata = {}
        if metadata_header:
            length = int.from_bytes(data[:4], 'little')
            if not 0 < length < min(1024*1024, len(data)-4):
                raise ValueError('Invalid engine metadata header')
            self.metadata = json.loads(data[4:4+length])
            data = data[4+length:]
        self.output_names = output_names
        self.shape = tuple(shape)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError(f"Cannot load {plan.name}")
        self.context = self.engine.create_execution_context()
        self.stream = c.c_void_p()
        check(CUDA.cudaStreamCreate(c.byref(self.stream)))
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        if self.engine.get_tensor_dtype(self.input) != trt.float32:
            raise ValueError('This runner requires float32 input bindings')
        self.context.set_input_shape(self.input, shape)
        self.input_buffer = alloc(int(np.prod(shape)) * 4)
        self.context.set_tensor_address(self.input, self.input_buffer.value)
        self.outputs = {}
        for name in names:
            if name != self.input:
                dims = tuple(self.context.get_tensor_shape(name))
                if min(dims) < 0:
                    raise RuntimeError(f"Unexpected dynamic tensor {name}: {dims}")
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                data = np.empty(dims, dtype=dtype)
                pointer = alloc(data.nbytes)
                self.outputs[name] = (data, pointer)
                self.context.set_tensor_address(name, pointer.value)

    def __call__(self, tensor):
        if tensor.shape != self.shape or tensor.dtype != np.float32 or not tensor.flags.c_contiguous:
            raise ValueError('Unexpected TensorRT input shape, dtype or layout')
        check(CUDA.cudaMemcpyAsync(self.input_buffer, c.c_void_p(tensor.ctypes.data),
                                   c.c_size_t(tensor.nbytes), 1, self.stream))
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT execution failed")
        check(CUDA.cudaStreamSynchronize(self.stream))
        outputs = {}
        for name, (data, pointer) in self.outputs.items():
            check(CUDA.cudaMemcpy(c.c_void_p(data.ctypes.data), pointer,
                                  c.c_size_t(data.nbytes), 2))
            outputs[name] = data
        names = self.output_names or (("1062", "1035") if "1062" in outputs else ("simcc_x", "simcc_y"))
        return [outputs[name] for name in names]
