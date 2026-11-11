import ctypes as c

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
    def __init__(self, plan, shape):
        trt.init_libnvinfer_plugins(LOGGER, "")
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Cannot load {plan.name}")
        self.context = self.engine.create_execution_context()
        self.stream = c.c_void_p()
        check(CUDA.cudaStreamCreate(c.byref(self.stream)))
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
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
        names = ("1062", "1035") if "1062" in outputs else ("simcc_x", "simcc_y")
        return [outputs[name] for name in names]
