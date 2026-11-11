# Source from the application directory.
cosmos_cuda="$PWD/runtime/cosmos-cuda"
cuda_target="$cosmos_cuda/usr/local/cuda-12.6/targets/aarch64-linux"
export LD_LIBRARY_PATH="$cosmos_cuda/cudss/lib:$cuda_target/lib:$cosmos_cuda/usr/local/cuda-12.6/extras/CUPTI/lib64:$cosmos_cuda/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
