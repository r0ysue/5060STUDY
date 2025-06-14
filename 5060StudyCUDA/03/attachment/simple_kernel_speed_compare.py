import time
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
from pycuda.elementwise import ElementwiseKernel

# 查询设备最大线程数
from pycuda.driver import Device
max_threads = Device(0).max_threads_per_block
print(f"Max threads per block: {max_threads}")

# 设置合理的block尺寸（通常256或512是安全值）
block_size = 256

# 生成数据
N = 10000000
a = np.random.randn(N).astype(np.float32)
b = np.random.randn(N).astype(np.float32)
a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)
c_gpu = gpuarray.empty_like(a_gpu)

mod = SourceModule("""
    __global__ void add(float *a, float *b, float *c) {
        int idx = threadIdx.x;
        c[idx] = a[idx] + b[idx];
    }
""")
add_func = mod.get_function("add")

add_kernel = ElementwiseKernel(
    "float *a, float *b, float *c",
    "c[i] = a[i] + b[i]",
    "add_kernel"
)

# 测试 SourceModule
grid_size = (N + block_size - 1) // block_size
start = time.time()
add_func(a_gpu, b_gpu, c_gpu, block=(block_size, 1, 1), grid=(grid_size, 1, 1))
print("SourceModule Time:", time.time() - start)

# 测试 ElementwiseKernel
start = time.time()
add_kernel(a_gpu, b_gpu, c_gpu)
print("ElementwiseKernel Time:", time.time() - start)