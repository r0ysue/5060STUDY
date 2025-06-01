import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule
import time

# 定义CUDA核函数（直接操作复数实部和虚部，避免复数类型）
mandelbrot_kernel = """
__global__ void mandelbrot(float *real_vals, float *imag_vals, 
                           float *output, int width, int height, 
                           int max_iters, float upper_bound) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float c_re = real_vals[x];
    float c_im = imag_vals[y];
    float z_re = 0.0f, z_im = 0.0f;
    int i;
    float upper_bound_sq = upper_bound * upper_bound;
    
    for (i = 0; i < max_iters; i++) {
        float z_re_sq = z_re * z_re;
        float z_im_sq = z_im * z_im;
        
        // 逃逸条件判断（避免平方根运算）
        if (z_re_sq + z_im_sq > upper_bound_sq) {
            output[y * width + x] = 0.0f;
            break;
        }
        
        // 复数运算：z = z^2 + c
        float new_z_re = z_re_sq - z_im_sq + c_re;
        float new_z_im = 2 * z_re * z_im + c_im;
        z_re = new_z_re;
        z_im = new_z_im;
    }
    
    if (i == max_iters) {
        output[y * width + x] = 1.0f;
    }
}
"""

# 编译CUDA核函数
mod = SourceModule(mandelbrot_kernel)
mandelbrot_gpu = mod.get_function("mandelbrot")

def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    # 生成实部和虚部网格（CPU）
    real_vals = np.linspace(real_low, real_high, width, dtype=np.float32)
    imag_vals = np.linspace(imag_low, imag_high, height, dtype=np.float32)
    
    # 将数据传输到GPU
    real_vals_gpu = gpuarray.to_gpu(real_vals)
    imag_vals_gpu = gpuarray.to_gpu(imag_vals)
    output_gpu = gpuarray.empty((height, width), dtype=np.float32)
    
    # 定义线程块和网格大小
    block_size = (16, 16, 1)  # 每个线程块16x16线程
    grid_size = (
        (width + block_size[0] - 1) // block_size[0],
        (height + block_size[1] - 1) // block_size[1],
        1
    )
    
    # 启动GPU核函数
    mandelbrot_gpu(
        real_vals_gpu, imag_vals_gpu, output_gpu, 
        np.int32(width), np.int32(height), 
        np.int32(max_iters), np.float32(upper_bound),
        block=block_size, grid=grid_size
    )
    
    # 将结果回传到CPU
    return output_gpu.get()

# 测试性能
if __name__ == "__main__":
    width, height = 1024, 1024
    max_iters = 100
    upper_bound = 2.0
    
    start_time = time.time()
    mandelbrot_graph = gpu_mandelbrot(
        width, height, -2.0, 1.0, -1.5, 1.5, max_iters, upper_bound
    )
    gpu_time = time.time() - start_time
    print(f"GPU计算耗时: {gpu_time:.4f}秒")