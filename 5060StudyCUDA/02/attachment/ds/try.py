from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.compiler import SourceModule

# CUDA kernel for Mandelbrot set calculation
mandelbrot_kernel = """
__global__ void mandelbrot(float *output, 
                           float real_low, float real_high,
                           float imag_low, float imag_high,
                           int max_iters, float upper_bound,
                           int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float real = real_low + x * (real_high - real_low) / (width - 1);
    float imag = imag_low + y * (imag_high - imag_low) / (height - 1);
    
    float z_real = 0.0f;
    float z_imag = 0.0f;
    float c_real = real;
    float c_imag = imag;
    
    int iter;
    for (iter = 0; iter < max_iters; iter++) {
        float z_real2 = z_real * z_real;
        float z_imag2 = z_imag * z_imag;
        
        if (z_real2 + z_imag2 > upper_bound * upper_bound) {
            break;
        }
        
        // Complex multiplication: (a+bi)^2 = (a^2 - b^2) + 2abi
        float new_z_real = z_real2 - z_imag2 + c_real;
        float new_z_imag = 2 * z_real * z_imag + c_imag;
        
        z_real = new_z_real;
        z_imag = new_z_imag;
    }
    
    output[y * width + x] = (iter == max_iters) ? 1.0f : 0.0f;
}
"""

def gpu_mandelbrot(width, height, real_low, real_high, imag_low, imag_high, max_iters, upper_bound):
    # Compile the CUDA kernel
    mod = SourceModule(mandelbrot_kernel)
    mandelbrot_func = mod.get_function("mandelbrot")
    
    # Allocate GPU memory for the output
    output_gpu = gpuarray.empty((height, width), dtype=np.float32)
    
    # Calculate block and grid dimensions
    block_size = (16, 16, 1)  # 256 threads per block
    grid_x = (width + block_size[0] - 1) // block_size[0]
    grid_y = (height + block_size[1] - 1) // block_size[1]
    grid_size = (grid_x, grid_y, 1)
    
    # Execute the kernel
    mandelbrot_func(output_gpu,
                   np.float32(real_low), np.float32(real_high),
                   np.float32(imag_low), np.float32(imag_high),
                   np.int32(max_iters), np.float32(upper_bound),
                   np.int32(width), np.int32(height),
                   block=block_size, grid=grid_size)
    
    # Copy the result back to host memory
    return output_gpu.get()

if __name__ == '__main__':
    t1 = time()
    mandel = gpu_mandelbrot(512, 512, -2, 2, -2, 2, 256, 2.5)
    t2 = time()
    mandel_time = t2 - t1
    
    t1 = time()
    fig = plt.figure(1)
    plt.imshow(mandel, extent=(-2, 2, -2, 2))
    plt.savefig('mandelbrot.png', dpi=fig.dpi)
    t2 = time()
    
    dump_time = t2 - t1
    
    print('It took {:.4f} seconds to calculate the Mandelbrot graph.'.format(mandel_time))
    print('It took {:.4f} seconds to dump the image.'.format(dump_time))