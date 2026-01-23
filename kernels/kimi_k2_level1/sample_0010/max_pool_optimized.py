import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Set HIP compiler
os.environ["CXX"] = "hipcc"

# Custom HIP kernel for Max Pooling 2D
max_pool_hip_source = """
#include <hip/hip_runtime.h>
#include <float.h>

#define BLOCK_SIZE_X 32  // threads per block in x dimension
#define BLOCK_SIZE_Y 8   // threads per block in y dimension
#define PIXELS_PER_THREAD_X 4  // each thread processes 4 output pixels in x
#define PIXELS_PER_THREAD_Y 1  // each thread processes 1 output pixel in y

__global__ void max_pool_2d_hip_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int KH, int KW, int SH, int SW,
    int PH, int PW, int DH, int DW,
    int OH, int OW
) {
    // Input shape: [N, C, H, W]
    // Output shape: [N, C, OH, OW]
    // Optimized: each thread handles 4 output pixels in a row for better memory coalescing
    
    int n = blockIdx.z;  // batch index
    int c = blockIdx.y;  // channel index
    
    // Each block handles a region of output: BLOCK_SIZE_Y rows x (BLOCK_SIZE_X * PIXELS_PER_THREAD_X) cols
    int output_start_y = blockIdx.x * BLOCK_SIZE_Y;
    int output_start_x = threadIdx.x * PIXELS_PER_THREAD_X;
    
    int thread_y = threadIdx.y;
    int thread_x = threadIdx.x;
    
    // Calculate output positions
    int oh = output_start_y + thread_y;
    
    if (n >= N || c >= C || oh >= OH) return;
    
    // Calculate starting position in input for this output row
    int ih_start = oh * SH - PH;
    
    // Prefetch values to improve memory access pattern
    float max_vals[PIXELS_PER_THREAD_X];
    #pragma unroll
    for (int i = 0; i < PIXELS_PER_THREAD_X; i++) {
        max_vals[i] = -FLT_MAX;
    }
    
    // Loop through kernel elements
    for (int kh = 0; kh < KH; kh++) {
        int ih = ih_start + kh * DH;
        if (ih < 0 || ih >= H) continue;
        
        for (int kw = 0; kw < KW; kw++) {
            int iw_base = (thread_x * PIXELS_PER_THREAD_X * SW) - PW + kw * DW;
            
            #pragma unroll
            for (int i = 0; i < PIXELS_PER_THREAD_X; i++) {
                int ow = output_start_x + i;
                if (ow >= OW) continue;
                
                int iw = iw_base + i * SW;
                if (iw < 0 || iw >= W) continue;
                
                float val = input[((n * C + c) * H + ih) * W + iw];
                max_vals[i] = fmaxf(max_vals[i], val);
            }
        }
    }
    
    // Write output with coalesced memory access
    #pragma unroll
    for (int i = 0; i < PIXELS_PER_THREAD_X; i++) {
        int ow = output_start_x + i;
        if (ow < OW) {
            output[((n * C + c) * OH + oh) * OW + ow] = max_vals[i];
        }
    }
}

torch::Tensor max_pool_2d_hip(torch::Tensor input, 
                              int kernel_h, int kernel_w,
                              int stride_h, int stride_w, 
                              int pad_h, int pad_w,
                              int dilation_h, int dilation_w) {
    
    auto input_cuda = input.cuda();
    auto N = input_cuda.size(0);   // batch
    auto C = input_cuda.size(1);   // channels
    auto H = input_cuda.size(2);   // height
    auto W = input_cuda.size(3);   // width
    
    // Calculate output dimensions
    int OH = (H + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int OW = (W + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    
    auto output = torch::zeros({N, C, OH, OW}, input_cuda.options());
    
    if (output.numel() == 0) {
        return output;
    }
    
    // Grid dimensions
    // x: blocks along height dimension
    // y: channels
    // z: batch
    dim3 block_dim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 grid_dim(
        (OH + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y,  // height blocks
        C,                                       // channels
        N                                        // batch
    );
    
    hipLaunchKernelGGL(
        max_pool_2d_hip_kernel,
        grid_dim,
        block_dim,
        0, 0,
        input_cuda.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernel_h, kernel_w, stride_h, stride_w,
        pad_h, pad_w, dilation_h, dilation_w,
        OH, OW
    );
    
    return output;
}
"""

# Compile the HIP kernel
max_pool_hip = load_inline(
    name="max_pool_hip",
    cpp_sources=max_pool_hip_source,
    functions=["max_pool_2d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.max_pool_hip = max_pool_hip

    def forward(self, x):
        return self.max_pool_hip.max_pool_2d_hip(
            x,
            self.kernel_size[0], self.kernel_size[1],
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1]
        )

# Input parameters
batch_size = 32
channels = 64
height = 512
width = 512
kernel_size = 4
stride = 1
padding = 1
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, channels, height, width).cuda()
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
