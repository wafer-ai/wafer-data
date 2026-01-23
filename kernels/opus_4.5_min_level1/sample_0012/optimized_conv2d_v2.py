import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Use implicit GEMM approach with shared memory for better performance
conv2d_hip_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

#define BLOCK_M 64
#define BLOCK_N 64
#define BLOCK_K 16
#define THREAD_M 4
#define THREAD_N 4

// Optimized conv2d using im2col + matmul approach with shared memory
__global__ void conv2d_implicit_gemm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int batch_size,
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int out_height,
    int out_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation
) {
    // Each thread computes multiple output elements
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    int out_spatial = out_height * out_width;
    int K = in_channels * kernel_size * kernel_size;  // reduction dimension
    
    // Block position
    int block_m = blockIdx.x * BLOCK_M;  // output channel block
    int block_n = blockIdx.y * BLOCK_N;  // output spatial block
    int batch = blockIdx.z;
    
    __shared__ float As[BLOCK_K][BLOCK_M + 1];  // weights: K x out_channels
    __shared__ float Bs[BLOCK_K][BLOCK_N + 1];  // input patch: K x spatial
    
    // Accumulators
    float acc[THREAD_M][THREAD_N] = {0.0f};
    
    int threads_per_block = blockDim.x * blockDim.y;
    int tid = tid_y * blockDim.x + tid_x;
    
    // Loop over K dimension in chunks
    for (int k_block = 0; k_block < K; k_block += BLOCK_K) {
        // Load weight tile into shared memory
        // Weight layout: [out_channels, in_channels, kH, kW]
        for (int i = tid; i < BLOCK_K * BLOCK_M; i += threads_per_block) {
            int k_idx = i / BLOCK_M;
            int m_idx = i % BLOCK_M;
            int global_k = k_block + k_idx;
            int global_m = block_m + m_idx;
            
            if (global_k < K && global_m < out_channels) {
                // weight is [out_channels, in_channels, kH, kW]
                As[k_idx][m_idx] = weight[global_m * K + global_k];
            } else {
                As[k_idx][m_idx] = 0.0f;
            }
        }
        
        // Load input patch into shared memory using im2col
        for (int i = tid; i < BLOCK_K * BLOCK_N; i += threads_per_block) {
            int k_idx = i / BLOCK_N;
            int n_idx = i % BLOCK_N;
            int global_k = k_block + k_idx;
            int global_n = block_n + n_idx;
            
            if (global_k < K && global_n < out_spatial) {
                // Decode k into (in_c, ky, kx)
                int in_c = global_k / (kernel_size * kernel_size);
                int k_rem = global_k % (kernel_size * kernel_size);
                int ky = k_rem / kernel_size;
                int kx = k_rem % kernel_size;
                
                // Decode n into (out_y, out_x)
                int out_y = global_n / out_width;
                int out_x = global_n % out_width;
                
                // Calculate input position
                int in_y = out_y * stride - padding + ky * dilation;
                int in_x = out_x * stride - padding + kx * dilation;
                
                if (in_y >= 0 && in_y < in_height && in_x >= 0 && in_x < in_width) {
                    int input_idx = ((batch * in_channels + in_c) * in_height + in_y) * in_width + in_x;
                    Bs[k_idx][n_idx] = input[input_idx];
                } else {
                    Bs[k_idx][n_idx] = 0.0f;
                }
            } else {
                Bs[k_idx][n_idx] = 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial results
        #pragma unroll
        for (int k = 0; k < BLOCK_K; ++k) {
            // Each thread handles THREAD_M x THREAD_N output elements
            #pragma unroll
            for (int tm = 0; tm < THREAD_M; ++tm) {
                float a_val = As[k][tid_y * THREAD_M + tm];
                #pragma unroll
                for (int tn = 0; tn < THREAD_N; ++tn) {
                    float b_val = Bs[k][tid_x * THREAD_N + tn];
                    acc[tm][tn] += a_val * b_val;
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results to output
    for (int tm = 0; tm < THREAD_M; ++tm) {
        int out_c = block_m + tid_y * THREAD_M + tm;
        if (out_c < out_channels) {
            for (int tn = 0; tn < THREAD_N; ++tn) {
                int spatial_idx = block_n + tid_x * THREAD_N + tn;
                if (spatial_idx < out_spatial) {
                    int out_y = spatial_idx / out_width;
                    int out_x = spatial_idx % out_width;
                    int output_idx = ((batch * out_channels + out_c) * out_height + out_y) * out_width + out_x;
                    output[output_idx] = acc[tm][tn];
                }
            }
        }
    }
}

torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
) {
    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int in_height = input.size(2);
    int in_width = input.size(3);
    
    int out_channels = weight.size(0);
    int kernel_size = weight.size(2);
    
    int out_height = (in_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int out_width = (in_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width}, input.options());
    
    int out_spatial = out_height * out_width;
    
    // Each thread computes THREAD_M x THREAD_N elements
    dim3 block(BLOCK_N / THREAD_N, BLOCK_M / THREAD_M);  // 16x16 = 256 threads
    dim3 grid(
        (out_channels + BLOCK_M - 1) / BLOCK_M,
        (out_spatial + BLOCK_N - 1) / BLOCK_N,
        batch_size
    );
    
    conv2d_implicit_gemm_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        in_height,
        in_width,
        out_height,
        out_width,
        kernel_size,
        stride,
        padding,
        dilation
    );
    
    return output;
}
"""

conv2d_cpp_source = """
torch::Tensor conv2d_hip(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int dilation
);
"""

conv2d_module = load_inline(
    name="conv2d_hip_v2",
    cpp_sources=conv2d_cpp_source,
    cuda_sources=conv2d_hip_source,
    functions=["conv2d_hip"],
    verbose=True,
    extra_cuda_cflags=["-O3"]
)


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Initialize weights same as nn.Conv2d
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = conv2d_module.conv2d_hip(
            x.contiguous(),
            self.weight.contiguous(),
            self.stride,
            self.padding,
            self.dilation
        )
        
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        
        return output
