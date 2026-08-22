import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

os.environ["CXX"] = "hipcc"

cpp_source = """
#include <hip/hip_runtime.h>
#include <torch/extension.h>

struct WelfordData {
    float mean;
    float m2;
    int n;
};

__device__ inline WelfordData welford_combine(WelfordData a, WelfordData b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    
    WelfordData res;
    res.n = a.n + b.n;
    float delta = b.mean - a.mean;
    
    float fn = (float)res.n;
    float fnb = (float)b.n;
    float fna = (float)a.n;
    
    res.mean = a.mean + delta * fnb / fn;
    res.m2 = a.m2 + b.m2 + delta * delta * fna * fnb / fn;
    return res;
}

__global__ void fused_instance_norm_divide_kernel_welford(
    float* __restrict__ data,
    int HW,
    float divide_by,
    float epsilon) 
{
    extern __shared__ char smem[];
    WelfordData* sdata = (WelfordData*)smem;
    
    int tid = threadIdx.x;
    int nc = blockIdx.x; 
    
    size_t offset = (size_t)nc * HW;
    float* img = data + offset;

    // --- Pass 1: Local Welford ---
    WelfordData local_data;
    local_data.mean = 0.0f;
    local_data.m2 = 0.0f;
    local_data.n = 0;
    
    for (int i = tid; i < HW; i += blockDim.x) {
        float val = img[i];
        local_data.n++;
        float delta = val - local_data.mean;
        local_data.mean += delta / local_data.n;
        float delta2 = val - local_data.mean;
        local_data.m2 += delta * delta2;
    }
    
    sdata[tid] = local_data;
    __syncthreads();
    
    // --- Reduction ---
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = welford_combine(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
    
    // --- Pass 2: Normalize & Divide ---
    __shared__ float sh_mean;
    __shared__ float sh_scale;
    
    if (tid == 0) {
        float mean = sdata[0].mean;
        float var = sdata[0].m2 / HW; 
        sh_mean = mean;
        sh_scale = rsqrtf(var + epsilon) / divide_by;
    }
    __syncthreads();
    
    float mean = sh_mean;
    float scale = sh_scale;
    
    for (int i = tid; i < HW; i += blockDim.x) {
        img[i] = (img[i] - mean) * scale;
    }
}

torch::Tensor fused_instance_norm_divide_hip(torch::Tensor input, float divide_by) {
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    
    int num_instances = N * C;
    int block_size = 256;
    size_t shared_mem_size = block_size * sizeof(WelfordData);
    
    fused_instance_norm_divide_kernel_welford<<<num_instances, block_size, shared_mem_size>>>(
        input.data_ptr<float>(),
        HW,
        divide_by,
        1e-5f
    );
    
    return input;
}
"""

fused_op = load_inline(
    name="fused_instance_norm_divide_v4",
    cpp_sources=cpp_source,
    functions=["fused_instance_norm_divide_hip"],
    verbose=True,
    extra_cflags=['-O3']
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = float(divide_by)
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        x = self.conv(x)
        x = fused_op.fused_instance_norm_divide_hip(x, self.divide_by)
        return x

batch_size = 128
in_channels  = 64  
out_channels = 128  
height = width = 128  
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
