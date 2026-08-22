import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Unique module name to avoid cache issues
matmul_swish_scale_cpp_source = """
#include <hip/hip_runtime.h>

__device__ __forceinline__ float sigmoid_hip(float x) {
    return 1.0f / (1.0f + expf(-fmaxf(-50.0f, fminf(50.0f, x))));
}

__global__ void matmul_swish_scale_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int M,
    int N,
    int K,
    float scaling_factor) {
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; k++) {
            sum += input[row * K + k] * weight[col * K + k];
        }
        
        float swish_val = sum * sigmoid_hip(sum);
        
        output[row * N + col] = swish_val * scaling_factor;
    }
}

torch::Tensor matmul_swish_scale_hip(
    torch::Tensor input,
    torch::Tensor weight,
    float scaling_factor) {
    
    int M = input.size(0);
    int K = input.size(1);
    int N = weight.size(0);
    
    auto output = torch::zeros({M, N}, input.options());
    
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    
    matmul_swish_scale_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        M, N, K,
        scaling_factor);
    
    return output;
}
"""

matmul_swish_scale = load_inline(
    name="matmul_swish_scale_fused_v2",
    cpp_sources=matmul_swish_scale_cpp_source,
    functions=["matmul_swish_scale_hip"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling_factor = scaling_factor
        self.matmul_swish_scale = matmul_swish_scale
        
        self.register_parameter(
            'weight',
            nn.Parameter(torch.Tensor(out_features, in_features))
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        x = x.float().cuda()
        weight = self.weight.float().cuda()
        out = self.matmul_swish_scale.matmul_swish_scale_hip(x, weight, self.scaling_factor)
        return out