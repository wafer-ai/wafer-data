import os
import torch
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

outer_product_cpp_source = """
#include <hip/hip_runtime.h>

__global__ void outer_product_kernel(
    const float* error,
    const float* k,
    float* output,
    int batch_size,
    int num_heads,
    int head_dim_v,
    int head_dim_qk)
{
    int batch = blockIdx.x;
    int head = blockIdx.y;
    
    if (batch >= batch_size || head >= num_heads) return;
    
    int idx = threadIdx.x;
    int total_v_threads = blockDim.x;
    
    for (int v_idx = idx; v_idx < head_dim_v; v_idx += total_v_threads) {
        float e_val = error[batch * num_heads * head_dim_v + head * head_dim_v + v_idx];
        
        float* out_row = output + batch * num_heads * head_dim_v * head_dim_qk + 
                         head * head_dim_v * head_dim_qk + 
                         v_idx * head_dim_qk;
        
        const float* k_ptr = k + batch * num_heads * head_dim_qk + head * head_dim_qk;
        
        for (int j = 0; j < head_dim_qk; j++) {
            out_row[j] = e_val * k_ptr[j];
        }
    }
}

torch::Tensor outer_product_hip(torch::Tensor error, torch::Tensor k) {
    auto batch_size = error.size(0);
    auto num_heads = error.size(1);
    auto head_dim_v = error.size(2);
    auto head_dim_qk = k.size(2);
    
    auto output = torch::empty({batch_size, num_heads, head_dim_v, head_dim_qk}, error.options());
    
    dim3 grid(batch_size, num_heads);
    int threads = 256;
    
    hipLaunchKernelGGL(
        outer_product_kernel,
        grid, threads, 0, 0,
        error.data_ptr<float>(),
        k.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_heads,
        head_dim_v,
        head_dim_qk
    );
    
    return output;
}
"""

outer_product_module = load_inline(
    name="outer_product_module",
    cpp_sources=outer_product_cpp_source,
    functions=["outer_product_hip"],
    verbose=True,
)

# Test the kernel
batch_size = 2
num_heads = 3
head_dim_v = 4
head_dim_qk = 5

error = torch.randn(batch_size, num_heads, head_dim_v).cuda()
k = torch.randn(batch_size, num_heads, head_dim_qk).cuda()

# Reference einsum
ref = torch.einsum('bhi,bhj->bhij', error, k)

# Kernel result
result = outer_product_module.outer_product_hip(error, k)

print("Kernel result shape:", result.shape)
print("Max diff:", (ref - result).abs().max())
print("Mean diff:", (ref - result).abs().mean())
print("Are close:", torch.allclose(ref, result, rtol=1e-4, atol=1e-4))

# Print sample values
print("\nSample values:")
for b in range(min(1, batch_size)):
    for h in range(min(1, num_heads)):
        for i in range(min(2, head_dim_v)):
            for j in range(min(2, head_dim_qk)):
                print(f"  [{b},{h},{i},{j}] ref={ref[b,h,i,j]:.6f} result={result[b,h,i,j]:.6f}")