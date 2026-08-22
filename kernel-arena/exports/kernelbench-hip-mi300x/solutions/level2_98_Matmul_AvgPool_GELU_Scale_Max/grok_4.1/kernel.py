import os
os.environ["CXX"] = "hipcc"
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

fused_post_cpp = """
#include <hip/hip_runtime.h>
#include <cmath>

__constant__ float SQRT_1_OVER_2 = 0.70710678118654752440f;
__constant__ float NEG_INF = -3.402823466e+38f;

__device__ float atomicMaxf(float* address, float val) {
   int* address_as_int = (int*)address;
   int old = *address_as_int, assumed;
   do {
      assumed = old;
      old = atomicCAS(address_as_int, assumed,
                      __float_as_int( fmaxf( val, __int_as_float(assumed) ) ) );
   } while (assumed != old);
   return __int_as_float(old);
}

__global__ void fused_postprocess_kernel(const float *input, float *output, const float scale, const int pool_k, const int out_features, const int batch_size, const int groups_per_block) {
    const int F = out_features;
    const int num_groups = F / pool_k;
    const int num_blocks_per_batch = (num_groups + groups_per_block - 1) / groups_per_block;
    const int block_b = blockIdx.x / num_blocks_per_batch;
    const int block_gstart = (blockIdx.x % num_blocks_per_batch) * groups_per_block;
    if (block_b >= batch_size) return;
    extern __shared__ float shmem[];
    float *vals = shmem;
    float *shared_max = shmem + (pool_k * groups_per_block);
    const int threads_per_group = pool_k;
    const int local_group_tid = threadIdx.x / threads_per_group;
    const int global_group_tid = block_gstart + local_group_tid;
    if (global_group_tid >= num_groups) return;
    const int lane = threadIdx.x % threads_per_group;
    const int base_idx = block_b * F + global_group_tid * pool_k + lane;
    const float val = __ldg(input + base_idx);
    vals[ local_group_tid * threads_per_group + lane ] = val;
    __syncthreads();
    if (lane == 0) {
        for (int lg = 0; lg < groups_per_block; lg++) {
            shared_max[lg] = NEG_INF;
        }
        for (int lg = 0; lg < groups_per_block; lg++) {
            const int g_global = block_gstart + lg;
            if (g_global >= num_groups) break;
            float sum_val = 0.0f;
            #pragma unroll
            for (int l = 0; l < pool_k; l++) {
                sum_val += vals[ lg * pool_k + l ];
            }
            const float avg = sum_val / static_cast<float>(pool_k);
            const float cdf = 0.5f * (1.0f + erf(avg * SQRT_1_OVER_2));
            const float gelu = avg * cdf;
            const float scaled_val = gelu * scale;
            shared_max[lg] = scaled_val;
        }
    }
    __syncthreads();
    // Block-wide reduce max on shared_max
    if (threadIdx.x < groups_per_block) {
        const int gt = threadIdx.x;
        for (int s = groups_per_block / 2; s > 0; s >>= 1) {
            if (gt < s) {
                shared_max[gt] = fmaxf(shared_max[gt], shared_max[gt + s]);
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) {
        atomicMaxf(output + block_b, shared_max[0]);
    }
}

torch::Tensor fused_postprocess_hip(torch::Tensor input, float scale_factor, int pool_kernel_size) {
    const auto batch_sz = input.size(0);
    const auto feat_sz = input.size(1);
    const int pool_k = pool_kernel_size;
    const int n_groups = feat_sz / pool_k;
    const int groups_per_block = 64;
    const int block_size = 1024;
    const int num_blocks_per_batch = (n_groups + groups_per_block - 1) / groups_per_block;
    dim3 threads(block_size);
    dim3 blocks(batch_sz * num_blocks_per_batch);
    size_t shmem_bytes = (pool_k * groups_per_block + groups_per_block) * sizeof(float);
    auto output = torch::full({batch_sz}, NEG_INF, input.options());
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(fused_postprocess_kernel),
        blocks,
        threads,
        shmem_bytes,
        0,
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scale_factor,
        pool_k,
        feat_sz,
        batch_sz,
        groups_per_block
    );
    return output;
}
"""

fused_module = load_inline(
    name="fused_post",
    cpp_sources=fused_post_cpp,
    functions=["fused_postprocess_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.fused_post = fused_module

    def forward(self, x):
        x = self.matmul(x)
        x = self.fused_post.fused_postprocess_hip(x, self.scale_factor, self.pool_kernel_size)
        return x

def get_inputs():
    return [torch.rand(1024, 8192)]

def get_init_inputs():
    return [8192, 8192, 16, 2.0]
