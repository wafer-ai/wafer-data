import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

layernorm_cpp = """
#include <hip/hip_runtime.h>

__global__ void layernorm_precompute_kernel(const float *x, float *sum, float *sumsq, int64_t num_prefix, int64_t norm_size, int num_tiles_per_prefix, int tile_size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int prefix_idx = blockIdx.x / num_tiles_per_prefix;
    int tile_idx = blockIdx.x % num_tiles_per_prefix;
    size_t slice_start = static_cast<size_t>(prefix_idx) * static_cast<size_t>(norm_size);
    size_t tile_start = static_cast<size_t>(tile_idx) * static_cast<size_t>(tile_size);
    size_t g_start = slice_start + tile_start;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = tid; i < tile_size; i += blockDim.x) {
        size_t gidx = g_start + i;
        if (gidx >= slice_start + static_cast<size_t>(norm_size)) break;
        float v = x[gidx];
        local_sum += v;
        local_sumsq += v * v;
    }
    int offset = blockDim.x;
    sdata[tid] = local_sum;
    sdata[tid + offset] = local_sumsq;
    __syncthreads();
    for (int half = blockDim.x / 2; half > 0; half >>= 1) {
        if (tid < half) {
            sdata[tid] += sdata[tid + half];
            sdata[tid + offset] += sdata[tid + offset + half];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(sum + prefix_idx, sdata[0]);
        atomicAdd(sumsq + prefix_idx, sdata[offset]);
    }
}

__global__ void finalize_stats_kernel(const float *sum, const float *sumsq, float *mean, float *invstd, int64_t num_prefix, float inv_volume, float eps) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (idx >= num_prefix) return;
    float s = sum[idx] * inv_volume;
    mean[idx] = s;
    float q = sumsq[idx] * inv_volume;
    float var = fmaxf(q - s * s, 0.0f) + eps;
    invstd[idx] = rsqrtf(var);
}

__global__ void layernorm_norm_kernel(const float *x, const float *mean, const float *invstd, const float *gamma, const float *beta, float *out, int64_t num_prefix, int64_t norm_size) {
    int64_t global_idx = static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) + static_cast<int64_t>(threadIdx.x);
    if (global_idx >= num_prefix * norm_size) return;
    int64_t prefix_idx = global_idx / norm_size;
    int64_t norm_idx = global_idx % norm_size;
    size_t gidx = static_cast<size_t>(global_idx);
    float val = x[gidx] - mean[prefix_idx];
    val *= invstd[prefix_idx];
    val *= gamma[norm_idx];
    val += beta[norm_idx];
    out[gidx] = val;
}

torch::Tensor layer_norm_hip(torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, double eps_d = 1e-5) {
    TORCH_CHECK(input.scalar_type() == torch::kFloat, "Only FP32 supported");
    TORCH_CHECK(gamma.scalar_type() == torch::kFloat);
    TORCH_CHECK(beta.scalar_type() == torch::kFloat);

    auto options = input.options();
    int64_t total_nelem = input.numel();
    int64_t norm_nelem = gamma.numel();
    TORCH_CHECK(beta.numel() == norm_nelem);
    TORCH_CHECK(total_nelem % norm_nelem == 0);
    int64_t num_prefixes = total_nelem / norm_nelem;

    auto sum_buf = torch::zeros({num_prefixes}, options);
    auto sumsq_buf = torch::zeros({num_prefixes}, options);
    auto mean_buf = torch::zeros({num_prefixes}, options);
    auto invstd_buf = torch::zeros({num_prefixes}, options);
    auto out = torch::empty_like(input);

    const int THREADS = 256;
    const int LOADS_PER_THREAD = 16;
    const int TILE_SIZE = THREADS * LOADS_PER_THREAD;
    int64_t num_tiles = (norm_nelem + TILE_SIZE - 1LL) / TILE_SIZE;
    int64_t total_blocks_ll = num_prefixes * num_tiles;
    TORCH_CHECK(total_blocks_ll <= (1LL<<20), "Too many blocks");
    int total_blocks = static_cast<int>(total_blocks_ll);
    dim3 blocks(total_blocks);
    dim3 threads(THREADS);
    size_t shared_mem_bytes = 2 * THREADS * sizeof(float);

    float *x_ptr = input.data_ptr<float>();
    float *g_ptr = gamma.data_ptr<float>();
    float *b_ptr = beta.data_ptr<float>();
    float *o_ptr = out.data_ptr<float>();
    float *s_ptr = sum_buf.data_ptr<float>();
    float *sq_ptr = sumsq_buf.data_ptr<float>();
    float *m_ptr = mean_buf.data_ptr<float>();
    float *is_ptr = invstd_buf.data_ptr<float>();

    float inv_vol = 1.0f / static_cast<float>(norm_nelem);
    float epsf = static_cast<float>(eps_d);

    int num_tiles_per_prefix = static_cast<int>(num_tiles);
    int tile_s = TILE_SIZE;

    // Precompute sums
    layernorm_precompute_kernel<<<blocks, threads, shared_mem_bytes>>>(x_ptr, s_ptr, sq_ptr, num_prefixes, norm_nelem, num_tiles_per_prefix, tile_s);
    (void) hipDeviceSynchronize();

    // Finalize stats
    int64_t stats_grid_size = (num_prefixes + THREADS - 1) / THREADS;
    dim3 sblocks(stats_grid_size);
    finalize_stats_kernel<<<sblocks, threads>>>(s_ptr, sq_ptr, m_ptr, is_ptr, num_prefixes, inv_vol, epsf);
    (void) hipDeviceSynchronize();

    // Normalize
    const int norm_block_size = 256;
    int64_t norm_num_blocks = (total_nelem + norm_block_size - 1) / norm_block_size;
    dim3 norm_blocks(norm_num_blocks);
    layernorm_norm_kernel<<<norm_blocks, norm_block_size>>>(x_ptr, m_ptr, is_ptr, g_ptr, b_ptr, o_ptr, num_prefixes, norm_nelem);

    return out;
}
"""

layernorm = load_inline(
    name="layernorm",
    cpp_sources=layernorm_cpp,
    functions=["layer_norm_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, normalized_shape: tuple) -> None:
        super().__init__()
        self.ln_weight = nn.Parameter(torch.ones(*normalized_shape))
        self.ln_bias = nn.Parameter(torch.zeros(*normalized_shape))
        self.eps = 1e-5
        self.custom_layer_norm = layernorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.custom_layer_norm.layer_norm_hip(x, self.ln_weight, self.ln_bias, self.eps)
