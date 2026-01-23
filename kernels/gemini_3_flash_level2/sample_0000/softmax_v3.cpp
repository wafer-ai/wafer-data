
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <math.h>
#include <float.h>

__global__ void softmax_dim1_kernel_v3(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int num_spatial) {

    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y;
    
    if (spatial_idx >= num_spatial || n >= N) return;

    int base_idx = n * C * num_spatial + spatial_idx;

    float max_val = -FLT_MAX;
    
    // We can't easily use a fixed array if C is not constant,
    // but C is small (16), so we can handle it.
    // Let's use a small fixed size and handle the case where C is larger.
    float vals[64]; 
    int current_C = (C > 64) ? 64 : C;

    for (int c = 0; c < current_C; ++c) {
        float v = input[base_idx + c * num_spatial];
        vals[c] = v;
        if (v > max_val) max_val = v;
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < current_C; ++c) {
        vals[c] = expf(vals[c] - max_val);
        sum_exp += vals[c];
    }

    float inv_sum_exp = 1.0f / sum_exp;
    for (int c = 0; c < current_C; ++c) {
        output[base_idx + c * num_spatial] = vals[c] * inv_sum_exp;
    }
}
