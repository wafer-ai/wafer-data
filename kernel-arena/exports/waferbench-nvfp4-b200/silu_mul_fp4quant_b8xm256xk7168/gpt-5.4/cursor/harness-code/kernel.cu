#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ uint32_t cvt_8xfp32_to_e2m1(
    float v0, float v1, float v2, float v3,
    float v4, float v5, float v6, float v7
) {
    uint32_t result;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}"
        : "=r"(result)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3),
          "f"(v4), "f"(v5), "f"(v6), "f"(v7)
    );
    return result;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ int64_t swizzled_scale_offset(
    int expert_idx, int row_idx_in_expert, int scale_idx, int padded_m, int padded_scale_k
) {
    const int m_tile_idx = row_idx_in_expert >> 7;
    const int k_tile_idx = scale_idx >> 2;
    const int outer_m_idx = row_idx_in_expert & 31;
    const int inner_m_idx = (row_idx_in_expert >> 5) & 3;
    const int inner_k_idx = scale_idx & 3;
    const int num_k_tiles = padded_scale_k >> 2;
    return static_cast<int64_t>(expert_idx) * padded_m * padded_scale_k
         + static_cast<int64_t>(m_tile_idx) * num_k_tiles * 32 * 4 * 4
         + static_cast<int64_t>(k_tile_idx) * 32 * 4 * 4
         + static_cast<int64_t>(outer_m_idx) * 4 * 4
         + static_cast<int64_t>(inner_m_idx) * 4
         + inner_k_idx;
}

template <int THREADS>
__global__ __launch_bounds__(THREADS) void silu_mul_fp4quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ mask,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int m_per_expert, int K, int n_experts, int padded_m, int padded_scale_k
) {
    constexpr int BLOCK_SIZE = 16;
    constexpr float FP4_MAX_RCP = 1.0f / 6.0f;

    const int row_idx = blockIdx.x;
    const int expert_idx = row_idx / m_per_expert;
    const int row_idx_in_expert = row_idx - expert_idx * m_per_expert;

    if (expert_idx >= n_experts || row_idx_in_expert >= mask[expert_idx]) {
        return;
    }

    const int scale_k = K / BLOCK_SIZE;
    const int K2 = K * 2;
    const __nv_bfloat16* row_ptr = input + static_cast<int64_t>(row_idx) * K2;
    const __nv_bfloat16* gate_ptr = row_ptr;
    const __nv_bfloat16* up_ptr = row_ptr + K;
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output + static_cast<int64_t>(row_idx) * (K / 2));
    const float global_scale = global_scales == nullptr ? 1.0f : global_scales[expert_idx];
    const float global_scale_rcp = rcp_approx_ftz(global_scale);

    for (int scale_idx = threadIdx.x; scale_idx < scale_k; scale_idx += THREADS) {
        const int col = scale_idx * BLOCK_SIZE;
        const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate_ptr + col);
        const __nv_bfloat162* up2 = reinterpret_cast<const __nv_bfloat162*>(up_ptr + col);

        float vals[16];
        float amax = 0.0f;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 gate_f = __bfloat1622float2(gate2[i]);
            const float2 up_f = __bfloat1622float2(up2[i]);
            const __nv_bfloat162 fused_bf16 = __float22bfloat162_rn(
                make_float2(silu(gate_f.x) * up_f.x, silu(gate_f.y) * up_f.y));
            const float2 fused_f = __bfloat1622float2(fused_bf16);
            vals[i * 2] = fused_f.x;
            vals[i * 2 + 1] = fused_f.y;
            amax = fmaxf(amax, fabsf(fused_f.x));
            amax = fmaxf(amax, fabsf(fused_f.y));
        }

        const float sf_value = global_scale * (amax * FP4_MAX_RCP);
        const __nv_fp8_e4m3 sf_narrow_fp8 = __nv_fp8_e4m3(sf_value);
        const float sf_narrow = static_cast<float>(sf_narrow_fp8);
        const float output_scale = amax != 0.0f
            ? rcp_approx_ftz(sf_narrow * global_scale_rcp)
            : 0.0f;

        block_scales[swizzled_scale_offset(
            expert_idx, row_idx_in_expert, scale_idx, padded_m, padded_scale_k)] = sf_narrow_fp8.__x;

        const int out_word_idx = scale_idx * 2;
        out_u32[out_word_idx] = cvt_8xfp32_to_e2m1(
            vals[0] * output_scale, vals[1] * output_scale,
            vals[2] * output_scale, vals[3] * output_scale,
            vals[4] * output_scale, vals[5] * output_scale,
            vals[6] * output_scale, vals[7] * output_scale);
        out_u32[out_word_idx + 1] = cvt_8xfp32_to_e2m1(
            vals[8] * output_scale, vals[9] * output_scale,
            vals[10] * output_scale, vals[11] * output_scale,
            vals[12] * output_scale, vals[13] * output_scale,
            vals[14] * output_scale, vals[15] * output_scale);
    }
}

std::vector<torch::Tensor> silu_and_mul_scaled_nvfp4_experts_quantize(
    torch::Tensor input,
    torch::Tensor mask,
    std::optional<torch::Tensor> global_scale
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 3);
    int B = input.size(0), M = input.size(1), K2 = input.size(2);
    TORCH_CHECK(K2 % 2 == 0);
    int K = K2 / 2;
    constexpr int BLOCK_SIZE = 16;
    TORCH_CHECK(K % BLOCK_SIZE == 0);
    TORCH_CHECK(mask.is_cuda() && mask.dtype() == torch::kInt32);
    TORCH_CHECK(mask.dim() == 1 && mask.size(0) == B);
    if (global_scale.has_value()) {
        TORCH_CHECK(global_scale->is_cuda() && global_scale->dtype() == torch::kFloat32);
        TORCH_CHECK(global_scale->dim() == 1 && global_scale->size(0) == B);
    }
    int scale_k = K / BLOCK_SIZE;
    int m_topk = B * M;

    float* gs_ptr = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto fp4_out = torch::empty({B, M, K / 2}, input.options().dtype(torch::kUInt8));
    int padded_m = ((M + 127) / 128) * 128;
    int padded_scale_k = ((scale_k + 3) / 4) * 4;
    auto swizzled_scales = torch::zeros(
        {B, padded_m, padded_scale_k},
        input.options().dtype(torch::kUInt8));

    constexpr int THREADS = 128;
    silu_mul_fp4quant_kernel<THREADS><<<m_topk, THREADS>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.view({m_topk, K2}).data_ptr<at::BFloat16>()),
        gs_ptr, mask.data_ptr<int32_t>(),
        fp4_out.view({m_topk, K / 2}).data_ptr<uint8_t>(),
        swizzled_scales.data_ptr<uint8_t>(),
        M, K, B, padded_m, padded_scale_k);

    fp4_out = fp4_out.permute({1, 2, 0});

    auto output_scales = swizzled_scales
        .view({B * padded_m, padded_scale_k})
        .view(c10::ScalarType::Float8_e4m3fn)
        .view({B, padded_m / 128, padded_scale_k / 4, 32, 4, 4})
        .permute({3, 4, 1, 5, 2, 0});

    return {fp4_out, output_scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul_scaled_nvfp4_experts_quantize",
          &silu_and_mul_scaled_nvfp4_experts_quantize,
          py::arg("input"), py::arg("mask"),
          py::arg("global_scale") = py::none());
}
