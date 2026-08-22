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

__global__ void silu_mul_fp4quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ mask,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ swizzled_scales,
    int m_per_expert, int K, int n_experts, int block_size
) {
    const int thread_idx = static_cast<int>(threadIdx.x);
    const int scale_k = K / block_size;
    const int row = static_cast<int>(blockIdx.x);
    const int expert_idx = row / m_per_expert;
    if (expert_idx >= n_experts) {
        return;
    }
    const int row_in_expert = row - expert_idx * m_per_expert;

    const int out_row_stride = K / 2;
    uint64_t* const output_row = reinterpret_cast<uint64_t*>(output + row * out_row_stride);

    if (row_in_expert >= mask[expert_idx]) {
        return;
    }

    const int factor = block_size * 4;
    const int padded_m = ((m_per_expert + 127) / 128) * 128;
    const int num_k_tiles = (K + factor - 1) / factor;
    const int m_tile_idx = row_in_expert / 128;
    const int outer_m_idx = row_in_expert % 32;
    const int inner_m_idx = (row_in_expert % 128) / 32;
    const float global_scale_val = global_scales[expert_idx];

    for (int scale_idx = thread_idx; scale_idx < scale_k; scale_idx += blockDim.x) {
        const int col0 = scale_idx * block_size;
        const __nv_bfloat16* const gate_ptr = input + row * (2 * K) + col0;
        const __nv_bfloat16* const up_ptr = gate_ptr + K;

        const __nv_bfloat162* const gate2_ptr =
            reinterpret_cast<const __nv_bfloat162*>(gate_ptr);
        const __nv_bfloat162* const up2_ptr =
            reinterpret_cast<const __nv_bfloat162*>(up_ptr);

        float vals[16];
        float amax = 0.0f;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float2 g = __bfloat1622float2(gate2_ptr[i]);
            const float2 u = __bfloat1622float2(up2_ptr[i]);

            const float silu0 = g.x / (1.0f + __expf(-g.x));
            const float silu1 = g.y / (1.0f + __expf(-g.y));
            const __nv_bfloat162 rounded =
                __float22bfloat162_rn(make_float2(silu0 * u.x, silu1 * u.y));
            const float2 v = __bfloat1622float2(rounded);
            const float v0 = v.x;
            const float v1 = v.y;

            vals[2 * i] = v0;
            vals[2 * i + 1] = v1;
            amax = fmaxf(amax, fabsf(v0));
            amax = fmaxf(amax, fabsf(v1));
        }

        const float scale_f = global_scale_val * (amax * (1.0f / 6.0f));
        const uint8_t scale_u8 =
            __nv_cvt_float_to_fp8(scale_f, __NV_SATFINITE, __NV_E4M3);

        const int k_tile_idx = scale_idx / 4;
        const int inner_k_idx = scale_idx % 4;
        const int scale_offset =
            expert_idx * (padded_m * num_k_tiles * 4) +
            m_tile_idx * (num_k_tiles * 32 * 4 * 4) +
            k_tile_idx * (32 * 4 * 4) +
            outer_m_idx * (4 * 4) +
            inner_m_idx * 4 +
            inner_k_idx;
        swizzled_scales[scale_offset] = scale_u8;

        float output_scale = 0.0f;
        if (scale_u8 != 0) {
            const __half rounded_scale_h = __nv_cvt_fp8_to_halfraw(scale_u8, __NV_E4M3);
            const float scale_narrow = __half2float(rounded_scale_h);
            output_scale =
                rcp_approx_ftz(scale_narrow * rcp_approx_ftz(global_scale_val));
        }

        const uint32_t pack0 = cvt_8xfp32_to_e2m1(
            vals[0] * output_scale, vals[1] * output_scale,
            vals[2] * output_scale, vals[3] * output_scale,
            vals[4] * output_scale, vals[5] * output_scale,
            vals[6] * output_scale, vals[7] * output_scale
        );
        const uint32_t pack1 = cvt_8xfp32_to_e2m1(
            vals[8] * output_scale, vals[9] * output_scale,
            vals[10] * output_scale, vals[11] * output_scale,
            vals[12] * output_scale, vals[13] * output_scale,
            vals[14] * output_scale, vals[15] * output_scale
        );

        output_row[scale_idx] =
            static_cast<uint64_t>(pack0) | (static_cast<uint64_t>(pack1) << 32);
    }
}

static torch::Tensor swizzle_scales(torch::Tensor linear, int M, int K, int block_size) {
    int scale_k = K / block_size;
    int padded_m = ((M + 127) / 128) * 128;
    int factor = block_size * 4;
    int padded_k_full = ((K + factor - 1) / factor) * factor;
    int padded_scale_k = padded_k_full / block_size;

    auto padded = torch::zeros({padded_m, padded_scale_k}, linear.options());
    padded.slice(0, 0, M).slice(1, 0, scale_k).copy_(linear);

    int rm = padded_m / 128;
    int rk = padded_scale_k / 4;
    auto reshaped = padded.view({rm, 4, 32, rk, 4});
    return reshaped.permute({0, 3, 2, 1, 4}).contiguous().view({padded_m, padded_scale_k});
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
    int scale_k = K / BLOCK_SIZE;
    int m_topk = B * M;

    float* gs_ptr = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto fp4_out = torch::empty({B, M, K / 2}, input.options().dtype(torch::kUInt8));
    int padded_m = ((M + 127) / 128) * 128;
    int factor = BLOCK_SIZE * 4;
    int padded_k_full = ((K + factor - 1) / factor) * factor;
    int padded_scale_k = padded_k_full / BLOCK_SIZE;
    auto swizzled_all = torch::zeros({B, padded_m, padded_scale_k},
                                     input.options().dtype(torch::kUInt8));

    int threads = scale_k < 128 ? scale_k : 128;
    silu_mul_fp4quant_kernel<<<m_topk, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.view({m_topk, K2}).data_ptr<at::BFloat16>()),
        gs_ptr, mask.data_ptr<int32_t>(),
        fp4_out.view({m_topk, K / 2}).data_ptr<uint8_t>(),
        swizzled_all.data_ptr<uint8_t>(),
        M, K, B, BLOCK_SIZE);
    int padded_k_int32 = padded_scale_k / 4;

    fp4_out = fp4_out.permute({1, 2, 0});

    auto output_scales = swizzled_all
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
