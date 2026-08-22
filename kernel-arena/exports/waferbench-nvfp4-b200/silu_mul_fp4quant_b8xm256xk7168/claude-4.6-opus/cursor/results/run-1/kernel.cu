#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
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

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ int swizzle_offset(
    int row_in_expert, int col_block, int expert_idx,
    int padded_m, int padded_scale_k
) {
    const int rk = padded_scale_k >> 2;
    return expert_idx * padded_m * padded_scale_k +
           (((((row_in_expert >> 7) * rk + (col_block >> 2)) * 32 +
              (row_in_expert & 31)) * 4 + ((row_in_expert >> 5) & 3)) * 4 +
            (col_block & 3));
}

__global__ void __launch_bounds__(896, 2) silu_mul_fp4quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ mask,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ swizzled_scales,
    int m_per_expert, int K, int n_experts,
    int padded_m, int padded_scale_k
) {
    const int row = blockIdx.x;
    const int col_block = threadIdx.x;

    const int expert_idx = row / m_per_expert;
    const int row_in_expert = row - expert_idx * m_per_expert;

    const int swiz_off = swizzle_offset(row_in_expert, col_block, expert_idx,
                                         padded_m, padded_scale_k);

    if (row_in_expert >= __ldg(&mask[expert_idx])) {
        swizzled_scales[swiz_off] = 0;
        return;
    }

    const float gs = __ldg(&global_scales[expert_idx]);

    const int64_t base = (int64_t)row * 2 * K + col_block * 16;
    const int4* gptr = reinterpret_cast<const int4*>(input + base);
    const int4* uptr = reinterpret_cast<const int4*>(input + base + K);

    // Load first 8 gate + 8 up elements
    int4 g0 = __ldg(gptr);
    int4 u0 = __ldg(uptr);

    __nv_bfloat162* gp0 = reinterpret_cast<__nv_bfloat162*>(&g0);
    __nv_bfloat162* up0 = reinterpret_cast<__nv_bfloat162*>(&u0);

    // Process first 8 elements while second 8 are loading
    __nv_bfloat162 bmax = {__ushort_as_bfloat16(0), __ushort_as_bfloat16(0)};

    // Prefetch second loads
    int4 g1 = __ldg(gptr + 1);
    int4 u1 = __ldg(uptr + 1);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 gf = __bfloat1622float2(gp0[i]);
        float2 uf = __bfloat1622float2(up0[i]);
        gf.x = silu_f(gf.x) * uf.x;
        gf.y = silu_f(gf.y) * uf.y;
        __nv_bfloat162 r = __float22bfloat162_rn(gf);
        gp0[i] = r;
        bmax = __hmax2(bmax, __habs2(r));
    }

    __nv_bfloat162* gp1 = reinterpret_cast<__nv_bfloat162*>(&g1);
    __nv_bfloat162* up1 = reinterpret_cast<__nv_bfloat162*>(&u1);

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 gf = __bfloat1622float2(gp1[i]);
        float2 uf = __bfloat1622float2(up1[i]);
        gf.x = silu_f(gf.x) * uf.x;
        gf.y = silu_f(gf.y) * uf.y;
        __nv_bfloat162 r = __float22bfloat162_rn(gf);
        gp1[i] = r;
        bmax = __hmax2(bmax, __habs2(r));
    }

    float2 lm = __bfloat1622float2(bmax);
    float block_max = fmaxf(lm.x, lm.y);

    float sf_val = gs * (block_max * rcp_approx_ftz(6.0f));
    __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_val);

    float os = (block_max != 0.0f)
        ? rcp_approx_ftz(static_cast<float>(sf_fp8) * rcp_approx_ftz(gs))
        : 0.0f;

    float2 f0 = __bfloat1622float2(gp0[0]);
    float2 f1 = __bfloat1622float2(gp0[1]);
    float2 f2 = __bfloat1622float2(gp0[2]);
    float2 f3 = __bfloat1622float2(gp0[3]);

    uint32_t packed0 = cvt_8xfp32_to_e2m1(
        f0.x*os, f0.y*os, f1.x*os, f1.y*os,
        f2.x*os, f2.y*os, f3.x*os, f3.y*os);

    float2 f4 = __bfloat1622float2(gp1[0]);
    float2 f5 = __bfloat1622float2(gp1[1]);
    float2 f6 = __bfloat1622float2(gp1[2]);
    float2 f7 = __bfloat1622float2(gp1[3]);

    uint32_t packed1 = cvt_8xfp32_to_e2m1(
        f4.x*os, f4.y*os, f5.x*os, f5.y*os,
        f6.x*os, f6.y*os, f7.x*os, f7.y*os);

    *reinterpret_cast<uint2*>(output + (int64_t)row * (K / 2) + col_block * 8) =
        make_uint2(packed0, packed1);

    swizzled_scales[swiz_off] = *reinterpret_cast<uint8_t*>(&sf_fp8);
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

    int padded_m = ((M + 127) / 128) * 128;
    int factor = BLOCK_SIZE * 4;
    int padded_k_full = ((K + factor - 1) / factor) * factor;
    int padded_scale_k = padded_k_full / BLOCK_SIZE;

    auto opts_u8 = input.options().dtype(torch::kUInt8);
    auto fp4_out = torch::empty({B, M, K / 2}, opts_u8);
    auto swizzled_all = torch::empty({B, padded_m, padded_scale_k}, opts_u8);

    auto stream = c10::cuda::getCurrentCUDAStream();
    silu_mul_fp4quant_kernel<<<m_topk, scale_k, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        gs_ptr, mask.data_ptr<int32_t>(),
        fp4_out.data_ptr<uint8_t>(),
        swizzled_all.data_ptr<uint8_t>(),
        M, K, B,
        padded_m, padded_scale_k);

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
