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

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

__device__ __forceinline__ uint64_t cvt_16xfp32_to_e2m1(float2* fp2) {
    uint64_t result;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3, b4, b5, b6, b7;\n"
        ".reg .b32 lo, hi;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0,  %2,  %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1,  %4,  %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2,  %6,  %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3,  %8,  %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b4, %10,  %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b5, %12, %11;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b6, %14, %13;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b7, %16, %15;\n"
        "mov.b32 lo, {b0, b1, b2, b3};\n"
        "mov.b32 hi, {b4, b5, b6, b7};\n"
        "mov.b64 %0, {lo, hi};\n"
        "}"
        : "=l"(result)
        : "f"(fp2[0].x), "f"(fp2[0].y), "f"(fp2[1].x), "f"(fp2[1].y),
          "f"(fp2[2].x), "f"(fp2[2].y), "f"(fp2[3].x), "f"(fp2[3].y),
          "f"(fp2[4].x), "f"(fp2[4].y), "f"(fp2[5].x), "f"(fp2[5].y),
          "f"(fp2[6].x), "f"(fp2[6].y), "f"(fp2[7].x), "f"(fp2[7].y)
    );
    return result;
}

__global__ void
__launch_bounds__(448, 8)
silu_mul_fp4quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ mask,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int m_per_expert, int K, int n_experts, int block_size
) {
    const int row = blockIdx.x;
    const int col_block = threadIdx.x;
    const int expert_idx = row / m_per_expert;
    const int row_in_expert = row - expert_idx * m_per_expert;

    if (row_in_expert >= __ldg(&mask[expert_idx])) return;

    const float gs = __ldg(&global_scales[expert_idx]);
    const int64_t row_base = (int64_t)row * (K * 2);
    const int elem_offset = col_block << 4;

    const int4* gate_v = reinterpret_cast<const int4*>(input + row_base + elem_offset);
    const int4* up_v   = reinterpret_cast<const int4*>(input + row_base + K + elem_offset);

    int4 gv0 = __ldg(gate_v);
    int4 gv1 = __ldg(gate_v + 1);
    int4 uv0 = __ldg(up_v);
    int4 uv1 = __ldg(up_v + 1);

    __nv_bfloat162* g0 = reinterpret_cast<__nv_bfloat162*>(&gv0);
    __nv_bfloat162* u0 = reinterpret_cast<__nv_bfloat162*>(&uv0);
    __nv_bfloat162* g1 = reinterpret_cast<__nv_bfloat162*>(&gv1);
    __nv_bfloat162* u1 = reinterpret_cast<__nv_bfloat162*>(&uv1);

    __nv_bfloat162 activated[8];
    __nv_bfloat162 localMax;

    float2 gf0 = __bfloat1622float2(g0[0]);
    float2 uf0 = __bfloat1622float2(u0[0]);
    gf0.x = silu_f(gf0.x) * uf0.x;
    gf0.y = silu_f(gf0.y) * uf0.y;
    activated[0] = __float22bfloat162_rn(gf0);
    localMax = __habs2(activated[0]);

    #pragma unroll
    for (int i = 1; i < 4; i++) {
        float2 gf = __bfloat1622float2(g0[i]);
        float2 uf = __bfloat1622float2(u0[i]);
        gf.x = silu_f(gf.x) * uf.x;
        gf.y = silu_f(gf.y) * uf.y;
        activated[i] = __float22bfloat162_rn(gf);
        localMax = __hmax2(localMax, __habs2(activated[i]));
    }
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float2 gf = __bfloat1622float2(g1[i]);
        float2 uf = __bfloat1622float2(u1[i]);
        gf.x = silu_f(gf.x) * uf.x;
        gf.y = silu_f(gf.y) * uf.y;
        activated[4 + i] = __float22bfloat162_rn(gf);
        localMax = __hmax2(localMax, __habs2(activated[4 + i]));
    }

    float amax = fmaxf(__bfloat162float(localMax.x), __bfloat162float(localMax.y));

    float sf_val = gs * (amax * rcp_approx_ftz(6.0f));
    __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf_val);
    uint8_t sf_byte = sf_fp8.__x;
    float sf_narrow = static_cast<float>(sf_fp8);
    float output_scale = (amax != 0.0f) ? rcp_approx_ftz(sf_narrow * rcp_approx_ftz(gs)) : 0.0f;

    float2 fp2[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        fp2[i] = __bfloat1622float2(activated[i]);
        fp2[i].x *= output_scale;
        fp2[i].y *= output_scale;
    }

    uint64_t e2m1 = cvt_16xfp32_to_e2m1(fp2);

    reinterpret_cast<uint64_t*>(output + (int64_t)row * (K / 2))[col_block] = e2m1;

    const int numCols_sf = ((K + 63) / 64) * 4;
    const int padded_m = ((m_per_expert + 127) / 128) * 128;
    const int numKTiles = (numCols_sf + 3) >> 2;

    const int mIdx = row_in_expert;
    const int kIdx = col_block;

    int64_t sf_offset = (int64_t)expert_idx * padded_m * (numKTiles << 2) +
                         (int64_t)(mIdx >> 7) * (numKTiles * 512) +
                         (int64_t)(kIdx >> 2) * 512 +
                         (mIdx & 31) * 16 +
                         ((mIdx & 127) >> 5) * 4 +
                         (kIdx & 3);

    block_scales[sf_offset] = sf_byte;
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
    int padded_scale_k = ((K + 63) / 64) * 4;

    auto swizzled_all = torch::empty({B, padded_m, padded_scale_k},
                                      input.options().dtype(torch::kUInt8));

    silu_mul_fp4quant_kernel<<<m_topk, scale_k>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.view({m_topk, K2}).data_ptr<at::BFloat16>()),
        gs_ptr, mask.data_ptr<int32_t>(),
        fp4_out.view({m_topk, K / 2}).data_ptr<uint8_t>(),
        swizzled_all.data_ptr<uint8_t>(),
        M, K, B, BLOCK_SIZE);

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
