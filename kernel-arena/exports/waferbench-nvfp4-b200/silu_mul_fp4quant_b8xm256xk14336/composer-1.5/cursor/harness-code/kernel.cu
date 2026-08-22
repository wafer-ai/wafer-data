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

__global__ void
silu_mul_fp4quant_kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ mask,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int m_per_expert, int K, int n_experts, int block_size
) {
    const int scale_k = K / block_size;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = tid / scale_k;
    const int t = tid % scale_k;
    if (row >= n_experts * m_per_expert || t >= scale_k) return;

    const int batch = row / m_per_expert;
    const float gs = global_scales ? global_scales[batch] : 1.0f;

    const __nv_bfloat162* gate_ptr = reinterpret_cast<const __nv_bfloat162*>(
        input + row * (2 * K) + t * block_size);
    const __nv_bfloat162* up_ptr = reinterpret_cast<const __nv_bfloat162*>(
        input + row * (2 * K) + K + t * block_size);

    float vals[16];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        __nv_bfloat162 g = gate_ptr[i];
        __nv_bfloat162 u = up_ptr[i];
        float2 gf = __bfloat1622float2(g);
        float2 uf = __bfloat1622float2(u);
        float v0 = silu(gf.x) * uf.x;
        float v1 = silu(gf.y) * uf.y;
        vals[2*i] = __bfloat162float(__float2bfloat16_rn(v0));
        vals[2*i+1] = __bfloat162float(__float2bfloat16_rn(v1));
    }

    float vmax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
        const float a = fabsf(vals[i]);
        vmax = fmaxf(vmax, a);
    }
    const float vec_max = fmaxf(vmax, 1e-12f);
    float scale = gs * (vec_max * rcp_approx_ftz(6.0f));
    __nv_fp8_e4m3 scale_fp8_val(scale);
    scale = static_cast<float>(scale_fp8_val);
    const float output_scale = vec_max != 0.0f
        ? rcp_approx_ftz(scale * rcp_approx_ftz(gs))
        : 0.0f;

    float scaled[16];
#pragma unroll
    for (int i = 0; i < 16; i++) {
        scaled[i] = fminf(fmaxf(vals[i] * output_scale, -6.0f), 6.0f);
    }

    const uint32_t packed0 = cvt_8xfp32_to_e2m1(
        scaled[0], scaled[1], scaled[2], scaled[3],
        scaled[4], scaled[5], scaled[6], scaled[7]);
    const uint32_t packed1 = cvt_8xfp32_to_e2m1(
        scaled[8], scaled[9], scaled[10], scaled[11],
        scaled[12], scaled[13], scaled[14], scaled[15]);

    uint8_t* out_ptr = output + row * (K / 2) + t * 8;
    *reinterpret_cast<uint32_t*>(out_ptr) = packed0;
    *reinterpret_cast<uint32_t*>(out_ptr + 4) = packed1;
    block_scales[row * scale_k + t] = reinterpret_cast<const uint8_t&>(scale_fp8_val);
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
    auto linear_scales = torch::zeros({m_topk, scale_k}, input.options().dtype(torch::kUInt8));

    {
        const int total_work = m_topk * scale_k;
        const int block_dim = 128;
        const int grid_dim = (total_work + block_dim - 1) / block_dim;
        silu_mul_fp4quant_kernel<<<grid_dim, block_dim>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.view({m_topk, K2}).data_ptr<at::BFloat16>()),
        gs_ptr, mask.data_ptr<int32_t>(),
        fp4_out.view({m_topk, K / 2}).data_ptr<uint8_t>(),
        linear_scales.data_ptr<uint8_t>(),
        M, K, B, BLOCK_SIZE);
    }

    int padded_m = ((M + 127) / 128) * 128;
    int factor = BLOCK_SIZE * 4;
    int padded_k_full = ((K + factor - 1) / factor) * factor;
    int padded_scale_k = padded_k_full / BLOCK_SIZE;
    int padded_k_int32 = padded_scale_k / 4;

    auto swizzled_all = torch::zeros({B, padded_m, padded_scale_k},
                                      input.options().dtype(torch::kUInt8));
    for (int b = 0; b < B; b++) {
        swizzled_all[b].copy_(
            swizzle_scales(linear_scales.slice(0, b * M, (b + 1) * M), M, K, BLOCK_SIZE));
    }

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
