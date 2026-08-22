#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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
        "}\n"
        : "=r"(result)
        : "f"(v0), "f"(v1), "f"(v2), "f"(v3),
          "f"(v4), "f"(v5), "f"(v6), "f"(v7)
    );
    return result;
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void __launch_bounds__(896) silu_mul_fp4quant_kernel_opt(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ global_scales,
    const int32_t* __restrict__ mask,
    uint8_t* __restrict__ output,
    uint8_t* __restrict__ block_scales,
    int M, int K, int B, int block_size
) {
    int m_idx = blockIdx.x;
    int b = m_idx / M;
    int m = m_idx % M;
    
    if (m >= mask[b]) return;

    float gs = global_scales ? global_scales[b] : 1.0f;
    
    int tid = threadIdx.x;
    
    const float4* row_in_f4 = reinterpret_cast<const float4*>(input + m_idx * (2 * K));
    int K_f4 = K / 8;
    
    int f4_idx = tid * 2;
    
    float4 gate0_f4 = __ldg(&row_in_f4[f4_idx]);
    float4 gate1_f4 = __ldg(&row_in_f4[f4_idx + 1]);
    float4 up0_f4 = __ldg(&row_in_f4[K_f4 + f4_idx]);
    float4 up1_f4 = __ldg(&row_in_f4[K_f4 + f4_idx + 1]);
    
    const __nv_bfloat162* gate0_bf2 = reinterpret_cast<const __nv_bfloat162*>(&gate0_f4);
    const __nv_bfloat162* gate1_bf2 = reinterpret_cast<const __nv_bfloat162*>(&gate1_f4);
    const __nv_bfloat162* up0_bf2 = reinterpret_cast<const __nv_bfloat162*>(&up0_f4);
    const __nv_bfloat162* up1_bf2 = reinterpret_cast<const __nv_bfloat162*>(&up1_f4);
    
    float vals[16];
    float max_val = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 gate = __bfloat1622float2(gate0_bf2[i]);
        float2 up = __bfloat1622float2(up0_bf2[i]);
        float val0 = silu(gate.x) * up.x;
        float val1 = silu(gate.y) * up.y;
        __nv_bfloat162 val_bf2 = __float22bfloat162_rn(make_float2(val0, val1));
        float2 val_f2 = __bfloat1622float2(val_bf2);
        vals[2 * i] = val_f2.x;
        vals[2 * i + 1] = val_f2.y;
        max_val = fmaxf(max_val, fabsf(val_f2.x));
        max_val = fmaxf(max_val, fabsf(val_f2.y));
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 gate = __bfloat1622float2(gate1_bf2[i]);
        float2 up = __bfloat1622float2(up1_bf2[i]);
        float val0 = silu(gate.x) * up.x;
        float val1 = silu(gate.y) * up.y;
        __nv_bfloat162 val_bf2 = __float22bfloat162_rn(make_float2(val0, val1));
        float2 val_f2 = __bfloat1622float2(val_bf2);
        vals[8 + 2 * i] = val_f2.x;
        vals[8 + 2 * i + 1] = val_f2.y;
        max_val = fmaxf(max_val, fabsf(val_f2.x));
        max_val = fmaxf(max_val, fabsf(val_f2.y));
    }
    
    float scale = gs * (max_val * (1.0f / 6.0f));
    __nv_fp8_storage_t scale_fp8_storage = __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3);
    float scale_f32 = (float)reinterpret_cast<__nv_fp8_e4m3&>(scale_fp8_storage);
    
    float output_scale = (scale_f32 == 0.0f) ? 0.0f : (1.0f / (scale_f32 * (1.0f / gs)));
    
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        vals[i] *= output_scale;
    }
    
    uint32_t out0 = cvt_8xfp32_to_e2m1(vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7]);
    uint32_t out1 = cvt_8xfp32_to_e2m1(vals[8], vals[9], vals[10], vals[11], vals[12], vals[13], vals[14], vals[15]);
    
    uint2* out_ptr = reinterpret_cast<uint2*>(output + m_idx * (K / 2));
    out_ptr[tid] = make_uint2(out0, out1);
    
    int k_idx = tid;
    int scale_k = K / block_size;
    int padded_m = ((M + 127) / 128) * 128;
    int factor = block_size * 4;
    int padded_k_full = ((K + factor - 1) / factor) * factor;
    int padded_scale_k = padded_k_full / block_size;
    int rk = padded_scale_k / 4;
    
    int m_rm = m / 128;
    int m_4 = (m % 128) / 32;
    int m_32 = m % 32;
    int k_rk = k_idx / 4;
    int k_4 = k_idx % 4;
    
    int swizzled_idx = b * (padded_m * padded_scale_k) +
                       m_rm * (rk * 512) +
                       k_rk * 512 +
                       m_32 * 16 +
                       m_4 * 4 +
                       k_4;
                       
    block_scales[swizzled_idx] = scale_fp8_storage;
}

std::vector<torch::Tensor> silu_and_mul_scaled_nvfp4_experts_quantize(
    torch::Tensor input,
    torch::Tensor mask,
    std::optional<torch::Tensor> global_scale
) {
    int B = input.size(0), M = input.size(1), K2 = input.size(2);
    int K = K2 / 2;
    constexpr int BLOCK_SIZE = 16;
    int m_topk = B * M;

    int padded_m = ((M + 127) / 128) * 128;
    int factor = BLOCK_SIZE * 4;
    int padded_k_full = ((K + factor - 1) / factor) * factor;
    int padded_scale_k = padded_k_full / BLOCK_SIZE;

    float* gs_ptr = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto fp4_out = torch::empty({B, M, K / 2}, input.options().dtype(torch::kUInt8));
    auto linear_scales = torch::zeros({B * padded_m * padded_scale_k}, input.options().dtype(torch::kUInt8));

    int threads_per_block = K / BLOCK_SIZE; // 896

    silu_mul_fp4quant_kernel_opt<<<m_topk, threads_per_block>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.view({m_topk, K2}).data_ptr<at::BFloat16>()),
        gs_ptr, mask.data_ptr<int32_t>(),
        fp4_out.data_ptr<uint8_t>(),
        linear_scales.data_ptr<uint8_t>(),
        M, K, B, BLOCK_SIZE);

    fp4_out = fp4_out.permute({1, 2, 0});

    int rm = padded_m / 128;
    int rk = padded_scale_k / 4;
    
    auto output_scales = linear_scales
        .view({B, rm, rk, 32, 4, 4})
        .view(c10::ScalarType::Float8_e4m3fn)
        .permute({3, 4, 1, 5, 2, 0});

    return {fp4_out, output_scales};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul_scaled_nvfp4_experts_quantize",
          &silu_and_mul_scaled_nvfp4_experts_quantize,
          py::arg("input"), py::arg("mask"),
          py::arg("global_scale") = py::none());
}
