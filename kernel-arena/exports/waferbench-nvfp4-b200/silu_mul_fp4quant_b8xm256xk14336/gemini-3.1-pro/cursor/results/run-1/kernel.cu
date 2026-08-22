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

__device__ __forceinline__ float fast_silu(float x) {
    return x * rcp_approx_ftz(1.0f + __expf(-x));
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
    uint8_t* __restrict__ block_scales,
    int m_per_expert, int K, int n_experts, int block_size,
    int padded_m, int padded_scale_k
) {
    int row = blockIdx.x;
    int b = row / m_per_expert;
    int m = row % m_per_expert;
    int tid = threadIdx.x;
    
    int rm_idx = m >> 7;
    int dim1 = (m & 127) >> 5;
    int dim2 = m & 31;
    int rk_idx = tid >> 2;
    int dim4 = tid & 3;
    int rk = padded_scale_k >> 2;
    int out_idx = b * (padded_m * padded_scale_k) + rm_idx * (rk << 9) + (rk_idx << 9) + (dim2 << 4) + (dim1 << 2) + dim4;

    if (m >= mask[b]) {
        ((uint64_t*)output)[row * (K >> 4) + tid] = 0;
        block_scales[out_idx] = 0;
        return;
    }

    int base_idx = (row * K) >> 2;
    int k_offset = K >> 3;
    
    const float4* input_f4 = (const float4*)input;
    
    float4 gate_f4[2];
    float4 up_f4[2];
    
    gate_f4[0] = __ldg(&input_f4[base_idx + (tid << 1)]);
    gate_f4[1] = __ldg(&input_f4[base_idx + (tid << 1) + 1]);
    up_f4[0] = __ldg(&input_f4[base_idx + k_offset + (tid << 1)]);
    up_f4[1] = __ldg(&input_f4[base_idx + k_offset + (tid << 1) + 1]);

    float x[16];
    float vec_max = 0.0f;
    
    #pragma unroll
    for (int k = 0; k < 2; k++) {
        uint4 gate_u4 = *(uint4*)&gate_f4[k];
        uint4 up_u4 = *(uint4*)&up_f4[k];
        
        uint32_t gate_u32[4] = {gate_u4.x, gate_u4.y, gate_u4.z, gate_u4.w};
        uint32_t up_u32[4] = {up_u4.x, up_u4.y, up_u4.z, up_u4.w};
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float2 gate_f2 = __bfloat1622float2(*(__nv_bfloat162*)&gate_u32[j]);
            float2 up_f2 = __bfloat1622float2(*(__nv_bfloat162*)&up_u32[j]);
            
            float2 x_f2;
            x_f2.x = fast_silu(gate_f2.x) * up_f2.x;
            x_f2.y = fast_silu(gate_f2.y) * up_f2.y;
            __nv_bfloat162 x_bf2 = __floats2bfloat162_rn(x_f2.x, x_f2.y);
            x_f2 = __bfloat1622float2(x_bf2);
            
            x[k * 8 + j * 2] = x_f2.x;
            x[k * 8 + j * 2 + 1] = x_f2.y;
            
            vec_max = fmaxf(vec_max, fabsf(x_f2.x));
            vec_max = fmaxf(vec_max, fabsf(x_f2.y));
        }
    }

    float gs = global_scales ? global_scales[b] : 1.0f;
    float scale = gs * (vec_max * 0.16666666666666666f);
    
    __nv_fp8_e4m3 scale_fp8(scale);
    float scale_fp8_f = float(scale_fp8);
    
    float rec_gs = (gs == 0.0f) ? 0.0f : (1.0f / gs);
    float output_scale = (scale_fp8_f * rec_gs == 0.0f) ? 0.0f : (1.0f / (scale_fp8_f * rec_gs));
    
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        x[j] = fminf(fmaxf(x[j] * output_scale, -6.0f), 6.0f);
    }
    
    uint32_t out0 = cvt_8xfp32_to_e2m1(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
    uint32_t out1 = cvt_8xfp32_to_e2m1(x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
    
    ((uint64_t*)output)[row * (K >> 4) + tid] = ((uint64_t)out1 << 32) | out0;
    block_scales[out_idx] = *(uint8_t*)&scale_fp8;
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

    auto output_scales = torch::zeros({B, padded_m, padded_scale_k}, input.options().dtype(torch::kUInt8));

    silu_mul_fp4quant_kernel<<<m_topk, scale_k>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        gs_ptr, mask.data_ptr<int32_t>(),
        fp4_out.data_ptr<uint8_t>(),
        output_scales.data_ptr<uint8_t>(),
        M, K, B, BLOCK_SIZE, padded_m, padded_scale_k);

    fp4_out = fp4_out.permute({1, 2, 0});

    auto output_scales_reshaped = output_scales
        .view({B * padded_m, padded_scale_k})
        .view(c10::ScalarType::Float8_e4m3fn)
        .view({B, padded_m / 128, padded_scale_k / 4, 32, 4, 4})
        .permute({3, 4, 1, 5, 2, 0});

    return {fp4_out, output_scales_reshaped};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul_scaled_nvfp4_experts_quantize",
          &silu_and_mul_scaled_nvfp4_experts_quantize,
          py::arg("input"), py::arg("mask"),
          py::arg("global_scale") = py::none());
}
