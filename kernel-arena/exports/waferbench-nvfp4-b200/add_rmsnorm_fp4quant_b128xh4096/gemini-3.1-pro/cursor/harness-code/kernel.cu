#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t pack_fp4_e2m1(float left, float right) {
    uint16_t byte0;
    asm("{\n"
        "    .reg .b8 b0;\n"
        "    cvt.rn.satfinite.e2m1x2.f32 b0, %1, %2;\n"
        "    cvt.u16.u8 %0, b0;\n"
        "}" : "=h"(byte0) : "f"(right), "f"(left));
    return (uint8_t)byte0;
}

__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    uint32_t res;
    asm("{\n"
        "    .reg .b16 fp8_pair;\n"
        "    .reg .f32 zero;\n"
        "    mov.f32 zero, 0f00000000;\n"
        "    cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, %1;\n"
        "    cvt.u32.u16 %0, fp8_pair;\n"
        "}" : "=r"(res) : "f"(val));
    return (uint8_t)(res & 0xFF);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    int biased_exp = (bits >> 3) & 0xF; int man = bits & 0x7;
    if (biased_exp == 0) return (float(man) / 8.0f) * (1.0f / 64.0f);
    return (1.0f + float(man) / 8.0f) * exp2f(float(biased_exp - E4M3_BIAS));
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t nibble) {
    uint8_t sign = (nibble >> 3) & 1;
    uint8_t exp  = (nibble >> 1) & 3;
    uint8_t man  = nibble & 1;
    float val;
    if (exp == 0) { val = man * 0.5f; }
    else { val = (1.0f + man * 0.5f) * float(1 << (exp - 1)); }
    return sign ? -val : val;
}

__global__ void nvfp4_dequant_kernel(
    const uint8_t* __restrict__ fp4_packed,
    const uint8_t* __restrict__ block_scales,
    float inv_global_scale,
    float* __restrict__ output,
    int total_elems, int hidden, int block_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elems) return;
    int row = idx / hidden, col = idx % hidden;
    int blk = col / block_size, within = col % block_size;
    uint8_t packed = fp4_packed[row * (hidden / 2) + blk * (block_size / 2) + within / 2];
    uint8_t nibble = (within % 2 == 0) ? (packed & 0xF) : ((packed >> 4) & 0xF);
    float bs = e4m3_to_float(block_scales[row * (hidden / block_size) + blk]);
    output[idx] = fp4_e2m1_to_float(nibble) * bs * inv_global_scale;
}

torch::Tensor nvfp4_dequant(torch::Tensor fp4_packed, torch::Tensor block_scales,
                            float global_scale, int hidden, int block_size) {
    int total_rows = fp4_packed.size(0);
    auto out = torch::empty({total_rows, hidden}, fp4_packed.options().dtype(torch::kFloat32));
    int n = total_rows * hidden;
    nvfp4_dequant_kernel<<<(n + 255) / 256, 256>>>(
        fp4_packed.data_ptr<uint8_t>(), block_scales.data_ptr<uint8_t>(),
        1.0f / global_scale, out.data_ptr<float>(), n, hidden, block_size);
    return out;
}

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    int num_warps = (blockDim.x + 31) / 32;
    val = (threadIdx.x < num_warps) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ d_global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blk = tid;
    
    if (blk * 16 >= hidden) return;
    
    int offset = row * hidden + blk * 16;
    
    float4 in_f4_0 = reinterpret_cast<const float4*>(input + offset)[0];
    float4 in_f4_1 = reinterpret_cast<const float4*>(input + offset)[1];
    
    float4 res_f4_0 = reinterpret_cast<const float4*>(residual + offset)[0];
    float4 res_f4_1 = reinterpret_cast<const float4*>(residual + offset)[1];
    
    __nv_bfloat162* in_bf2_0 = reinterpret_cast<__nv_bfloat162*>(&in_f4_0);
    __nv_bfloat162* in_bf2_1 = reinterpret_cast<__nv_bfloat162*>(&in_f4_1);
    
    __nv_bfloat162* res_bf2_0 = reinterpret_cast<__nv_bfloat162*>(&res_f4_0);
    __nv_bfloat162* res_bf2_1 = reinterpret_cast<__nv_bfloat162*>(&res_f4_1);
    
    float sum_sq = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 in_f2 = __bfloat1622float2(in_bf2_0[i]);
        float2 res_f2 = __bfloat1622float2(res_bf2_0[i]);
        float2 h_f2;
        h_f2.x = in_f2.x + res_f2.x;
        h_f2.y = in_f2.y + res_f2.y;
        res_bf2_0[i] = __float22bfloat162_rn(h_f2);
        sum_sq += h_f2.x * h_f2.x + h_f2.y * h_f2.y;
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 in_f2 = __bfloat1622float2(in_bf2_1[i]);
        float2 res_f2 = __bfloat1622float2(res_bf2_1[i]);
        float2 h_f2;
        h_f2.x = in_f2.x + res_f2.x;
        h_f2.y = in_f2.y + res_f2.y;
        res_bf2_1[i] = __float22bfloat162_rn(h_f2);
        sum_sq += h_f2.x * h_f2.x + h_f2.y * h_f2.y;
    }
    
    reinterpret_cast<float4*>(residual + offset)[0] = res_f4_0;
    reinterpret_cast<float4*>(residual + offset)[1] = res_f4_1;
    
    float total_sum_sq = blockReduceSum(sum_sq);
    
    __shared__ float s_rms;
    if (tid == 0) {
        s_rms = rsqrtf(total_sum_sq / hidden + eps);
    }
    __syncthreads();
    
    float rms = s_rms;
    float global_scale = __ldg(d_global_scale);
    
    float4 w_f4_0 = __ldg(reinterpret_cast<const float4*>(weight + blk * 16));
    float4 w_f4_1 = __ldg(reinterpret_cast<const float4*>(weight + blk * 16) + 1);
    
    __nv_bfloat162* w_bf2_0 = reinterpret_cast<__nv_bfloat162*>(&w_f4_0);
    __nv_bfloat162* w_bf2_1 = reinterpret_cast<__nv_bfloat162*>(&w_f4_1);
    
    float vals[16];
    float amax = 0.0f;
    
    __nv_bfloat162 hw_h2[8];
    float y_f32[16];
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 h_f2 = __bfloat1622float2(res_bf2_0[i]);
        float2 w_f2 = __bfloat1622float2(w_bf2_0[i]);
        
        y_f32[i*2] = (h_f2.x * rms) * w_f2.x;
        y_f32[i*2+1] = (h_f2.y * rms) * w_f2.y;
        
        amax = fmaxf(amax, fabsf(y_f32[i*2]));
        amax = fmaxf(amax, fabsf(y_f32[i*2+1]));
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 h_f2 = __bfloat1622float2(res_bf2_1[i]);
        float2 w_f2 = __bfloat1622float2(w_bf2_1[i]);
        
        y_f32[8+i*2] = (h_f2.x * rms) * w_f2.x;
        y_f32[8+i*2+1] = (h_f2.y * rms) * w_f2.y;
        
        amax = fmaxf(amax, fabsf(y_f32[8+i*2]));
        amax = fmaxf(amax, fabsf(y_f32[8+i*2+1]));
    }
    
    float fp4_max_rcp;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(fp4_max_rcp) : "f"(6.0f));
    float scale = amax * global_scale * fp4_max_rcp;
    scale = fminf(scale, 448.0f);
    uint8_t e4m3_scale_bits = float_to_e4m3(scale);
    out_scales[row * (hidden / 16) + blk] = e4m3_scale_bits;
    
    uint32_t fp8_val = e4m3_scale_bits;
    float rcp_scale;
    asm(
        "{\n"
        "    .reg .pred p_zero;\n"
        "    .reg .u32 exp_u, mant_u;\n"
        "    .reg .s32 exp_s;\n"
        "    .reg .f32 exp_f, mant_f, fp8_float, result;\n"
        "    setp.eq.u32 p_zero, %1, 0;\n"
        "    and.b32 mant_u, %1, 7;\n"
        "    shr.b32 exp_u, %1, 3;\n"
        "    and.b32 exp_u, exp_u, 15;\n"
        "    sub.s32 exp_s, exp_u, 7;\n"
        "    cvt.rn.f32.s32 exp_f, exp_s;\n"
        "    ex2.approx.f32 exp_f, exp_f;\n"
        "    cvt.rn.f32.u32 mant_f, mant_u;\n"
        "    fma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;\n"
        "    mul.f32 fp8_float, exp_f, mant_f;\n"
        "    rcp.approx.ftz.f32 result, fp8_float;\n"
        "    selp.f32 %0, 0f00000000, result, p_zero;\n"
        "}"
        : "=f"(rcp_scale) : "r"(fp8_val)
    );
    float inv_scale = rcp_scale * global_scale;
    
    uint8_t packed[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        packed[i] = pack_fp4_e2m1(y_f32[i*2] * inv_scale, y_f32[i*2+1] * inv_scale);
    }
    
    int out_offset = row * (hidden / 2) + blk * 8;
    reinterpret_cast<uint2*>(out_fp4 + out_offset)[0] = reinterpret_cast<uint2*>(packed)[0];
}

std::vector<torch::Tensor> add_rmsnorm_fp4quant(
    torch::Tensor input, torch::Tensor residual, torch::Tensor weight,
    std::optional<torch::Tensor> y_fp4,
    std::optional<torch::Tensor> block_scale,
    std::optional<torch::Tensor> global_scale,
    double eps, int64_t block_size,
    std::optional<std::string> scale_format,
    bool is_sf_swizzled_layout,
    bool output_both_sf_layouts,
    std::optional<torch::Tensor> block_scale_unswizzled
) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kBFloat16);
    TORCH_CHECK(input.dim() == 2);
    int batch = input.size(0), hidden = input.size(1);
    TORCH_CHECK(hidden % block_size == 0);

    const float* d_gs = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    auto scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));

    int num_blocks_per_row = hidden / (int)block_size;
    int block_threads = 1;
    while (block_threads < num_blocks_per_row) block_threads <<= 1;
    TORCH_CHECK(block_threads <= 1024);
    fused_add_rmsnorm_nvfp4_kernel<<<batch, block_threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
        d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
        batch, hidden, (int)block_size);

    torch::Tensor fp4_out, scale_out;
    if (y_fp4.has_value()) {
        y_fp4->view(torch::kUInt8).copy_(fp4_raw);
        fp4_out = *y_fp4;
    } else {
        fp4_out = fp4_raw.view(c10::ScalarType::Float4_e2m1fn_x2);
    }
    if (block_scale.has_value()) {
        block_scale->view(torch::kUInt8).copy_(scales_raw);
        scale_out = *block_scale;
    } else {
        scale_out = scales_raw.view(c10::ScalarType::Float8_e4m3fn);
    }
    return {fp4_out, scale_out};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_rmsnorm_fp4quant", &add_rmsnorm_fp4quant,
          py::arg("input"), py::arg("residual"), py::arg("weight"),
          py::arg("y_fp4") = py::none(), py::arg("block_scale") = py::none(),
          py::arg("global_scale") = py::none(), py::arg("eps") = 1e-6,
          py::arg("block_size") = 16, py::arg("scale_format") = py::none(),
          py::arg("is_sf_swizzled_layout") = false,
          py::arg("output_both_sf_layouts") = false,
          py::arg("block_scale_unswizzled") = py::none());
    m.def("nvfp4_dequant", &nvfp4_dequant);
}
