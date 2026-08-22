#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint8_t sign = 0;
    if (val < 0.0f) { sign = 1; val = -val; }
    if (val > FP4_MAX) val = FP4_MAX;
    uint8_t encoded;
    if      (val <= 0.25f) encoded = 0b000;
    else if (val < 0.75f)  encoded = 0b001;
    else if (val <= 1.25f) encoded = 0b010;
    else if (val < 1.75f)  encoded = 0b011;
    else if (val <= 2.5f)  encoded = 0b100;
    else if (val < 3.5f)   encoded = 0b101;
    else if (val <= 5.0f)  encoded = 0b110;
    else                   encoded = 0b111;
    return (sign << 3) | encoded;
}

__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    if (val <= 0.0f) return 0;
    if (val > E4M3_MAX) val = E4M3_MAX;
    float min_normal = 1.0f / 64.0f;
    if (val < min_normal) {
        int man = __float2int_rn(val * 64.0f * 8.0f);
        if (man > 7) man = 7; if (man < 0) man = 0;
        return (uint8_t)man;
    }
    int exp_unbiased = (int)floorf(log2f(val));
    if (exp_unbiased < -6) exp_unbiased = -6;
    if (exp_unbiased >  8) exp_unbiased = 8;
    int biased_exp = exp_unbiased + E4M3_BIAS;
    if (biased_exp < 1) biased_exp = 1; if (biased_exp > 15) biased_exp = 15;
    float pow2 = exp2f((float)exp_unbiased);
    int man = __float2int_rn((val / pow2 - 1.0f) * 8.0f);
    if (man > 7) { man = 0; biased_exp++; }
    if (man < 0) man = 0;
    if (biased_exp > 15) { biased_exp = 15; man = 6; }
    return (uint8_t)((biased_exp << 3) | man);
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

template <typename T>
__device__ __forceinline__ T bit_cast_u32(uint32_t x) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return *reinterpret_cast<T*>(&x);
}

template <typename T>
__device__ __forceinline__ uint32_t bit_cast_to_u32(T x) {
    static_assert(sizeof(T) == sizeof(uint32_t));
    return *reinterpret_cast<uint32_t*>(&x);
}

__device__ __forceinline__ uint32_t bit_cast_f32(float x) {
    return *reinterpret_cast<uint32_t*>(&x);
}

__device__ __forceinline__ float warp_reduce_sum(float val, unsigned mask) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
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

template <int BLOCK_SIZE>
__launch_bounds__(256, 2) __global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ d_global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    constexpr int kPairsPerThread = BLOCK_SIZE / 2;
    constexpr float kInvFp4Max = 1.0f / FP4_MAX;

    const int row = blockIdx.x;
    const int block_idx = threadIdx.x;
    const int num_blocks = hidden / BLOCK_SIZE;
    if (row >= batch || block_idx >= num_blocks) {
        return;
    }

    __shared__ float smem[33];
    __shared__ float s_inv_rms;
    __shared__ float s_global_scale;

    float vals[BLOCK_SIZE];
    float sumsq = 0.0f;

    const int row_offset = row * hidden;
    const int col_offset = block_idx * BLOCK_SIZE;

    const uint4* input_pack =
        reinterpret_cast<const uint4*>(input + row_offset + col_offset);
    uint4* residual_pack =
        reinterpret_cast<uint4*>(residual + row_offset + col_offset);

    uint4 residual_out0, residual_out1;
    auto process_pair = [&](int pair_idx, uint32_t in_bits, uint32_t res_bits, uint32_t& out_bits) {
        __nv_bfloat162 in_bf2 = bit_cast_u32<__nv_bfloat162>(in_bits);
        __nv_bfloat162 res_bf2 = bit_cast_u32<__nv_bfloat162>(res_bits);
        float2 in2 = __bfloat1622float2(in_bf2);
        float2 res2 = __bfloat1622float2(res_bf2);
        float h0 = in2.x + res2.x;
        float h1 = in2.y + res2.y;
        __nv_bfloat162 h_bf = __floats2bfloat162_rn(h0, h1);
        float2 h2 = __bfloat1622float2(h_bf);
        vals[2 * pair_idx] = h2.x;
        vals[2 * pair_idx + 1] = h2.y;
        sumsq = fmaf(h0, h0, sumsq);
        sumsq = fmaf(h1, h1, sumsq);
        out_bits = bit_cast_to_u32(h_bf);
    };

    uint4 in0 = input_pack[0];
    uint4 in1 = input_pack[1];
    uint4 res0 = residual_pack[0];
    uint4 res1 = residual_pack[1];

    process_pair(0, in0.x, res0.x, residual_out0.x);
    process_pair(1, in0.y, res0.y, residual_out0.y);
    process_pair(2, in0.z, res0.z, residual_out0.z);
    process_pair(3, in0.w, res0.w, residual_out0.w);
    process_pair(4, in1.x, res1.x, residual_out1.x);
    process_pair(5, in1.y, res1.y, residual_out1.y);
    process_pair(6, in1.z, res1.z, residual_out1.z);
    process_pair(7, in1.w, res1.w, residual_out1.w);

    residual_pack[0] = residual_out0;
    residual_pack[1] = residual_out1;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;
    const unsigned mask = __activemask();

    float warp_sum = warp_reduce_sum(sumsq, mask);
    if (lane == 0) {
        smem[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = (lane < num_warps) ? smem[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum, 0xffffffffu);
        if (lane == 0) {
            s_inv_rms = rsqrtf(block_sum / static_cast<float>(hidden) + eps);
            s_global_scale = d_global_scale != nullptr ? __ldg(d_global_scale) : 1.0f;
        }
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;
    const float global_scale = s_global_scale;

    const uint4* weight_pack = reinterpret_cast<const uint4*>(weight + col_offset);
    uint4 w0 = weight_pack[0];
    uint4 w1 = weight_pack[1];
    const uint32_t weight_bits[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};

    float block_amax = 0.0f;
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        float2 w2 = __bfloat1622float2(bit_cast_u32<__nv_bfloat162>(weight_bits[i]));
        float y0 = vals[2 * i] * inv_rms * w2.x;
        float y1 = vals[2 * i + 1] * inv_rms * w2.y;
        vals[2 * i] = y0;
        vals[2 * i + 1] = y1;
        block_amax = fmaxf(block_amax, fabsf(y0));
        block_amax = fmaxf(block_amax, fabsf(y1));
    }

    float scale = global_scale * (block_amax * kInvFp4Max);
    uint8_t scale_bits;
    float quant_scale;
    if constexpr (BLOCK_SIZE == 16) {
        scale_bits = static_cast<uint8_t>(
            __nv_cvt_float_to_fp8(scale, __NV_SATFINITE, __NV_E4M3));
        quant_scale = e4m3_to_float(scale_bits);
    } else {
        const uint32_t rounded_bits =
            (bit_cast_f32(scale) + 0x007FFFFFu) & 0x7F800000u;
        scale_bits = static_cast<uint8_t>(rounded_bits >> 23);
        quant_scale = bit_cast_u32<float>(rounded_bits);
    }
    out_scales[row * num_blocks + block_idx] = scale_bits;

    const float output_scale =
        quant_scale > 0.0f ? (global_scale / quant_scale) : 0.0f;

    uint8_t* out_row = out_fp4 + row * (hidden / 2) + block_idx * kPairsPerThread;
    uint64_t packed_fp4 = 0;
#pragma unroll
    for (int i = 0; i < kPairsPerThread; ++i) {
        float2 q = make_float2(vals[2 * i] * output_scale, vals[2 * i + 1] * output_scale);
        packed_fp4 |= static_cast<uint64_t>(
            static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(q, __NV_E2M1, cudaRoundNearest)))
            << (i * 8);
    }
    *reinterpret_cast<uint64_t*>(out_row) = packed_fp4;
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
    TORCH_CHECK(residual.is_cuda() && residual.dtype() == torch::kBFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16);
    TORCH_CHECK(hidden % block_size == 0);
    TORCH_CHECK(block_size == 16 || block_size == 32);

    const float* d_gs = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    auto scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));

    int num_blocks_per_row = hidden / (int)block_size;
    int block_threads = 1;
    while (block_threads < num_blocks_per_row) block_threads <<= 1;
    TORCH_CHECK(block_threads <= 1024);
    TORCH_CHECK(block_threads == num_blocks_per_row,
                "hidden/block_size must be a power of two for this kernel");
    TORCH_CHECK(hidden == 4096,
                "optimized kernel expects hidden size 4096");
    TORCH_CHECK(block_size == 16,
                "optimized kernel expects block size 16");

    fused_add_rmsnorm_nvfp4_kernel<16><<<batch, block_threads>>>(
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
