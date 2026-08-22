#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdint>
#include <cmath>

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

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

template<int BLOCK_SIZE>
__device__ __forceinline__ void pack_fp4_block(
    const float* __restrict__ vals,
    float quant_mul,
    uint8_t* __restrict__ out_ptr
) {
    constexpr int PACKED_BYTES = BLOCK_SIZE / 2;
    constexpr int PACKS64 = PACKED_BYTES / 8;
    uint64_t* out64 = reinterpret_cast<uint64_t*>(out_ptr);

    #pragma unroll
    for (int pack_idx = 0; pack_idx < PACKS64; ++pack_idx) {
        uint64_t packed64 = 0;
        #pragma unroll
        for (int byte_idx = 0; byte_idx < 8; ++byte_idx) {
            const int elem_idx = pack_idx * 16 + byte_idx * 2;
            const uint8_t packed8 = static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(
                make_float2(vals[elem_idx] * quant_mul, vals[elem_idx + 1] * quant_mul),
                __NV_E2M1, cudaRoundNearest));
            packed64 |= static_cast<uint64_t>(packed8) << (byte_idx * 8);
        }
        out64[pack_idx] = packed64;
    }
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

template<int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ d_global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    static_assert(BLOCK_SIZE == 16 || BLOCK_SIZE == 32);
    constexpr int PAIRS_PER_THREAD = BLOCK_SIZE / 2;
    __shared__ float warp_sums[32];
    __shared__ float inv_rms_shmem;
    __shared__ float global_scale_shmem;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_blocks_per_row = hidden / BLOCK_SIZE;
    const bool active = tid < num_blocks_per_row;

    if (tid == 0) {
        global_scale_shmem = d_global_scale != nullptr ? __ldg(d_global_scale) : 1.0f;
    }
    __syncthreads();

    float sumsq = 0.0f;
    __nv_bfloat162 h_pairs[PAIRS_PER_THREAD];
    if (active) {
        const int row_offset = row * hidden + tid * BLOCK_SIZE;
        const __nv_bfloat162* input2 =
            reinterpret_cast<const __nv_bfloat162*>(input + row_offset);
        __nv_bfloat162* residual2 =
            reinterpret_cast<__nv_bfloat162*>(residual + row_offset);

        #pragma unroll
        for (int i = 0; i < PAIRS_PER_THREAD; ++i) {
            const float2 in_f = __bfloat1622float2(input2[i]);
            const float2 res_f = __bfloat1622float2(residual2[i]);
            const float2 h_f = make_float2(in_f.x + res_f.x, in_f.y + res_f.y);
            sumsq = fmaf(h_f.x, h_f.x, sumsq);
            sumsq = fmaf(h_f.y, h_f.y, sumsq);
            const __nv_bfloat162 h_bf16 = __float22bfloat162_rn(h_f);
            h_pairs[i] = h_bf16;
            residual2[i] = h_bf16;
        }
    }

    sumsq = warp_sum(sumsq);
    if (lane == 0) {
        warp_sums[warp_id] = sumsq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float block_sum = lane < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            inv_rms_shmem = rsqrtf(block_sum / static_cast<float>(hidden) + eps);
        }
    }
    __syncthreads();

    if (!active) {
        return;
    }

    const float inv_rms = inv_rms_shmem;
    const float global_scale = global_scale_shmem;
    const int fp4_row_offset = row * (hidden / 2) + tid * (BLOCK_SIZE / 2);
    const int scale_offset = row * num_blocks_per_row + tid;
    const __nv_bfloat162* weight2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + tid * BLOCK_SIZE);

    float vals[BLOCK_SIZE];
    float max_abs = 0.0f;
    #pragma unroll
    for (int i = 0; i < PAIRS_PER_THREAD; ++i) {
        const float2 h_f = __bfloat1622float2(h_pairs[i]);
        const float2 w_f = __bfloat1622float2(weight2[i]);
        const float v0 = (h_f.x * inv_rms) * w_f.x;
        const float v1 = (h_f.y * inv_rms) * w_f.y;
        vals[i * 2] = v0;
        vals[i * 2 + 1] = v1;
        max_abs = fmaxf(max_abs, fabsf(v0));
        max_abs = fmaxf(max_abs, fabsf(v1));
    }

    const float raw_scale = global_scale * (max_abs * (1.0f / FP4_MAX));
    const __nv_fp8_e4m3 scale_fp8(raw_scale);
    const uint8_t scale_bits = scale_fp8.__x;
    out_scales[scale_offset] = scale_bits;

    const float quant_scale = static_cast<float>(scale_fp8);
    const float quant_mul = quant_scale > 0.0f ? (global_scale / quant_scale) : 0.0f;
    pack_fp4_block<BLOCK_SIZE>(vals, quant_mul, out_fp4 + fp4_row_offset);
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
    TORCH_CHECK(residual.is_cuda() && residual.dtype() == torch::kBFloat16);
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kBFloat16);
    int batch = input.size(0), hidden = input.size(1);
    TORCH_CHECK(hidden % block_size == 0);
    TORCH_CHECK(block_size == 16 || block_size == 32,
                "Only block sizes 16 and 32 are supported");

    const float* d_gs = global_scale.has_value() ? global_scale->data_ptr<float>() : nullptr;

    auto fp4_raw = torch::empty({batch, hidden / 2}, input.options().dtype(torch::kUInt8));
    auto scales_raw = torch::empty({batch, hidden / (int)block_size}, input.options().dtype(torch::kUInt8));

    int num_blocks_per_row = hidden / (int)block_size;
    int block_threads = 1;
    while (block_threads < num_blocks_per_row) block_threads <<= 1;
    if (block_threads < 32) block_threads = 32;
    TORCH_CHECK(block_threads <= 1024);
    if (block_size == 16) {
        fused_add_rmsnorm_nvfp4_kernel<16><<<batch, block_threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
            batch, hidden, (int)block_size);
    } else {
        fused_add_rmsnorm_nvfp4_kernel<32><<<batch, block_threads>>>(
            reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
            reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
            reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
            d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
            batch, hidden, (int)block_size);
    }

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
