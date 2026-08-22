#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr float FP4_MAX = 6.0f;
constexpr float E4M3_MAX = 448.0f;
constexpr int E4M3_BIAS = 7;

// Branchless FP4 E2M1 encoding using sorted threshold comparison
// Thresholds: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
// Note: boundaries alternate <= and < to match reference rounding
__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint32_t sign = 0;
    if (val < 0.0f) { sign = 8; val = -val; }
    // Use sequential comparisons but the compiler can optimize well with -use_fast_math
    uint32_t code = (val > 0.25f) + (val >= 0.75f) + (val > 1.25f) + (val >= 1.75f)
                  + (val > 2.5f) + (val >= 3.5f) + (val > 5.0f);
    return (uint8_t)(sign | code);
}

__device__ __forceinline__ uint8_t float_to_e4m3_fast(float val) {
    if (val <= 0.0f) return 0;
    if (val > E4M3_MAX) val = E4M3_MAX;
    int fp32_exp = ((__float_as_uint(val) >> 23) & 0xFF) - 127;
    if (fp32_exp < -9) return 0;
    if (fp32_exp < -6) {
        int man = __float2int_rn(val * 512.0f);
        if (man > 7) man = 7;
        if (man < 0) man = 0;
        return (uint8_t)man;
    }
    int biased_exp = fp32_exp + E4M3_BIAS;
    if (biased_exp < 1) biased_exp = 1;
    if (biased_exp > 15) biased_exp = 15;
    float pow2 = __uint_as_float(((unsigned int)(fp32_exp + 127)) << 23);
    int man = __float2int_rn((val / pow2 - 1.0f) * 8.0f);
    if (man > 7) { man = 0; biased_exp++; }
    if (man < 0) man = 0;
    if (biased_exp > 15) { biased_exp = 15; man = 6; }
    return (uint8_t)((biased_exp << 3) | man);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    int biased_exp = (bits >> 3) & 0xF;
    int man = bits & 0x7;
    if (biased_exp == 0) return (float(man) / 8.0f) * (1.0f / 64.0f);
    return (1.0f + float(man) / 8.0f) * __uint_as_float(((unsigned int)(biased_exp - E4M3_BIAS + 127)) << 23);
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

// One thread per element approach: 1024 threads per block
// Each 16-thread group cooperates on one quantization block
__global__ void fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ d_global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden, int block_size
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    const int row_offset = row * hidden;
    const __nv_bfloat16* inp_row = input + row_offset;
    __nv_bfloat16* res_row = residual + row_offset;
    const float gs = d_global_scale ? __ldg(d_global_scale) : 1.0f;

    // Phase 1: Each thread handles 2 elements (bf16x2)
    // hidden=2048, 1024 threads -> each handles 2 elements
    float h0 = 0.0f, h1 = 0.0f;
    float sq_sum = 0.0f;

    const int elem_idx = tid * 2;
    if (elem_idx < hidden) {
        __nv_bfloat162 iv = __ldg(reinterpret_cast<const __nv_bfloat162*>(inp_row + elem_idx));
        __nv_bfloat162 rv = *reinterpret_cast<const __nv_bfloat162*>(res_row + elem_idx);
        __nv_bfloat162 hv = __hadd2(iv, rv);
        *reinterpret_cast<__nv_bfloat162*>(res_row + elem_idx) = hv;
        h0 = __bfloat162float(hv.x);
        h1 = __bfloat162float(hv.y);
        sq_sum = h0 * h0 + h1 * h1;
    }

    // Phase 2: Reduce sum-of-squares across all threads
    const unsigned FULL_MASK = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sq_sum += __shfl_xor_sync(FULL_MASK, sq_sum, offset);

    __shared__ float smem[32];
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int num_warps = num_threads >> 5;

    if (lane_id == 0)
        smem[warp_id] = sq_sum;
    __syncthreads();

    if (tid < 32) {
        float val = (tid < num_warps) ? smem[tid] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_xor_sync(FULL_MASK, val, offset);
        smem[tid] = val;
    }
    __syncthreads();
    float total_sq = smem[0];

    const float rms_inv = rsqrtf(total_sq / (float)hidden + eps);

    // Phase 3: Normalize
    if (elem_idx >= hidden) return;

    const int blk_idx = elem_idx / block_size;
    const int within_blk = elem_idx % block_size;

    __nv_bfloat162 wv = __ldg(reinterpret_cast<const __nv_bfloat162*>(weight + elem_idx));
    float n0 = h0 * rms_inv * __bfloat162float(wv.x);
    float n1 = h1 * rms_inv * __bfloat162float(wv.y);

    // Phase 4: Compute block amax using warp-level reduction within block_size groups
    // With block_size=16, each quantization block = 8 threads (each handles 2 elements)
    float my_amax = fmaxf(fabsf(n0), fabsf(n1));

    // Reduce within 8 threads (half-warp) for block_size=16
    const int threads_per_qblock = block_size / 2;  // = 8
    #pragma unroll
    for (int offset = threads_per_qblock / 2; offset > 0; offset >>= 1)
        my_amax = fmaxf(my_amax, __shfl_xor_sync(FULL_MASK, my_amax, offset));

    // Thread with within_blk=0 has the block amax
    float block_amax = __shfl_sync(FULL_MASK, my_amax, (lane_id / threads_per_qblock) * threads_per_qblock);

    float scale_val = gs * block_amax * (1.0f / FP4_MAX);
    uint8_t scale_e4m3 = float_to_e4m3_fast(scale_val);
    float scale_dequant = e4m3_to_float(scale_e4m3);
    float inv_scale = (scale_dequant != 0.0f) ? gs / scale_dequant : 0.0f;

    // Only the first thread in each quantization block writes the scale
    int num_blocks_per_row = hidden / block_size;
    if (within_blk == 0)
        out_scales[row * num_blocks_per_row + blk_idx] = scale_e4m3;

    // Phase 5: Quantize and pack FP4
    float v0 = n0 * inv_scale;
    float v1 = n1 * inv_scale;
    uint8_t fp4_0 = float_to_fp4_e2m1(v0);
    uint8_t fp4_1 = float_to_fp4_e2m1(v1);
    uint8_t packed = (fp4_1 << 4) | fp4_0;

    out_fp4[row * (hidden / 2) + elem_idx / 2] = packed;
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

    int block_threads = hidden / 2;
    if (block_threads > 1024) block_threads = 1024;
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
