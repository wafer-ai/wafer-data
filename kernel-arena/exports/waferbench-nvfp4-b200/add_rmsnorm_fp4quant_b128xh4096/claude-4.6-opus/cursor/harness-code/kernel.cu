#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

constexpr int E4M3_BIAS = 7;

__device__ __forceinline__ uint8_t float_to_e4m3_fast(float val) {
    if (val <= 0.0f) return 0;
    if (val > 448.0f) val = 448.0f;
    unsigned int bits = __float_as_uint(val);
    int fp32_exp = ((bits >> 23) & 0xFF) - 127;
    unsigned int fp32_man = bits & 0x7FFFFF;

    if (fp32_exp < -9) return 0;

    if (fp32_exp < -6) {
        int shift = -6 - fp32_exp;
        unsigned int subnorm_man = (0x800000 | fp32_man) >> (20 + shift);
        unsigned int round_bit = ((0x800000 | fp32_man) >> (19 + shift)) & 1;
        subnorm_man += round_bit;
        if (subnorm_man > 7) subnorm_man = 7;
        return (uint8_t)subnorm_man;
    }

    int biased = fp32_exp + E4M3_BIAS;
    unsigned int man3 = (fp32_man + 0x80000) >> 20;
    if (man3 > 7) { man3 = 0; biased++; }
    if (biased > 15) { biased = 15; man3 = 6; }
    if (biased < 1) biased = 1;
    return (uint8_t)((biased << 3) | man3);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t bits) {
    int biased_exp = (bits >> 3) & 0xF;
    int man = bits & 0x7;
    if (biased_exp == 0)
        return (float(man) / 8.0f) * (1.0f / 64.0f);
    int fp32_exp = biased_exp - E4M3_BIAS + 127;
    unsigned int fp32_bits = (fp32_exp << 23) | (man << 20);
    return __uint_as_float(fp32_bits);
}

__device__ __forceinline__ uint8_t float_to_fp4_nibble(float val) {
    unsigned int bits = __float_as_uint(val);
    unsigned int sign = bits >> 31;
    bits &= 0x7FFFFFFF;

    // Branchless FP4 E2M1 encoding: 7 comparisons on unsigned int (same order as float for positives)
    unsigned int enc;
    enc  = (bits > 0x3E800000u);  // > 0.25
    enc += (bits >= 0x3F400000u); // >= 0.75
    enc += (bits > 0x3FA00000u);  // > 1.25
    enc += (bits >= 0x3FE00000u); // >= 1.75
    enc += (bits > 0x40200000u);  // > 2.5
    enc += (bits >= 0x40600000u); // >= 3.5
    enc += (bits > 0x40A00000u);  // > 5.0
    return (uint8_t)((sign << 3) | enc);
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

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val, 8);
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}

// Primary kernel: 1 thread per quantization block.
// NTHREADS = num_blocks_per_row = hidden / BLOCK_SIZE
template <int BLOCK_SIZE, int NTHREADS>
__global__ void __launch_bounds__(NTHREADS, 2)
fused_add_rmsnorm_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ d_global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden
) {
    constexpr int NWARPS = NTHREADS / 32;
    int tid = threadIdx.x;
    int row = blockIdx.x;

    __shared__ float s_warp_sums[NWARPS];

    float global_scale = d_global_scale ? __ldg(d_global_scale) : 1.0f;

    int num_blocks_per_row = hidden / BLOCK_SIZE;
    int row_offset = row * hidden;
    int base_col = tid * BLOCK_SIZE;

    // Phase 1: Add input+residual, write back residual, accumulate sq sum
    float regs[BLOCK_SIZE];
    float local_ss = 0.0f;

    {
        const uint4* in_v  = reinterpret_cast<const uint4*>(input + row_offset + base_col);
        uint4*       res_v = reinterpret_cast<uint4*>(residual + row_offset + base_col);

        #pragma unroll
        for (int v = 0; v < BLOCK_SIZE / 8; v++) {
            uint4 iv = __ldg(in_v + v);
            uint4 rv = res_v[v];

            nv_bfloat162* ip = reinterpret_cast<nv_bfloat162*>(&iv);
            nv_bfloat162* rp = reinterpret_cast<nv_bfloat162*>(&rv);

            // Use bf16 add then store, but keep float for precision in sq accumulation
            nv_bfloat162 sum0 = __hadd2(ip[0], rp[0]);
            nv_bfloat162 sum1 = __hadd2(ip[1], rp[1]);
            nv_bfloat162 sum2 = __hadd2(ip[2], rp[2]);
            nv_bfloat162 sum3 = __hadd2(ip[3], rp[3]);

            uint4 out_v;
            nv_bfloat162* op = reinterpret_cast<nv_bfloat162*>(&out_v);
            op[0] = sum0; op[1] = sum1; op[2] = sum2; op[3] = sum3;
            res_v[v] = out_v;

            float f0 = __bfloat162float(sum0.x);
            float f1 = __bfloat162float(sum0.y);
            float f2 = __bfloat162float(sum1.x);
            float f3 = __bfloat162float(sum1.y);
            float f4 = __bfloat162float(sum2.x);
            float f5 = __bfloat162float(sum2.y);
            float f6 = __bfloat162float(sum3.x);
            float f7 = __bfloat162float(sum3.y);

            regs[v*8+0] = f0; regs[v*8+1] = f1;
            regs[v*8+2] = f2; regs[v*8+3] = f3;
            regs[v*8+4] = f4; regs[v*8+5] = f5;
            regs[v*8+6] = f6; regs[v*8+7] = f7;

            local_ss += f0*f0 + f1*f1 + f2*f2 + f3*f3 + f4*f4 + f5*f5 + f6*f6 + f7*f7;
        }
    }

    // Phase 2: Warp + block reduction for sum of squares
    float warp_sum = warp_reduce_sum(local_ss);
    int warp_id = tid >> 5;
    int lane_id = tid & 31;

    if (lane_id == 0) s_warp_sums[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        float val = (tid < NWARPS) ? s_warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) s_warp_sums[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(s_warp_sums[0] / float(hidden) + eps);

    // Phase 3: Normalize, find max, quantize
    const uint4* w_v = reinterpret_cast<const uint4*>(weight + base_col);
    float max_abs = 0.0f;

    #pragma unroll
    for (int v = 0; v < BLOCK_SIZE / 8; v++) {
        uint4 wv = __ldg(w_v + v);
        nv_bfloat162* wp = reinterpret_cast<nv_bfloat162*>(&wv);

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            int idx = v * 8 + k * 2;
            float n0 = regs[idx]     * rms_inv * __bfloat162float(wp[k].x);
            float n1 = regs[idx + 1] * rms_inv * __bfloat162float(wp[k].y);
            regs[idx]     = n0;
            regs[idx + 1] = n1;
            max_abs = fmaxf(max_abs, fmaxf(fabsf(n0), fabsf(n1)));
        }
    }

    // Compute block scale
    float gs_over_6 = global_scale * 0.16666667f;
    uint8_t scale_e4m3 = float_to_e4m3_fast(gs_over_6 * max_abs);
    out_scales[row * num_blocks_per_row + tid] = scale_e4m3;

    float scale_f = e4m3_to_float(scale_e4m3);
    float inv_scale = (scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;

    // Phase 4: Quantize to FP4 and pack
    int fp4_base = row * (hidden / 2) + tid * (BLOCK_SIZE / 2);

    uint64_t packed64 = 0;
    #pragma unroll
    for (int j = 0; j < BLOCK_SIZE; j += 2) {
        float v0 = regs[j]     * inv_scale;
        float v1 = regs[j + 1] * inv_scale;
        uint8_t byte_val = float_to_fp4_nibble(v0) | (float_to_fp4_nibble(v1) << 4);
        packed64 |= ((uint64_t)byte_val) << (j / 2 * 8);
    }
    *reinterpret_cast<uint64_t*>(out_fp4 + fp4_base) = packed64;
}

// Large hidden kernel: uses shared memory for intermediate h values
template <int BLOCK_SIZE>
__global__ void fused_add_rmsnorm_nvfp4_kernel_large(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ d_global_scale, float eps,
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ out_scales,
    int batch, int hidden
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    float* s_h = reinterpret_cast<float*>(smem_raw);
    float* s_warp_sums = s_h + hidden;

    int num_blocks_per_row = hidden / BLOCK_SIZE;
    int row_offset = row * hidden;

    float global_scale = d_global_scale ? __ldg(d_global_scale) : 1.0f;
    float gs_over_6 = global_scale * 0.16666667f;

    float local_ss = 0.0f;

    for (int col = tid * 8; col < hidden; col += num_threads * 8) {
        uint4 iv = __ldg(reinterpret_cast<const uint4*>(input + row_offset + col));
        uint4 rv = *reinterpret_cast<const uint4*>(residual + row_offset + col);
        nv_bfloat162* ip = reinterpret_cast<nv_bfloat162*>(&iv);
        nv_bfloat162* rp = reinterpret_cast<nv_bfloat162*>(&rv);

        nv_bfloat162 sum0 = __hadd2(ip[0], rp[0]);
        nv_bfloat162 sum1 = __hadd2(ip[1], rp[1]);
        nv_bfloat162 sum2 = __hadd2(ip[2], rp[2]);
        nv_bfloat162 sum3 = __hadd2(ip[3], rp[3]);

        uint4 out_v;
        nv_bfloat162* op = reinterpret_cast<nv_bfloat162*>(&out_v);
        op[0] = sum0; op[1] = sum1; op[2] = sum2; op[3] = sum3;
        *reinterpret_cast<uint4*>(residual + row_offset + col) = out_v;

        #pragma unroll
        for (int k = 0; k < 4; k++) {
            nv_bfloat162 s = (k==0) ? sum0 : (k==1) ? sum1 : (k==2) ? sum2 : sum3;
            float f0 = __bfloat162float(s.x);
            float f1 = __bfloat162float(s.y);
            s_h[col + k*2]     = f0;
            s_h[col + k*2 + 1] = f1;
            local_ss += f0*f0 + f1*f1;
        }
    }

    float warp_sum = warp_reduce_sum(local_ss);
    int warp_id = tid >> 5;
    int lane_id = tid & 31;
    int num_warps = num_threads >> 5;

    if (lane_id == 0) s_warp_sums[warp_id] = warp_sum;
    __syncthreads();

    if (tid < 32) {
        float val = (tid < num_warps) ? s_warp_sums[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) s_warp_sums[0] = val;
    }
    __syncthreads();

    float rms_inv = rsqrtf(s_warp_sums[0] / float(hidden) + eps);

    for (int blk_idx = tid; blk_idx < num_blocks_per_row; blk_idx += num_threads) {
        int base_col = blk_idx * BLOCK_SIZE;
        float max_abs = 0.0f;
        float normed_vals[32];

        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j += 2) {
            nv_bfloat162 wv = *reinterpret_cast<const nv_bfloat162*>(weight + base_col + j);
            float n0 = s_h[base_col + j]     * rms_inv * __bfloat162float(wv.x);
            float n1 = s_h[base_col + j + 1] * rms_inv * __bfloat162float(wv.y);
            normed_vals[j]     = n0;
            normed_vals[j + 1] = n1;
            max_abs = fmaxf(max_abs, fmaxf(fabsf(n0), fabsf(n1)));
        }

        uint8_t scale_e4m3 = float_to_e4m3_fast(gs_over_6 * max_abs);
        out_scales[row * num_blocks_per_row + blk_idx] = scale_e4m3;

        float scale_f = e4m3_to_float(scale_e4m3);
        float inv_scale = (scale_f != 0.0f) ? (global_scale / scale_f) : 0.0f;

        int fp4_base = row * (hidden / 2) + blk_idx * (BLOCK_SIZE / 2);
        #pragma unroll
        for (int j = 0; j < BLOCK_SIZE; j += 2) {
            float v0 = fminf(fmaxf(normed_vals[j]     * inv_scale, -6.0f), 6.0f);
            float v1 = fminf(fmaxf(normed_vals[j + 1] * inv_scale, -6.0f), 6.0f);
            out_fp4[fp4_base + j / 2] = float_to_fp4_nibble(v0) | (float_to_fp4_nibble(v1) << 4);
        }
    }
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

    if (num_blocks_per_row <= 1024) {
        if (block_size == 16 && num_blocks_per_row == 256) {
            fused_add_rmsnorm_nvfp4_kernel<16, 256><<<batch, 256>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
                batch, hidden);
        } else if (block_size == 16 && num_blocks_per_row == 512) {
            fused_add_rmsnorm_nvfp4_kernel<16, 512><<<batch, 512>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
                batch, hidden);
        } else {
            int threads = num_blocks_per_row;
            int smem = hidden * sizeof(float) + ((threads + 31) / 32) * sizeof(float);
            if (block_size == 16) {
                fused_add_rmsnorm_nvfp4_kernel_large<16><<<batch, threads, smem>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                    d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
                    batch, hidden);
            } else {
                fused_add_rmsnorm_nvfp4_kernel_large<32><<<batch, threads, smem>>>(
                    reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                    reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                    d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
                    batch, hidden);
            }
        }
    } else {
        int smem = hidden * sizeof(float) + 32 * sizeof(float);
        if (block_size == 16) {
            fused_add_rmsnorm_nvfp4_kernel_large<16><<<batch, 1024, smem>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
                batch, hidden);
        } else {
            fused_add_rmsnorm_nvfp4_kernel_large<32><<<batch, 1024, smem>>>(
                reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                reinterpret_cast<__nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
                reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<at::BFloat16>()),
                d_gs, (float)eps, fp4_raw.data_ptr<uint8_t>(), scales_raw.data_ptr<uint8_t>(),
                batch, hidden);
        }
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
