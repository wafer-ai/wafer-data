import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

custom_cpp = """
#include <hip/hip_runtime.h>

__global__ void conv3d_kernel(const float *input, const float *weight, const float *bias, float *output,
                              int N, int Cin, int Din, int Hin, int Win,
                              int Cout, int kd, int kh, int kw,
                              int Dout, int Hout, int Wout) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = 1LL * N * Cout * Dout * Hout * Wout;
    if (idx >= total) return;
    int b = idx / (Cout * Dout * Hout * Wout);
    int c = (idx / (Dout * Hout * Wout)) % Cout;
    int z = (idx / (Hout * Wout)) % Dout;
    int y = (idx / Wout) % Hout;
    int x = idx % Wout;
    float accum = bias[c];
    for (int ci = 0; ci < Cin; ci++) {
        for (int kz = 0; kz < kd; kz++) {
            int zi = z + kz;
            if (zi >= Din || zi < 0) continue;
            for (int ky = 0; ky < kh; ky++) {
                int yi = y + ky;
                if (yi >= Hin || yi < 0) continue;
                for (int kx = 0; kx < kw; kx++) {
                    int xi = x + kx;
                    if (xi >= Win || xi < 0) continue;
                    long long iidx = ((1LL * b * Cin + ci) * Din + zi) * Hin + yi;
                    iidx = iidx * Win + xi;
                    float ival = input[iidx];
                    long long widx = (((1LL * c * Cin + ci) * kd + kz) * kh + ky) * kw + kx;
                    float wval = weight[widx];
                    accum += ival * wval;
                }
            }
        }
    }
    long long oidx = ((1LL * b * Cout + c) * Dout + z) * Hout + y;
    oidx = oidx * Wout + x;
    output[oidx] = accum;
}

torch::Tensor conv3d_hip(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    torch::IntArrayRef ish = input.sizes();
    int64_t N = ish[0], Cin = ish[1], Din = ish[2], Hin = ish[3], Win = ish[4];
    torch::IntArrayRef wsh = weight.sizes();
    int64_t Cout = wsh[0];
    int64_t kd = wsh[2], kh = wsh[3], kw = wsh[4];
    int64_t Dout = Din - kd + 1;
    int64_t Hout = Hin - kh + 1;
    int64_t Wout = Win - kw + 1;
    int64_t out_shape_arr[5] = {N, Cout, Dout, Hout, Wout};
    auto out_shape = torch::IntArrayRef(out_shape_arr, 5);
    auto out = torch::empty(out_shape, input.options());
    const float* bptr = bias.data_ptr<float>();
    const int threads = 1024;
    int64_t num_elem = N * Cout * Dout * Hout * Wout;
    int blocks = (num_elem + threads - 1) / threads;
    conv3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), weight.data_ptr<float>(), bptr, out.data_ptr<float>(),
        (int)N, (int)Cin, (int)Din, (int)Hin, (int)Win,
        (int)Cout, (int)kd, (int)kh, (int)kw,
        (int)Dout, (int)Hout, (int)Wout);
    return out;
}

__global__ void softmax_channel_kernel(const float *input, float *output, int N, int C, int D, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total_pos = 1LL * N * D * H * W;
    if (idx >= total_pos) return;
    int b = idx / (D * H * W);
    int rem = idx % (D * H * W);
    int z = rem / (H * W);
    int y = (rem / W) % H;
    int x = rem % W;
    float maxv = -1e30f;
    for(int c = 0; c < C; c++) {
        long long ii = ((1LL*b * C + c) * D + z) * H + y;
        ii = ii * W + x;
        float v = input[ii];
        if (v > maxv) maxv = v;
    }
    float sum_exp = 0.0f;
    for(int c = 0; c < C; c++) {
        long long ii = ((1LL*b * C + c) * D + z) * H + y;
        ii = ii * W + x;
        float v = input[ii];
        sum_exp += __expf(v - maxv);
    }
    for(int c = 0; c < C; c++) {
        long long ii = ((1LL*b * C + c) * D + z) * H + y;
        ii = ii * W + x;
        float v = input[ii];
        output[ii] = __expf(v - maxv) / sum_exp;
    }
}

torch::Tensor softmax_channel_hip(torch::Tensor input) {
    torch::IntArrayRef shape = input.sizes();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t D = shape[2];
    int64_t H = shape[3];
    int64_t W = shape[4];
    auto out = torch::empty_like(input);
    const int threads = 256;
    int64_t num_pos = N * D * H * W;
    int blocks = (num_pos + threads - 1) / threads;
    softmax_channel_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), out.data_ptr<float>(), (int)N, (int)C, (int)D, (int)H, (int)W);
    return out;
}

__global__ void maxpool3d_kernel(const float *input, float *output,
                                 int N, int C, int D_in, int H_in, int W_in,
                                 int D_out, int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long total = 1LL * N * C * D_out * H_out * W_out;
    if(idx >= total) return;
    int b = idx / (C * D_out * H_out * W_out);
    int c = (idx / (D_out * H_out * W_out)) % C;
    int z = (idx / (H_out * W_out)) % D_out;
    int y = (idx / W_out) % H_out;
    int x = idx % W_out;
    float mval = -1e30f;
    for(int kz=0; kz<2; kz++) {
        int zi = z * 2 + kz;
        if(zi >= D_in) continue;
        for(int ky=0; ky<2; ky++) {
            int yi = y * 2 + ky;
            if(yi >= H_in) continue;
            for(int kx=0; kx<2; kx++) {
                int xi = x * 2 + kx;
                if(xi >= W_in) continue;
                long long ii = ((1LL * b * C + c) * D_in + zi) * H_in + yi;
                ii = ii * W_in + xi;
                float val = input[ii];
                if(val > mval) mval = val;
            }
        }
    }
    long long oi = ((1LL * b * C + c) * D_out + z) * H_out + y;
    oi = oi * W_out + x;
    output[oi] = mval;
}

torch::Tensor maxpool3d_hip(torch::Tensor input) {
    torch::IntArrayRef shape = input.sizes();
    int64_t N = shape[0], C = shape[1], Din = shape[2], Hin = shape[3], Win = shape[4];
    int64_t Dout = Din / 2, Hout = Hin / 2, Wout = Win / 2;
    int64_t out_shape_arr[5] = {N, C, Dout, Hout, Wout};
    auto out_shape = torch::IntArrayRef(out_shape_arr, 5);
    auto out = torch::empty(out_shape, input.options());
    const int threads = 256;
    int64_t num_elem = N * C * Dout * Hout * Wout;
    int blocks = (num_elem + threads - 1) / threads;
    maxpool3d_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), out.data_ptr<float>(),
        (int)N, (int)C, (int)Din, (int)Hin, (int)Win, (int)Dout, (int)Hout, (int)Wout);
    return out;
}
"""

custom_ops = load_inline(
    name="custom_ops",
    cpp_sources=[custom_cpp],
    functions=["conv3d_hip", "softmax_channel_hip", "maxpool3d_hip"],
    verbose=True,
)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.custom_ops = custom_ops

    def forward(self, x):
        x = self.custom_ops.conv3d_hip(x, self.conv.weight, self.conv.bias)
        x = self.custom_ops.softmax_channel_hip(x)
        x = self.custom_ops.maxpool3d_hip(x)
        x = self.custom_ops.maxpool3d_hip(x)
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
