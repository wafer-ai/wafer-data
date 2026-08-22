import os
import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline

os.environ["CXX"] = "hipcc"

# Very simple fused kernel: linear + dropout + softmax

src = """
#include <hip/hip_runtime.h>

__global__ void kernel(float* __restrict__ out, const float* in, const float* w, 
                        const float* b, int bs, int inf, int outf, float dp, int seed) {
    int bx = blockIdx.x;
    if (bx >= bs) return;
    
    int tx = threadIdx.x;
    int n = blockDim.x;
    
    int start = (outf * tx) / n;
    int end = (outf * (tx + 1)) / n;
    
    __shared__ float sm[1024];
    
    // Pass 1: matmul + dropout, find max
    float mx = -1e30;
    const float* ip = in + bx * inf;
    
    for (int j = start; j < end; j++) {
        float sum = b[j];
        const float* wp = w + j * inf;
        for (int i = 0; i < inf; i++) sum += wp[i] * ip[i];
        
        unsigned int h = seed + (unsigned int)(bx * outf + j);
        float r = (float)h / (float)0xffffffffu;
        if (r < dp) sum = 0.0f; else sum /= (1.0f - dp);
        
        out[bx * outf + j] = sum;
        mx = fmaxf(mx, sum);
    }
    
    sm[tx] = mx;
    __syncthreads();
    for (int s = n/2; s; s>>=1) {
        if (tx < s) sm[tx] = fmaxf(sm[tx], sm[tx+s]);
        __syncthreads();
    }
    mx = sm[0];
    
    // Pass 2: exp and sum
    float tot = 0.0f;
    for (int j = start; j < end; j++) {
        float v = expf(out[bx * outf + j] - mx);
        out[bx * outf + j] = v;
        tot += v;
    }
    
    sm[tx] = tot;
    __syncthreads();
    for (int s = n/2; s; s>>=1) {
        if (tx < s) sm[tx] += sm[tx+s];
        __syncthreads();
    }
    tot = sm[0];
    
    // Pass 3: normalize
    for (int j = start; j < end; j++) {
        out[bx * outf + j] /= tot;
    }
}

torch::Tensor myfunc(torch::Tensor a, torch::Tensor w, torch::Tensor b, float dp) {
    int bs = a.size(0);
    int inf = a.size(1);
    int outf = w.size(0);
    auto out = torch::zeros({bs, outf}, a.options());
    
    int th = min(1024, max(256, outf));
    kernel<<<dim3(bs), dim3(th>>>(out.data_ptr<float>(), a.data_ptr<float>(), 
                                   w.data_ptr<float>(), b.data_ptr<float>(),
                                   bs, inf, outf, dp, 12345);
    return out;
}
"""

fused_kernel = load_inline(name="msds", cpp_sources=src, functions=["myfunc"], verbose=True)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(ModelNew, self).__init__()
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.dropout_p = dropout_p
        
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
        self.fused_op = fused_kernel
    
    def forward(self, x):
        return self.fused_op.myfunc(x, self.weight, self.bias, self.dropout_p)