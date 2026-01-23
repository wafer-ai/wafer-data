import torch
import torch.nn as nn

# FP8 Matrix Multiplication - Modified for MI300X compatibility
# Performs mathematically equivalent operation using FP16 compute

class Model(nn.Module):
    """
    FP8-style Matrix Multiplication adapted for MI300X.
    
    Since MI300X doesn't support torch._scaled_mm with FP8 types,
    this implements the mathematically equivalent operation:
    - Compute per-tensor scales
    - Quantize to simulated FP8 range
    - Perform matmul with proper dequantization
    """

    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3

        # FP8 format specifications
        if use_e4m3:
            self.fp8_max = 448.0  # Max representable value in E4M3
        else:
            self.fp8_max = 57344.0  # Max representable value in E5M2

        # Weight matrix stored in FP16
        self.weight = nn.Parameter(torch.randn(K, N, dtype=torch.float16) * 0.02)

    def compute_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-tensor scale for FP8 quantization."""
        amax = x.abs().max()
        scale = self.fp8_max / amax.clamp(min=1e-12)
        return scale

    def quantize_and_dequantize(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Simulate FP8 quantization by clamping and scaling."""
        x_scaled = x.float() * scale.float()
        x_clamped = x_scaled.clamp(-self.fp8_max, self.fp8_max)
        # Simulate FP8 precision loss by rounding
        x_rounded = torch.round(x_clamped)
        return (x_rounded / scale.float()).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FP8-equivalent matmul: x @ weight

        Input x: (batch, seq_len, K) in FP16
        Weight: (K, N) in FP16
        Output: (batch, seq_len, N) in FP16
        """
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Reshape for matmul: (batch, seq, K) -> (batch*seq, K)
        x_2d = x.view(-1, self.K)

        # Compute scales
        x_scale = self.compute_scale(x_2d)
        w_scale = self.compute_scale(self.weight)

        # Quantize and dequantize (simulates FP8 precision)
        x_q = self.quantize_and_dequantize(x_2d, x_scale)
        w_q = self.quantize_and_dequantize(self.weight, w_scale)

        # Standard matmul 
        out = torch.matmul(x_q, w_q)

        return out.view(batch_size, seq_len, self.N)


# Configuration
batch_size = 8
seq_len = 2048
M = batch_size * seq_len
K = 4096
N = 4096
use_e4m3 = True


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [M, K, N, use_e4m3]
