import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelNew(nn.Module):
    """
    Optimized MoE Expert with Gated GEMM (SiLU-gated FFN).

    Optimization: Use torch.compile for graph-level optimization
    This allows PyTorch to:
    - Fuse compatible operations across the entire forward pass
    - Eliminate temporary tensor allocations
    - Optimize kernel launch sequence
    - Enable better memory bandwidth utilization
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        # Expert weights
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )

    @torch.compiler.compile()
    def expert_forward(
        self,
        expert_input: torch.Tensor,
        expert_idx: int,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Optimized computation for a single expert using compiled graph."""
        gate = F.silu(F.linear(expert_input, self.gate_proj[expert_idx]))
        up = F.linear(expert_input, self.up_proj[expert_idx])
        intermediate = gate * up
        expert_output = F.linear(intermediate, self.down_proj[expert_idx])
        return expert_output * weights.unsqueeze(-1)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]

        # Reshape for processing
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]

        # Initialize output accumulator
        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which (token, slot) pairs use this expert
            expert_mask = (expert_indices == expert_idx)

            if not expert_mask.any():
                continue

            # Get token indices and their routing weights
            batch_idx, seq_idx, slot_idx = torch.where(expert_mask)
            token_indices = batch_idx * seq_len + seq_idx
            weights = expert_weights[batch_idx, seq_idx, slot_idx]

            # Get tokens for this expert
            expert_input = x_flat[token_indices]

            # OPTIMIZED: Compute expert output using compiled function
            expert_output = self.expert_forward(expert_input, expert_idx, weights)

            # Accumulate output
            output.index_add_(0, token_indices, expert_output)

        return output.view(batch, seq_len, self.hidden_size)


(batch_size, seq_len, hidden_size, intermediate_size, num_experts, top_k) = (4, 2048, 4096, 14336, 8, 2)

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Random expert selection
    expert_indices = torch.stack([
        torch.randperm(num_experts)[:top_k]
        for _ in range(batch_size * seq_len)
    ]).view(batch_size, seq_len, top_k)

    # Random routing weights (normalized)
    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)

    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_experts]