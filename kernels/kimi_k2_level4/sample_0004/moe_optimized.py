import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
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
        
        self.gate_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.up_proj = nn.Parameter(torch.randn(num_experts, intermediate_size, hidden_size) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(num_experts, hidden_size, intermediate_size) * 0.02)

    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        
        x_flat = x.view(-1, self.hidden_size)
        num_tokens = x_flat.shape[0]
        output = torch.zeros(num_tokens, self.hidden_size, device=x.device, dtype=x.dtype)
        
        # Process all experts in parallel using vectorized operations
        for expert_idx in range(self.num_experts):
            mask = (expert_indices == expert_idx)
            token_count = mask.sum().item()
            
            if token_count == 0:
                continue
            
            # Get indices for this expert
            batch_idx, seq_idx, slot_idx = torch.where(mask)
            token_indices = batch_idx * seq_len + seq_idx
            weights = expert_weights[batch_idx, seq_idx, slot_idx]
            expert_input = x_flat[token_indices]
            
            # Fused operations with minimal memory overhead
            # gate projection + SiLU + up projection in one step
            gate_val = F.linear(expert_input, self.gate_proj[expert_idx])
            up_val = F.linear(expert_input, self.up_proj[expert_idx])
            
            # Compute SiLU: gate * sigmoid(gate) * up in-place
            intermediate = gate_val * F.sigmoid(gate_val) * up_val
            
            # Down projection
            expert_output = F.linear(intermediate, self.down_proj[expert_idx])
            
            # Accumulate with expert weights applied
            output.index_add_(0, token_indices, expert_output * weights.unsqueeze(-1))
        
        return output.view(batch, seq_len, self.hidden_size)

batch_size = 4
seq_len = 2048
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2

def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    expert_indices = torch.stack([torch.randperm(num_experts)[:top_k] for _ in range(batch_size * seq_len)]).view(batch_size, seq_len, top_k)
    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)
    return [x, expert_indices, expert_weights]

def get_init_inputs():
    return [hidden_size, intermediate_size, num_experts]
