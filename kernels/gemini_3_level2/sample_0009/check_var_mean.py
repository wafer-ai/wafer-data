
import torch
x = torch.randn(10, 10)
v, m = torch.var_mean(x)
print(f"Var: {v}, Mean: {m}")
