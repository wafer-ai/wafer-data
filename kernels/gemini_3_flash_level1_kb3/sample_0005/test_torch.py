
import torch
import torch.nn as nn
import time

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A, B):
        return torch.matmul(A, B)

M = 32768
N = 32

def get_inputs():
    A = torch.rand(M, N).cuda()
    B = torch.rand(N, M).cuda()
    return [A, B]

model = Model().cuda()
A, B = get_inputs()

# Warmup
for _ in range(10):
    C = model(A, B)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    C = model(A, B)
torch.cuda.synchronize()
end = time.time()
print(f"Time: {(end - start) / 100 * 1000:.3f} ms")
