
import torch
import torch.nn as nn
import time

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        N = A.size(0)
        M = N // 2
        
        A11 = A[:M, :M]
        A12 = A[:M, M:]
        A22 = A[M:, M:]
        
        B11 = B[:M, :M]
        B12 = B[:M, M:]
        B22 = B[M:, M:]
        
        # AB = [A11*B11,  A11*B12 + A12*B22]
        #      [0,        A22*B22          ]
        
        C = torch.empty_like(A)
        C[:M, :M] = torch.matmul(A11, B11)
        C[M:, M:] = torch.matmul(A22, B22)
        C[:M, M:] = torch.matmul(A11, B12) + torch.matmul(A12, B22)
        C[M:, :M] = 0
        
        return C

# Testing performance
N = 4096
A = torch.triu(torch.rand(N, N)).cuda()
B = torch.triu(torch.rand(N, N)).cuda()

# Benchmark Ref
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    ref = torch.triu(torch.matmul(A, B))
torch.cuda.synchronize()
print(f"Ref time: {(time.time() - start) / 10 * 1000:.3f}ms")

# Benchmark New
model = ModelNew()
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    new = model(A, B)
torch.cuda.synchronize()
print(f"New time: {(time.time() - start) / 10 * 1000:.3f}ms")

# Correctness
print(f"Correct: {torch.allclose(ref, new, atol=1e-3, rtol=1e-3)}")
