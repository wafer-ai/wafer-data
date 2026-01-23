
import torch
import torch.nn as nn
import time

def split_matmul(A, B, k):
    N = A.size(0)
    m = N // k
    C = torch.zeros_like(A)
    for i in range(k):
        for j in range(i, k):
            # C[i, j] = sum_{l=i}^j A[i, l] * B[l, j]
            # (using block indices)
            res = None
            for l in range(i, j + 1):
                A_block = A[i*m:(i+1)*m, l*m:(l+1)*m]
                B_block = B[l*m:(l+1)*m, j*m:(j+1)*m]
                if res is None:
                    res = torch.matmul(A_block, B_block)
                else:
                    res += torch.matmul(A_block, B_block)
            C[i*m:(i+1)*m, j*m:(j+1)*m] = res
    return C

# Testing performance
N = 4096
A = torch.triu(torch.rand(N, N)).cuda()
B = torch.triu(torch.rand(N, N)).cuda()

torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    ref = torch.triu(torch.matmul(A, B))
torch.cuda.synchronize()
print(f"Ref time: {(time.time() - start) / 10 * 1000:.3f}ms")

for k in [2, 4, 8]:
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        new = split_matmul(A, B, k)
    torch.cuda.synchronize()
    print(f"k={k} time: {(time.time() - start) / 10 * 1000:.3f}ms")
    print(f"k={k} Correct: {torch.allclose(ref, new, atol=1e-3, rtol=1e-3)}")
