
import torch
import time

batch_size = 32768
in_features = 1024
out_features = 4096

x = torch.randn(batch_size, in_features).cuda()
w = torch.randn(in_features, out_features).cuda()
b = torch.randn(out_features).cuda()

# Warmup
for _ in range(10):
    y = torch.mm(x, w) + b

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    y = torch.mm(x, w) + b
torch.cuda.synchronize()
end = time.time()

print(f"Matmul time (no transpose): {(end - start) * 10:.3f} ms")
