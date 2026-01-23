
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs matrix multiplication for upper triangular matrices
    by skipping known zero blocks and reducing the total number of operations.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.
        The algorithm splits the matrices into 4x4 blocks and only performs 
        multiplications that contribute to the non-zero elements of the 
        resulting upper triangular matrix.
        """
        k = 4
        N = A.size(0)
        m = N // k
        C = torch.zeros_like(A)
        
        # Pre-slicing blocks to minimize slicing overhead within loops
        A_blocks = [[A[i*m:(i+1)*m, j*m:(j+1)*m] for j in range(k)] for i in range(k)]
        B_blocks = [[B[i*m:(i+1)*m, j*m:(j+1)*m] for j in range(k)] for i in range(k)]
        
        for i in range(k):
            for j in range(i, k):
                # Only need to sum A[i, l] * B[l, j] for l from i to j
                # Initial block result
                res = torch.matmul(A_blocks[i][i], B_blocks[i][j])
                # Accumulate subsequent contributing block products
                for l in range(i + 1, j + 1):
                    res = torch.addmm(res, A_blocks[i][l], B_blocks[l][j])
                C[i*m:(i+1)*m, j*m:(j+1)*m] = res
                
        return C
