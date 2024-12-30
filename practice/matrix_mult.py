import triton
import triton.language as tl
import torch

@triton.jit
def mult_kernel(a_ptr, b_ptr, # pointers to a and b matrices
                output_ptr, # output pointer
                am_stride, ak_stride, # row and col strides for matrix a
                bk_stride, bn_stride, # row and col strides for matrix b
                block_size_m:tl.constexpr, block_size_k: tl.constexpr, block_size_n:tl.constexpr, # block sizes for a and b
                M,
                N,
                K,
                ):
    
    # create the grid. Remember: we are creating a 2d grid here so that is why we create "pseudo pids" for rows and cols. Essentially, 
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, block_size_m)
    grid_n = tl.cdiv(N, block_size_n)
    pid_m = pid / grid_n
    pid_n = pid % grid_n

    # calculate offsets
    offsets_am = (pid_m * block_size_m + tl.arange(0, block_size_m)) % M
    offsets_bn = (pid_n * block_size_n + tl.arange(0, block_size_n)) % N
    offsets_k = tl.arange(0, block_size_k)

    # determine the pointers for the a and b matrix
    a_pointers = a_ptr + (offsets_am[:, None] * am_stride + offsets_k[None, :] * ak_stride)
    b_pointers = b_ptr + (offsets_k[:, None] * bk_stride + offsets_bn[None, :] * bn_stride)

    a_pointers += block_size_k * ak_stride
    b_pointers += block_size_k * bk_stride

    # perform the dot product of the k values
    for k in range(0, tl.cdiv(K, block_size_k)):
        a = tl.load(a_pointers, mask=offsets_k[None, :] < K - k * block_size_k, other=0.0)
        b = tl.load(b_pointers, mask=offsets_k[:, None] < K - k * block_size_k, other=0.0)
    

    # perform dot product of the rows and cols in question
    # triton.dot(a_pointers, b_pointers)

    # what are the offsets for the output?


def matrix_mult(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k1 = a.shape
    k2, n = b.shape

    assert k1 == k2

    output_buffer = torch.empty_like((m, n))
    
    # set block sizes for a and b

    return output_buffer

if __name__ == "__main__":
    a = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float32, device='cuda')
    b = torch.tensor([[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]], dtype=torch.float32, device='cuda')
    print(f"Torch matrix multiplication: \n {torch.matmul(a, b)}")
    # import numpy as np

    # pid_m = 0
    # BLOCK_SIZE_M = 100
    # print (pid_m * BLOCK_SIZE_M + np.arange(0, BLOCK_SIZE_M) % 256)