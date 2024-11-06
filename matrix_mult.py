import triton
import triton.language as tl
import torch

@triton.jit
def mult_kernel(output_ptr, # output pointer
                a_ptr, b_ptr, # pointers to a and b matrices
                am_stride, ak_stride, # row and col strides for matrix a
                bk_stride, bn_stride, # row and col strides for matrix b
                block_size_a:tl.constexpr, block_size_b:tl.constexpr, # block sizes for a and b
                M,
                N,
                K,
                ):
    
    # create the grid
    pid = tl.program_id(axis=0)
    grid_m = (M + block_size_a - 1) // block_size_a
    grid_n = (N + block_size_b - 1) // block_size_b
    pid_m = pid / grid_n
    pid_n = pid % grid_n

    

    # what are the offsets for matrices a and b?


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