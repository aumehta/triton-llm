import triton
import triton.language as tl
import torch

def softmax_og(x: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=1)[0]
    safe_x = x - x_max[:, None]
    row_exp = torch.exp(safe_x)
    row_sum = row_exp.sum(dim=1)
    res = row_exp / row_sum[:, None]
    return res

@triton.jit
def softmax_kernel(input_ptr, output_ptr, stride_input_row, stride_output_row, num_cols, block_size: tl.constexpr):
    # get pid which will be utilized as row_idx
    row_idx = tl.program_id(axis=0)

    row_start_ptr = input_ptr + (row_idx * stride_input_row)
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    # create mask for sections we do not want to compute
    mask = col_offsets < num_cols

    # move to SRAM 
    row = tl.load(input_pointers, mask=mask, other=float('-inf'))

    # perform softmax
    safe_row = row - tl.max(row, axis=0)
    row_exp = tl.exp(safe_row) 
    row_sum = tl.sum(row_exp, axis=0)
    output = row_exp / row_sum

    # calculate output pointers
    output_start_ptr = output_ptr + (row_idx * stride_output_row)
    output_pointers = output_start_ptr + col_offsets
    
    # writes to DRAM
    tl.store(output_pointers, output, mask=mask)

def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape

    # create output buffer
    output_buffer = torch.empty_like(x)
    
    # create block size that is the next power of 2 greater than the number of columns
    # triton works with powers of 2 so we make the blocks powers of 2 and any extras will be masked out in our kernel function
    block_size = triton.next_power_of_2(cols)

    # the number of warps to use for the kernel when compiled for GPUs. 
    # for example, if num_warps=8, then each kernel instance will be automatically parallelized to cooperatively execute using 8 * 32 = 256 threads.
    num_warps = 4

    if block_size > 2047:
        num_warps = 8
    if block_size > 4095:
        num_warps = 16
    
    # create grid
    grid = (rows,)

    # compute using triton kernel function we created
    softmax_kernel[grid](x, 
                         output_buffer, 
                         x.stride(0), 
                         output_buffer.stride(0), 
                         cols, 
                         block_size=block_size,
                         num_warps = num_warps
    )
    return output_buffer

if __name__ == "__main__":
    x = torch.tensor([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=torch.float32, device='cuda')
    print(f"Triton implemented Softmax: \n {softmax_triton(x)}")
    print(f"OG Softmax: \n {softmax_og(x)}")



