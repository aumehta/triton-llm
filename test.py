import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(a_ptr, b_ptr, output_ptr, num_elements: tl.constexpr, block_size: tl.constexpr):
    # get pid
    pid = tl.program_id(axis=0)
    # identify the portion of the a and b tensors that need to be summed
    block_start = pid * block_size
    thread_offsets = block_start + tl.arange(0, block_size)

    # create mask for sections we do not want to compute
    mask = thread_offsets < num_elements

    a_pointers = tl.load(a_ptr + thread_offsets, mask=mask)
    b_pointers = tl.load(b_ptr + thread_offsets, mask=mask)

    output = a_pointers + b_pointers
    tl.store(output_ptr + thread_offsets, output, mask=mask)

def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # create output_buffer
    output_buffer = torch.empty_like(a)

    # checks to ensure that a and b are of same size
    assert a.is_cuda and b.is_cuda
    num_elements = a.numel()
    assert num_elements == b.numel()

    # calculate number of blocks needed for task
    grid = lambda meta: (triton.cdiv(num_elements, meta['block_size']), )

    # compute using triton kernel function we created 
    add_kernel[grid](a, b, output_buffer, num_elements, block_size=128)
    return output_buffer

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = vector_addition(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')