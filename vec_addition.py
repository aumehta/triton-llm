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

    # tl.load() moves to SRAM 
    a_pointers = tl.load(a_ptr + thread_offsets, mask=mask)
    b_pointers = tl.load(b_ptr + thread_offsets, mask=mask)
 
    # perform vector addition
    output = a_pointers + b_pointers
    
    # writes to DRAM
    tl.store(output_ptr + thread_offsets, output, mask=mask)

def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # create output buffer
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

def verify():
    torch.manual_seed(0)
    size = 98432
    a = torch.rand(size, device='cuda')
    b = torch.rand(size, device='cuda')
    output_torch = a + b
    output_triton = vector_addition(a, b)
    print(output_torch)
    print(output_triton)
    print(f'The maximum difference between torch and triton is '
        f'{torch.max(torch.abs(output_torch - output_triton))}')
    
    '''
    can also use torch.close(output_torch, output_triton) 
    this will basically return True if the two vectors are within the absolute and relative tolerance
    https://pytorch.org/docs/main/generated/torch.allclose.html
    '''

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: vector_addition(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)
    
if __name__ == "__main__":
    verify()
    benchmark.run(print_data=True, show_plots=True)