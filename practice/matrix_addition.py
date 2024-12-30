import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(a_ptr, b_ptr, 
               output_ptr, 
               am_stride, an_stride,
               bm_stride, bn_stride,
               block_size_m: tl.constexpr, block_size_n: tl.constexpr, 
               M, N):
    
    # create the grid
    pid = tl.program_id(axis=0)
    grid_m = tl.cdiv(M, block_size_m)
    grid_n = tl.cdiv(N, block_size_n)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    # calculate starting positions of the current block
    block_start_m = pid_m * block_size_m
    block_start_n = pid_n * block_size_n

    # compute the indices for the current thread in the block
    offsets_m = block_start_m + tl.arange(0, block_size_m)
    offsets_n = block_start_n + tl.arange(0, block_size_n)

    mask_m = offsets_m < M
    mask_n = offsets_n < N

    # load elements from input matrices A and B
    a_pointers = tl.load(a_ptr + offsets_m[:, None] * am_stride + offsets_n[None, :] * an_stride, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    b_pointers = tl.load(b_ptr + offsets_m[:, None] * bm_stride + offsets_n[None, :] * bn_stride, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    output = a_pointers + b_pointers

    # write output to DRAM
    tl.store(output_ptr + offsets_m[:, None] * am_stride + offsets_n[None, :] * an_stride, output, mask=mask_m[:, None] & mask_n[None, :])

def matrix_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    output_buffer = torch.empty_like(a)

    assert a.shape == b.shape
    M, N = a.shape

    grid = lambda meta: (triton.cdiv(M, meta['block_size_m']) * triton.cdiv(N, meta['block_size_n']),)
    add_kernel[grid](a, b, output_buffer, a.stride(0), a.stride(1), b.stride(0), b.stride(1), block_size_m=32, block_size_n=32, M=M, N=N)

    return output_buffer

def verify():
    torch.manual_seed(0)
    size = (1024, 1024)
    a = torch.rand(size, device='cuda', dtype=torch.float32)
    b = torch.rand(size, device='cuda', dtype=torch.float32)
    output_torch = a + b
    output_triton = matrix_addition(a, b)

    print("Output (Torch):")
    print(output_torch)
    print("\nOutput (Triton):")
    print(output_triton)

    max_diff = torch.max(torch.abs(output_torch - output_triton))
    print(f"The maximum difference between Torch and Triton is {max_diff}")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(5, 11)],  # Different possible values for matrix sizes (rows/cols).
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='matrix-add-performance',  # Name for the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    a = torch.rand((size, size), device='cuda', dtype=torch.float32)
    b = torch.rand((size, size), device='cuda', dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a + b, quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matrix_addition(a, b), quantiles=quantiles)

    gbps = lambda ms: 3 * a.numel() * a.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    verify()
    benchmark.run(print_data=True, show_plots=True)