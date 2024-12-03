import triton
import triton.language as tl
import torch

@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, output_ptr, 
    q_stride, k_stride, v_stride, output_stride, 
    scale, num_heads, seq_len, dim_per_head, 
    block_size: tl.constexpr
):
    # Grid info: batch x head x block_row
    batch_head_id = tl.program_id(axis=0)
    block_row = tl.program_id(axis=1)

    # Extract batch, head, and row
    batch_id = batch_head_id // num_heads
    head_id = batch_head_id % num_heads

    # Offsets for query, key, value
    q_start = Q_ptr + batch_id * q_stride + head_id * dim_per_head
    k_start = K_ptr + batch_id * k_stride + head_id * dim_per_head
    v_start = V_ptr + batch_id * v_stride + head_id * dim_per_head
    o_start = output_ptr + batch_id * output_stride + head_id * dim_per_head

    # Compute attention scores (Q @ K^T)
    row_offsets = block_row * block_size + tl.arange(0, block_size)
    col_offsets = tl.arange(0, block_size)
    query_row = tl.load(q_start + row_offsets[:, None], mask=row_offsets < seq_len)
    key_col = tl.load(k_start + col_offsets[None, :], mask=col_offsets < seq_len)
    attn_scores = tl.dot(query_row, key_col.T) * scale

    # Apply softmax to scores
    attn_scores = attn_scores - tl.max(attn_scores, axis=1)[:, None]
    exp_scores = tl.exp(attn_scores)
    sum_exp_scores = tl.sum(exp_scores, axis=1)
    normalized_scores = exp_scores / sum_exp_scores[:, None]

    # Compute attention output (normalized_scores @ V)
    value_matrix = tl.load(v_start + col_offsets[None, :], mask=col_offsets < seq_len)
    attn_output = tl.dot(normalized_scores, value_matrix)

    # Store output
    tl.store(o_start + row_offsets[:, None], attn_output, mask=row_offsets < seq_len)

def self_attention(Q, K, V, scale, num_heads):
    B, H, S, D = Q.shape  # Batch, Heads, Seq_len, Dim_per_head
    output = torch.empty_like(Q)

    grid = (B * H, (S + 31) // 32)  # Adjust block size based on seq_len
    attention_kernel[grid](
        Q, K, V, output, 
        Q.stride(0), K.stride(0), V.stride(0), output.stride(0), 
        scale, num_heads, S, D, block_size=32
    )
    return output

if __name__ == "__main__":
    batch_size = 2
    num_heads = 4
    seq_len = 128
    dim_per_head = 64

    scale = 1.0 / (dim_per_head ** 0.5)
    Q = torch.rand((batch_size, num_heads, seq_len, dim_per_head), device='cuda', dtype=torch.float32)
    K = torch.rand((batch_size, num_heads, seq_len, dim_per_head), device='cuda', dtype=torch.float32)
    V = torch.rand((batch_size, num_heads, seq_len, dim_per_head), device='cuda', dtype=torch.float32)

    output = self_attention(Q, K, V, scale, num_heads)
    print(f"Self-attention output: {output.shape}")
