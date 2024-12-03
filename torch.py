import torch
import torch.nn.functional as F

def self_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, d: int) -> torch.Tensor:
    """
    Compute self-attention mechanism.

    Args:
        Q (torch.Tensor): Query matrix of shape (batch_size, seq_len, d).
        K (torch.Tensor): Key matrix of shape (batch_size, seq_len, d).
        V (torch.Tensor): Value matrix of shape (batch_size, seq_len, d).
        d (int): Dimension of the feature space (d_model).

    Returns:
        torch.Tensor: Output of the self-attention mechanism of shape (batch_size, seq_len, d).
    """
    # Step 1: Compute scaled dot-product attention scores
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d, dtype=torch.float32, device=Q.device))

    # Step 2: Apply softmax to get attention weights
    attention_weights = F.softmax(attention_scores, dim=-1)

    # Step 3: Multiply weights with value matrix
    output = torch.matmul(attention_weights, V)

    return output

# Example Usage
if __name__ == "__main__":
    batch_size = 2
    seq_len = 5
    d = 4

    # Random Q, K, V matrices
    Q = torch.randn(batch_size, seq_len, d, device='cuda')
    K = torch.randn(batch_size, seq_len, d, device='cuda')
    V = torch.randn(batch_size, seq_len, d, device='cuda')

    # Self-attention output
    attention_output = self_attention(Q, K, V, d)
    print("Self-Attention Output:\n", attention_output)
