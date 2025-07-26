import torch
import torch.nn.functional as F
import torch.nn as nn
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: shape [batch_size, num_heads, seq_len, d_k]
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # [B, H, T_q, T_k]
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # Mask các vị trí không hợp lệ
    attn = F.softmax(scores, dim=-1)  # Chuẩn hóa theo chiều key
    return torch.matmul(attn, V), attn  # Output: context vector + attention weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0  # đảm bảo chia đều
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.W_q = nn.Linear(d_model, d_model)  # Dự báo Q
        self.W_k = nn.Linear(d_model, d_model)  # Dự báo K
        self.W_v = nn.Linear(d_model, d_model)  # Dự báo V
        self.fc_out = nn.Linear(d_model, d_model)  # Kết quả tổng hợp

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        # Tách thành nhiều head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Attention từng head
        x, attn = scaled_dot_product_attention(Q, K, V, mask)
        # Gộp lại các head
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.fc_out(x)  # Output: [B, T, d_model]
