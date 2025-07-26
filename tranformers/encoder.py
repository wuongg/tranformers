import torch.nn as nn
from attention import MultiHeadAttention
from layers import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.mha(x, x, x, mask))  # Residual + LayerNorm sau MHA
        x = self.norm2(x + self.ff(x))               # Residual + LayerNorm sau FFN
        return x