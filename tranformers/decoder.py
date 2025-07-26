import torch.nn as nn
from attention import MultiHeadAttention
from layers import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_mha = MultiHeadAttention(d_model, num_heads)     # Masked MHA
        self.enc_dec_mha = MultiHeadAttention(d_model, num_heads)  # Encoder-Decoder Attention
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_mha(x, x, x, tgt_mask))             # Masked Self Attention
        x = self.norm2(x + self.enc_dec_mha(x, enc_output, enc_output, src_mask))  # Cross Attention
        x = self.norm3(x + self.ff(x))                                   # Feed Forward
        return x