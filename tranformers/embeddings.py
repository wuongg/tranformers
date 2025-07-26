import torch
import math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # PE: shape [1, max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # sin cho các chỉ số chẵn
        pe[:, 1::2] = torch.cos(position * div_term)  # cos cho chỉ số lẻ
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # không tính gradient

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]