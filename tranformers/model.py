import torch.nn as nn
from embeddings import PositionalEncoding
from encoder import EncoderLayer
from decoder import DecoderLayer


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=100):
        super().__init__()
        # Embedding từ ID → vector
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        # Stack nhiều encoder/decoder layer
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)  # Dự đoán token tiếp theo

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.pos_encoding(self.src_embed(src))  # [B, T_src, d_model]
        tgt = self.pos_encoding(self.tgt_embed(tgt))  # [B, T_tgt, d_model]
        for layer in self.encoder_layers:
            src = layer(src, src_mask)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, src_mask, tgt_mask)
        return self.fc_out(tgt)  # [B, T_tgt, vocab_size]