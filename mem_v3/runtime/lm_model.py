from __future__ import annotations

import torch
import torch.nn as nn


import torch.nn.functional as F


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        # Attention sub-layer with Pre-LN and native SDPA
        qkv = self.qkv(self.ln1(x)).reshape(b, t, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).reshape(b, t, c)
        x = x + self.proj(att)
        # Feedforward sub-layer with Pre-LN
        ff = self.fc2(F.gelu(self.fc1(self.ln2(x))))
        x = x + ff
        return x


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 768,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.max_seq_len = max(max_seq_len, seq_len)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # GPT-style initialization.
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


def build_tiny_causal_lm(vocab_size: int, seq_len: int, preset: str = "tiny_decoder") -> nn.Module:

    if preset in {"medium_50m", "decoder_50m", "50m_decoder"}:
        # Approx. 45-50M parameters with GPT-2 vocabulary and tied embeddings.
        # Practical 8GB preset: lower optimizer/VRAM pressure than 100M while
        # preserving a meaningful transformer workload for controller evaluation.
        return TinyCausalTransformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=512,
            nhead=8,
            num_layers=7,
            dim_feedforward=2048,
        )

    if preset in {"medium_75m", "decoder_75m", "75m_decoder"}:
        # Approx. 72M parameters with GPT-2 vocabulary and tied embeddings.
        # Practical 8GB preset: high capacity while fitting smoothly in memory.
        return TinyCausalTransformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=640,
            nhead=10,
            num_layers=8,
            dim_feedforward=2560,
        )

    if preset in {"medium_100m", "decoder_100m", "100m_decoder"}:
        # Approx. 95-105M parameters with GPT-2 vocabulary and tied embeddings.
        # This is the validated 100M-class local endurance preset.
        return TinyCausalTransformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=768,
            nhead=12,
            num_layers=8,
            dim_feedforward=3072,
        )
    if preset == "small_decoder":
        return TinyCausalTransformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
        )
    return TinyCausalTransformer(
        vocab_size=vocab_size,
        seq_len=seq_len,
        d_model=192,
        nhead=6,
        num_layers=4,
        dim_feedforward=768,
    )

