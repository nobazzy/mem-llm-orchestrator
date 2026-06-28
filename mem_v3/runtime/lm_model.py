from __future__ import annotations

import torch
import torch.nn as nn


class TinyCausalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        d_model: int = 192,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 768,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        base_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", base_mask, persistent=False)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # GPT-style initialization.
        # PyTorch nn.Embedding defaults to std~1.0; with tied lm_head this creates
        # huge logits and very high initial CE loss. std=0.02 keeps the zero-run
        # loss near ln(vocab) instead of exploding to 20+.
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
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        mask = self.causal_mask[:length, :length].to(device=input_ids.device, dtype=x.dtype)
        x = self.blocks(x, mask=mask)
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
        # Approx. 45-50M parameters with GPT-2 vocabulary and tied embeddings.
        # Practical 8GB preset: lower optimizer/VRAM pressure than 100M while
        # preserving a meaningful transformer workload for controller evaluation.
        return TinyCausalTransformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            d_model=512,
            nhead=8,
            num_layers=8,
            dim_feedforward=2048,
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

