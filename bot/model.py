"""Transformer value/policy network.

A faithful, lightly cleaned port of the reference architecture. Torch is
imported at module load, so this module is only imported lazily by the
evaluator/training code (never required for the heuristic agent to run).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import ENCODER_SIZE, NUM_WORDS_ENCODER, FeatureDims, SparseVector


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_feedforward: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, d_feedforward)
        self.fc2 = nn.Linear(d_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, encoder_out: torch.Tensor) -> torch.Tensor:
        y, _ = self.attention(x, encoder_out, encoder_out, need_weights=False)
        res = self.norm1(x + y)
        y = F.relu(self.fc1(res))
        y = self.fc2(y)
        return self.norm2(res + y)


class ValuePolicyNet(nn.Module):
    """Encoder (state) + decoder (per-action) producing a value and policy."""

    def __init__(
        self,
        decoder_size: int,
        d_model: int = 128,
        num_heads: int = 2,
        d_feedforward: int = 256,
        num_layers_encoder: int = 1,
        num_layers_decoder: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.encoder_bag = nn.EmbeddingBag(ENCODER_SIZE, d_model, mode="sum")
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, d_feedforward, 0)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers_encoder, enable_nested_tensor=False
        )
        self.encoder_fc = nn.Linear(d_model, 1)
        self.decoder_bag = nn.EmbeddingBag(decoder_size, d_model, mode="sum")
        self.decoder = nn.ModuleList(
            DecoderLayer(d_model, num_heads, d_feedforward) for _ in range(num_layers_decoder)
        )
        self.decoder_fc = nn.Linear(d_model, 1)

    def forward(
        self,
        index_encoder: torch.Tensor,
        value_encoder: torch.Tensor,
        offset_encoder: torch.Tensor,
        index_decoder: torch.Tensor,
        value_decoder: torch.Tensor,
        offset_decoder: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.encoder_bag(index_encoder, offset_encoder, value_encoder)
        v = v.reshape(-1, NUM_WORDS_ENCODER, self.d_model).transpose(0, 1)
        batch_size = v.size(1)
        encoder_out = self.encoder(v)
        v = self.encoder_fc(encoder_out)
        value = torch.tanh(v.mean(0))

        p = self.decoder_bag(index_decoder, offset_decoder, value_decoder)
        p = p.reshape(batch_size, -1, self.d_model).transpose(0, 1)
        for layer in self.decoder:
            p = layer(p, encoder_out)
        p = self.decoder_fc(p)
        p = p.transpose(0, 1).view(batch_size, -1)
        policy = torch.tanh(p)
        return value, policy


def build_model(dims: FeatureDims, **kwargs) -> ValuePolicyNet:
    return ValuePolicyNet(dims.decoder_size, **kwargs)


def eval_batch(
    model: ValuePolicyNet, sv_enc: SparseVector, sv_dec: SparseVector
) -> tuple[float, list[float]]:
    """Run a single-sample forward pass; returns (value, policy list)."""

    device = next(model.parameters()).device
    value, policy = model(
        torch.tensor(sv_enc.index, dtype=torch.int32, device=device),
        torch.tensor(sv_enc.value, dtype=torch.float32, device=device),
        torch.tensor(sv_enc.offset, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.index, dtype=torch.int32, device=device),
        torch.tensor(sv_dec.value, dtype=torch.float32, device=device),
        torch.tensor(sv_dec.offset, dtype=torch.int32, device=device),
    )
    return value.tolist()[0][0], policy.tolist()[0]
