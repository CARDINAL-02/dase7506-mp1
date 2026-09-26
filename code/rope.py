"""Rotary position embeddings.

Everything is derived from ``head_dim`` at runtime. The contract tests build a
32-wide, 4-head model, so ``head_dim`` is 8 there and 32 under the baseline
config -- any hardcoded frequency table would break the gate.
"""
import torch


def build_rope(head_dim, context, base=10000.0):
    """Return (cos, sin) tables of shape [context, head_dim // 2]."""
    if head_dim % 2:
        raise ValueError(f'RoPE needs an even head_dim, got {head_dim}.')
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    freqs = torch.outer(torch.arange(context).float(), inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """Rotate [batch, heads, time, head_dim] in place-free fashion.

    Pairs elements (0,1), (2,3), ... exactly as ``build_rope``'s even-indexed
    frequency schedule assumes, and restores the interleaved layout.
    """
    cos, sin = cos[:x.shape[-2]].to(x.dtype), sin[:x.shape[-2]].to(x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
