"""Student model: the baseline GPT with a switchable positional-encoding mode.

``config['pos_mode']`` selects the ablation arm; everything else is held fixed.

    'learned'  learned position embedding      -- the baseline arm (1,088,256 params)
    'none'     no positional information       -- lower-bound control (1,055,488)
    'rope'     rotary position embeddings      -- the method            (1,055,488)

The 'learned' arm is byte-for-byte the behaviour of ``model.py``'s GPT, so the
three arms differ only in how position reaches attention.

``config['dropout']`` (default 0.0) adds embedding and residual dropout. At 0 it
is an exact no-op: nn.Dropout(0.0) neither draws from the RNG nor changes any
value, so with dropout unset the training run reaches the same weights it did
before this key existed. Note that the checkpoint *bytes* do differ, because
torch.save pickles the module tree and there is now one more child in it; the
state dict is identical tensor for tensor.
"""
import torch
from torch import nn
from torch.nn import functional as F
from rope import apply_rope, build_rope

POS_MODES = ('learned', 'none', 'rope')


class Block(nn.Module):
    def __init__(self, width, heads, pos_mode, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.proj = nn.Linear(width, 3 * width), nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, cos=None, sin=None):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if cos is not None:
            # Rotating q and k by absolute position makes attention scores depend
            # only on their relative offset. Applied before the causal mask, so
            # position t still sees nothing past t.
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # Each position attends only to itself and earlier input tokens.
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(attended.transpose(1, 2).reshape(batch, length, width)))
        return x + self.drop(self.mlp(self.norm2(x)))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width = config['width']
        self.pos_mode = config.get('pos_mode', 'rope')
        if self.pos_mode not in POS_MODES:
            raise ValueError(f'pos_mode must be one of {POS_MODES}, got {self.pos_mode!r}.')
        dropout = config.get('dropout', 0.0)
        self.token = nn.Embedding(config['vocab'], width)
        if self.pos_mode == 'learned':
            self.pos = nn.Embedding(self.context, width)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(width, config['heads'], self.pos_mode, dropout)
                                     for _ in range(config['depth'])])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self.initialize)
        self.head.weight = self.token.weight
        if self.pos_mode == 'rope':
            # Non-persistent: constant tables, rebuilt from config on load, kept
            # out of the checkpoint. Slice to the actual length inside apply_rope.
            cos, sin = build_rope(width // config['heads'], self.context)
            self.register_buffer('rope_cos', cos, persistent=False)
            self.register_buffer('rope_sin', sin, persistent=False)

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=.02)
            if getattr(module, 'bias', None) is not None:
                nn.init.zeros_(module.bias)

    def features(self, ids):
        x = self.token(ids)
        if self.pos_mode == 'learned':
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device))
        x = self.drop(x)
        cos = self.rope_cos if self.pos_mode == 'rope' else None
        sin = self.rope_sin if self.pos_mode == 'rope' else None
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)

    def forward(self, ids):
        """Training interface: unnormalized next-token logits [batch, time, vocab]."""
        return self.head(self.features(ids))

    def predict_log_probs(self, ids):
        """Evaluation interface: normalized log probabilities, with no access to targets.

        Stateless: every call rebuilds the activations from ``ids`` alone, so each
        evaluation window starts fresh. Dropout is inactive here because the scorer
        puts the model in eval mode, and p=0 would be a no-op regardless.
        """
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return GPT(config)
