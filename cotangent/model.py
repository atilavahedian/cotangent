"""A newly written pre-norm byte-level causal transformer."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .backward import Policy, ResearchLinear


@dataclass
class ModelConfig:
    width: int = 256
    layers: int = 4
    heads: int = 4
    sequence: int = 256
    vocab: int = 256


class Attention(nn.Module):
    def __init__(self, cfg, policy, name):
        super().__init__()
        self.heads = cfg.heads
        self.qkv = ResearchLinear(cfg.width, 3 * cfg.width, policy, name + ".qkv", bias=False)
        self.out = ResearchLinear(cfg.width, cfg.width, policy, name + ".out", bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q, k, v = [z.reshape(b, t, self.heads, c // self.heads).transpose(1, 2) for z in [q, k, v]]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
        return self.out(y.transpose(1, 2).contiguous().reshape(b, t, c))


class Block(nn.Module):
    def __init__(self, cfg, policy, index):
        super().__init__()
        name = f"blocks.{index}"
        self.norm1 = nn.LayerNorm(cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.attention = Attention(cfg, policy, name + ".attention")
        self.up = ResearchLinear(cfg.width, 4 * cfg.width, policy, name + ".up", bias=False)
        self.down = ResearchLinear(4 * cfg.width, cfg.width, policy, name + ".down", bias=False)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig, policy: Policy):
        super().__init__()
        self.cfg = cfg
        self.policy = policy
        self.token = nn.Embedding(cfg.vocab, cfg.width)
        self.position = nn.Embedding(cfg.sequence, cfg.width)
        self.blocks = nn.ModuleList([Block(cfg, policy, i) for i in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        # Exact untied output head: avoids conflating approximation with weight tying.
        self.head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("attention.out.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab).float(), targets.reshape(-1)) if targets is not None else None
        return logits, loss
