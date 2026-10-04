"""Additional byte-language architectures for the extension study."""
from dataclasses import asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer

CONFIGS = {
    "transformer-small": dict(architecture="transformer", width=256, layers=4, heads=4, sequence=256, vocab=256),
    "transformer-medium": dict(architecture="transformer", width=512, layers=6, heads=8, sequence=256, vocab=256),
    "gated-gqa": dict(architecture="gated-gqa", width=384, layers=6, heads=6, sequence=256, vocab=256),
    "causal-conv": dict(architecture="causal-conv", width=256, layers=6, heads=4, sequence=256, vocab=256),
}

class GatedBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads, self.dim = cfg.heads, 2, cfg.width // cfg.heads
        self.norm1, self.norm2 = nn.RMSNorm(cfg.width), nn.RMSNorm(cfg.width)
        self.q = nn.Linear(cfg.width, cfg.width, bias=False)
        self.kv = nn.Linear(cfg.width, 2 * self.kv_heads * self.dim, bias=False)
        self.out = nn.Linear(cfg.width, cfg.width, bias=False)
        hidden = 8 * cfg.width // 3
        self.gate, self.up = nn.Linear(cfg.width, hidden, bias=False), nn.Linear(cfg.width, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.width, bias=False)

    def forward(self, x):
        b, t, _ = x.shape
        z = self.norm1(x)
        q = self.q(z).view(b, t, self.heads, self.dim).transpose(1, 2)
        k, v = self.kv(z).chunk(2, -1)
        k, v = [a.view(b, t, self.kv_heads, self.dim).transpose(1, 2).repeat_interleave(self.heads // self.kv_heads, dim=1) for a in (k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
        x = x + self.out(y.transpose(1, 2).contiguous().view(b, t, -1))
        z = self.norm2(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))

class ConvBlock(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.dilation = 2 ** index
        self.norm1, self.norm2 = nn.LayerNorm(cfg.width), nn.LayerNorm(cfg.width)
        self.conv = nn.Conv1d(cfg.width, cfg.width, 3, dilation=self.dilation, groups=cfg.width, bias=False)
        self.mix = nn.Linear(cfg.width, cfg.width, bias=False)
        self.up, self.down = nn.Linear(cfg.width, 4 * cfg.width, bias=False), nn.Linear(4 * cfg.width, cfg.width, bias=False)

    def forward(self, x):
        z = self.norm1(x).transpose(1, 2)
        z = self.conv(F.pad(z, (2 * self.dilation, 0))).transpose(1, 2)
        x = x + self.mix(F.gelu(z))
        return x + self.down(F.gelu(self.up(self.norm2(x))))

class ByteModel(nn.Module):
    def __init__(self, cfg, architecture):
        super().__init__()
        self.cfg = cfg
        self.token, self.position = nn.Embedding(cfg.vocab, cfg.width), nn.Embedding(cfg.sequence, cfg.width)
        self.blocks = nn.ModuleList([GatedBlock(cfg) if architecture == "gated-gqa" else ConvBlock(cfg, i) for i in range(cfg.layers)])
        self.norm = nn.RMSNorm(cfg.width) if architecture == "gated-gqa" else nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab, bias=False)
        self.apply(Transformer._init)
        for name, p in self.named_parameters():
            if name.endswith("out.weight") or name.endswith("down.weight") or name.endswith("mix.weight"):
                nn.init.normal_(p, std=.02 / math.sqrt(2 * cfg.layers))

    def forward(self, tokens, targets=None):
        x = self.token(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab).float(), targets.reshape(-1)) if targets is not None else None
        return logits, loss

def make_model(name, seed, device="mps"):
    torch.manual_seed(seed)
    values = dict(CONFIGS[name])
    architecture = values.pop("architecture")
    cfg = ModelConfig(**values)
    model = Transformer(cfg, Policy(mode="native")) if architecture == "transformer" else ByteModel(cfg, architecture)
    return model.to(device)
