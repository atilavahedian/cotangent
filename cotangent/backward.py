"""Explicit, auditable linear-layer backward algorithms.

The main candidate sketches the parameter gradient and keeps activation-gradient
propagation exact. This avoids multiplying approximation bias across layer
Jacobians. It does NOT reduce saved-activation memory. The separate WHT-style
control compresses activations and both backward products, with biased gradients.
All implementations in this file are newly written for this project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Policy:
    mode: str = "native"
    fraction: float = 0.25
    tolerance: float = 0.5
    audit_every: int = 32
    warmup: int = 32
    min_rows: int = 64
    mixture: float = 0.05
    step: int = 0
    records: list[dict] = field(default_factory=list)
    states: dict[str, dict] = field(default_factory=dict)
    collect: bool = True
    pending_audits: list = field(default_factory=list)

    def start_step(self, step: int):
        self.step = step
        self.records.clear()
        self.pending_audits.clear()

    def is_audit(self):
        return self.mode == "adaptive" and (self.step < self.warmup or self.step % self.audit_every == 0)

    def aggregate(self):
        if not self.records:
            return {"sample_fraction": 1.0, "audit_layers": 0, "gradient_error": None}
        errors = [r["relative_error"] for r in self.records if r.get("relative_error") is not None]
        return {"sample_fraction": sum(r["fraction"] for r in self.records) / len(self.records),
                "audit_layers": sum(bool(r.get("audit")) for r in self.records),
                "gradient_error": sum(errors) / len(errors) if errors else None,
                "sampled_layers": sum(r["fraction"] < 1 for r in self.records)}

    def finish_step(self):
        """One batched device read for all audit diagnostics, charged to timing."""
        if not self.pending_audits:
            return
        values = torch.stack([entry[1] for entry in self.pending_audits]).detach().float().cpu().tolist()
        for (name, _, n, record), (norm2, moment, error2) in zip(self.pending_audits, values):
            norm2 = max(norm2, 1e-30)
            required = math.ceil(max(0, moment / norm2 - 1) / self.tolerance**2)
            k = min(n, max(self.min_rows, required))
            self.states[name] = {"gradient_norm2": norm2, "rows": k, "audited_step": self.step}
            record["relative_error"] = math.sqrt(max(0, error2) / norm2)


def probabilities(x: Tensor, delta: Tensor, mixture: float = 0.05) -> Tensor:
    """Positive importance probabilities proportional to outer-product norms."""
    score = x.float().square().sum(-1).sqrt() * delta.float().square().sum(-1).sqrt()
    # Clamp the denominator; a zero signal falls back to uniform sampling.
    p = score / score.sum().clamp_min(1e-30)
    p = (1 - mixture) * p + mixture / len(p)
    return p / p.sum()


def estimate_weight_gradient(x: Tensor, delta: Tensor, p: Tensor, rows: int,
                             generator: torch.Generator | None = None) -> Tensor:
    """With-replacement Horvitz–Thompson outer-product estimator.

    E[estimate | x, delta] == delta.T @ x before floating-point roundoff.
    Both dense sample products and duplicate draws count toward runtime.
    """
    indices = torch.multinomial(p, rows, replacement=True, generator=generator)
    xs = x.index_select(0, indices)
    ds = delta.index_select(0, indices)
    ds = ds * (1 / (rows * p.index_select(0, indices))).to(ds.dtype).unsqueeze(-1)
    return ds.transpose(0, 1) @ xs


def variance_bound(x: Tensor, delta: Tensor, p: Tensor) -> Tensor:
    """E||G_hat-G||_F^2 = (second_moment - ||G||_F^2) / k."""
    z2 = x.float().square().sum(-1) * delta.float().square().sum(-1)
    return (z2 / p).sum()


class SampledLinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="mps", cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight, bias, policy, name):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.policy = policy
        ctx.name = name
        return F.linear(x, weight, bias)

    @staticmethod
    @torch.amp.custom_bwd(device_type="mps")
    def backward(ctx, delta):
        x, weight = ctx.saved_tensors
        policy = ctx.policy
        xf = x.reshape(-1, x.shape[-1])
        df = delta.reshape(-1, delta.shape[-1])
        n = len(xf)
        # Exact activation gradient: no projection of error through earlier layers.
        dx = delta @ weight if ctx.needs_input_grad[0] else None
        db = df.sum(0) if ctx.has_bias else None
        record = {"layer": ctx.name, "fraction": 1.0, "audit": False, "relative_error": None}
        if policy.mode == "exact_custom":
            dw = df.transpose(0, 1) @ xf
        else:
            if policy.mode == "adaptive":
                state = policy.states.setdefault(ctx.name, {})
                if policy.is_audit() or "rows" not in state:
                    p = probabilities(xf, df, policy.mixture)
                    second = variance_bound(xf, df, p)
                    dw = df.transpose(0, 1) @ xf
                    norm2 = dw.float().square().sum()
                    # An independent sampled estimate measures the current error.
                    k = max(policy.min_rows, int(n * policy.fraction))
                    trial = estimate_weight_gradient(xf, df, p, k)
                    error2 = (trial.float() - dw.float()).square().sum()
                    record.update(audit=True)
                    policy.pending_audits.append((ctx.name, torch.stack([norm2, second, error2]), n, record))
                else:
                    # The previous audit selects a static budget between audits.
                    # This is a heuristic, NOT a certified per-step error bound.
                    k = min(n, state["rows"])
                    # Account for sampling/probability overhead: use dense when
                    # the requested rank removes too little work.
                    if k >= int(0.75 * n):
                        dw = df.transpose(0, 1) @ xf
                    else:
                        p = probabilities(xf, df, policy.mixture)
                        dw = estimate_weight_gradient(xf, df, p, k)
                        record["fraction"] = k / n
            else:
                uniform = policy.mode == "uniform"
                p = torch.full((n,), 1 / n, device=x.device) if uniform else probabilities(xf, df, policy.mixture)
                k = min(n, max(policy.min_rows, int(n * policy.fraction)))
                dw = estimate_weight_gradient(xf, df, p, k)
                record["fraction"] = k / n
        if policy.collect:
            policy.records.append(record)
        return dx, dw, db, None, None


_BASES: dict[tuple, Tensor] = {}


def hadamard_basis(length: int, rank: int, device, dtype) -> Tensor:
    """First natural-order columns of a normalized Sylvester Hadamard matrix.

    This is a WHT-style control, not a complete reproduction of LBP-WHT's
    task-specific basis selection or of INSTANT's learned projections.
    """
    if length & (length - 1):
        raise ValueError("WHT sequence length must be a power of two")
    key = (length, rank, str(device), dtype)
    if key not in _BASES:
        h = torch.ones(1, 1, dtype=torch.float64)
        while len(h) < length:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _BASES[key] = (h[:, :rank] / math.sqrt(length)).to(device=device, dtype=dtype)
    return _BASES[key]


class ProjectedLinearFunction(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="mps", cast_inputs=torch.bfloat16)
    def forward(ctx, x, weight, bias, basis, policy, name):
        xc = basis.transpose(0, 1) @ x
        ctx.save_for_backward(xc, weight, basis)
        ctx.has_bias = bias is not None
        ctx.policy, ctx.name = policy, name
        return F.linear(x, weight, bias)

    @staticmethod
    @torch.amp.custom_bwd(device_type="mps")
    def backward(ctx, delta):
        xc, weight, basis = ctx.saved_tensors
        dc = basis.transpose(0, 1) @ delta
        dx = basis @ (dc @ weight) if ctx.needs_input_grad[0] else None
        dw = (dc.transpose(-2, -1) @ xc).sum(0)
        db = delta.sum((0, 1)) if ctx.has_bias else None
        if ctx.policy.collect:
            ctx.policy.records.append({"layer": ctx.name, "fraction": basis.shape[1] / basis.shape[0], "audit": False})
        return dx, dw, db, None, None, None


class ResearchLinear(nn.Linear):
    def __init__(self, in_features, out_features, policy: Policy, name: str, bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.policy = policy
        self.research_name = name

    def forward(self, x):
        if not self.training or self.policy.mode == "native":
            return F.linear(x, self.weight, self.bias)
        if self.policy.mode == "adaptive" and not self.policy.is_audit():
            state = self.policy.states.get(self.research_name, {})
            n = x.numel() // x.shape[-1]
            if state.get("rows", 0) >= int(0.75 * n):
                # Dense fallback uses native autograd, avoiding sampling overhead.
                if self.policy.collect:
                    self.policy.records.append({"layer": self.research_name, "fraction": 1.0, "audit": False})
                return F.linear(x, self.weight, self.bias)
        if self.policy.mode == "wht":
            rank = max(1, int(x.shape[-2] * self.policy.fraction))
            basis = hadamard_basis(x.shape[-2], rank, x.device, x.dtype)
            return ProjectedLinearFunction.apply(x, self.weight, self.bias, basis, self.policy, self.research_name)
        return SampledLinearFunction.apply(x, self.weight, self.bias, self.policy, self.research_name)
