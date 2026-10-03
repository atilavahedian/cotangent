"""Exact gradient and optimizer-state packing; no sampled cotangents."""
from __future__ import annotations

import torch


class PackedAdamW:
    """Reindex equal-hyperparameter AdamW into one contiguous parameter vector.

    The model keeps independent autograd leaves sharing storage with ``master``.
    Autograd computes every derivative normally. Only the completed gradients
    are concatenated. This avoids a SliceBackward graph through the full master
    vector and permits one norm reduction and one optimizer tensor update.
    Construct before any forward pass; do not move/reassign parameters afterward.
    Sparse gradients, mixed dtypes, parameter groups and partial updates are
    deliberately unsupported rather than silently changing AdamW semantics.
    """

    def __init__(self, model, *, fused=False, **kwargs):
        self.parameters = tuple(model.parameters())
        if not self.parameters or any(not p.requires_grad for p in self.parameters):
            raise ValueError("Packing requires a nonempty set of trainable parameters")
        device, dtype = self.parameters[0].device, self.parameters[0].dtype
        if any(p.device != device or p.dtype != dtype or not p.is_contiguous()
               for p in self.parameters):
            raise ValueError("Packing requires homogeneous contiguous parameters")
        self.master = torch.nn.Parameter(torch.cat([p.detach().reshape(-1) for p in self.parameters]))
        offset = 0
        with torch.no_grad():
            for parameter in self.parameters:
                size = parameter.numel()
                parameter.set_(self.master.detach()[offset:offset + size].view_as(parameter))
                offset += size
        self.optimizer = torch.optim.AdamW([self.master], fused=fused, **kwargs)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.parameters:
            parameter.grad = None

    def clip_and_step(self, max_norm=1.0):
        gradients = [p.grad for p in self.parameters]
        if any(g is None or g.is_sparse for g in gradients):
            raise RuntimeError("Exact packing requires every parameter's dense gradient")
        self.master.grad = torch.cat([g.reshape(-1) for g in gradients])
        norm = torch.nn.utils.clip_grad_norm_([self.master], max_norm, foreach=False)
        self.optimizer.step()
        return norm
