"""Factorial gradient handling and executable native baselines."""
from collections import defaultdict
import torch
from research.v2.packed import PackedAdamW

ARMS = ("scalar-native", "bucket-native", "scalar-packed", "bucket-packed", "foreach-native", "auto-native", "loop-native", "compile-native", "aot-native")

def scalar_norm(gradients):
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2.) for g in gradients]), 2.)

def bucket_norm(gradients, buckets):
    norms = [None] * len(gradients)
    for indices in buckets:
        matrix = torch.stack([gradients[i].reshape(-1) for i in indices])
        b, k = matrix.shape
        values = torch.stack([torch.linalg.vector_norm(gradients[i], 2.) for i in indices]) if k < 4 or k % 2 else torch.linalg.vector_norm(matrix.reshape(b, k // 2, 2), 2., dim=(1, 2))
        for i, value in zip(indices, values.unbind()):
            norms[i] = value
    return torch.linalg.vector_norm(torch.stack(norms), 2.)

class Handler:
    def __init__(self, model, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.parameters = arm, tuple(model.parameters())
        ranges = []
        for p in self.parameters:
            if not p.requires_grad or not p.is_contiguous():
                raise ValueError("Dense contiguous active parameters are required")
            storage = p.untyped_storage().data_ptr()
            lo, hi = p.storage_offset(), p.storage_offset() + p.numel()
            if any(storage == s and max(lo, a) < min(hi, b) for s, a, b in ranges):
                raise ValueError("Distinct parameter objects may not overlap storage")
            ranges.append((storage, lo, hi))
        self.packed = arm.endswith("packed")
        opts = dict(lr=.001, betas=(.9, .95), eps=1e-8, weight_decay=.1)
        if self.packed:
            self.wrapper = PackedAdamW(model, fused=True, **opts)
            self.optimizer = self.wrapper.optimizer
        else:
            fused = False if arm == "loop-native" else None if arm == "auto-native" else True
            self.optimizer = torch.optim.AdamW(self.parameters, fused=fused, foreach=False if fused is not None else None, **opts)
        groups = defaultdict(list)
        for i, p in enumerate(self.parameters):
            groups[p.numel()].append(i)
        self.buckets = list(groups.values())

    def zero_grad(self):
        if self.packed:
            self.wrapper.zero_grad()
        else:
            self.optimizer.zero_grad(set_to_none=True)

    def lr(self, value):
        for group in self.optimizer.param_groups:
            group["lr"] = value

    @torch.no_grad()
    def step(self):
        gradients = [p.grad for p in self.parameters]
        if any(g is None or g.is_sparse or not g.is_contiguous() for g in gradients):
            raise ValueError("Every parameter requires a dense contiguous gradient")
        if self.arm.startswith("bucket"):
            norm = bucket_norm(gradients, self.buckets)
        elif self.arm == "foreach-native":
            norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(gradients, 2.)), 2.)
        else:
            norm = scalar_norm(gradients)
        c = torch.clamp(1. / (norm + 1e-6), max=1.)
        if self.packed:
            self.wrapper.master.grad = torch.cat([g.reshape(-1) for g in gradients])
            self.wrapper.master.grad.mul_(c)
        elif self.arm == "foreach-native":
            torch._foreach_mul_(gradients, c)
        else:
            for g in gradients:
                g.mul_(c)
        self.optimizer.step()
        return norm

def forward_callable(model, arm):
    if arm == "compile-native":
        return torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
    if arm == "aot-native":
        return torch.compile(model, backend="aot_eager", fullgraph=True, dynamic=False)
    return model
