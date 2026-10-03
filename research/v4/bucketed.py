"""Exploratory size-bucketed clipping: preserve one norm per parameter."""
from collections import defaultdict
import torch
from research.v2.packed import PackedAdamW


class BucketedAdamW(PackedAdamW):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        buckets = defaultdict(list)
        for index, p in enumerate(self.parameters):
            buckets[p.numel()].append(index)
        self.buckets = list(buckets.values())

    def clip_and_step(self, max_norm=1.):
        gradients = [p.grad for p in self.parameters]
        if any(g is None or g.is_sparse for g in gradients):
            raise RuntimeError("Every dense gradient is required")
        norms = [None]*len(gradients)
        for indices in self.buckets:
            matrix = torch.stack([gradients[i].reshape(-1) for i in indices])
            length = matrix.shape[1]
            if length < 4 or length % 2:
                values = torch.stack([torch.linalg.vector_norm(gradients[i], 2.) for i in indices])
            else:
                # Two reduced axes avoid MPS's different single-axis inner fast
                # path. The generic reduction uses the same per-parameter size,
                # thread count and linear element order as native scalar norms.
                values = torch.linalg.vector_norm(matrix.reshape(len(indices), length//2, 2), 2., dim=(1,2))
            for i, value in zip(indices, values.unbind()):
                norms[i] = value
        norm = torch.linalg.vector_norm(torch.stack(norms), 2.)
        self.master.grad = torch.cat([g.reshape(-1) for g in gradients])
        coefficient = torch.clamp(max_norm/(norm+1e-6), max=1.)
        self.master.grad.mul_(coefficient)
        self.optimizer.step()
        return norm
