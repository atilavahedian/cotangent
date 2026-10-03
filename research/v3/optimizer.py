"""Pack the optimizer, while preserving native per-parameter norm reduction."""
import torch
from research.v2.packed import PackedAdamW


class NativeNormPackedAdamW(PackedAdamW):
    def clip_and_step(self, max_norm=1.):
        gradients = [p.grad for p in self.parameters]
        if any(g is None or g.is_sparse for g in gradients):
            raise RuntimeError("Every dense parameter gradient is required")
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2.) for g in gradients]), 2.)
        self.master.grad = torch.cat([g.reshape(-1) for g in gradients])
        coefficient = torch.clamp(max_norm / (norm + 1e-6), max=1.)
        self.master.grad.mul_(coefficient)
        self.optimizer.step()
        return norm
