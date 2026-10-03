"""Build stable byte-row segments alongside CPU batching, without GPU sorting."""
import numpy as np
import torch
from torch.nn import functional as F
from research.v3.embedding import PositionEmbedding
from research.v3 import segmented


def plan(values, vocab=256):
    rows = np.asarray(values).reshape(-1)
    permutation = np.argsort(rows, kind="stable")
    counts = np.bincount(rows.astype(np.int64), minlength=vocab)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return np.concatenate((permutation, offsets)).astype(np.int64)


def reduce(row_plan, gradient, vocab):
    n = gradient.numel() // gradient.shape[-1]
    g = gradient.reshape(n, gradient.shape[-1]).contiguous()
    if g.dtype != torch.float32:
        raise ValueError("The Metal kernel requires FP32 embedding gradients")
    output = torch.empty((vocab, g.shape[-1]), device=g.device, dtype=g.dtype)
    if segmented._library is None:
        segmented._library = torch.mps.compile_shader(segmented.SOURCE)
    segmented._library.embedding_segment_sum(g, row_plan[:n], row_plan[n:], output,
        g.shape[-1], output.numel(), threads=output.numel(), group_size=128)
    return output


class PresortedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight, row_plan):
        ctx.save_for_backward(row_plan)
        ctx.vocab = weight.shape[0]
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, gradient):
        row_plan, = ctx.saved_tensors
        return None, reduce(row_plan, gradient, ctx.vocab), None


def install(model):
    model.token._row_plan = None
    def forward(indices):
        if not torch.is_grad_enabled():
            return F.embedding(indices, model.token.weight)
        if model.token._row_plan is None:
            raise RuntimeError("Presorted backward requires the CPU batch row plan")
        return PresortedEmbedding.apply(indices, model.token.weight, model.token._row_plan)
    model.token.forward = forward
    model.position.forward = lambda indices: PositionEmbedding.apply(indices, model.position.weight)
    return model


def batch(model, data, offsets, sequence, device):
    values = np.stack([data[int(o):int(o)+sequence+1] for o in offsets]).astype(np.int64)
    row_plan = plan(values[:, :-1], model.cfg.vocab)
    # One CPU-to-MPS copy for data and one for the 18 KiB segment plan; both
    # transfers, stable sort and histogram are charged inside every training step.
    tensor = torch.from_numpy(values).to(device)
    model.token._row_plan = torch.from_numpy(row_plan).to(device)
    return tensor[:, :-1], tensor[:, 1:]
