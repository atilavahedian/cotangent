"""Exact incidence-matrix embedding backward for bounded byte vocabularies."""
import torch
from torch.nn import functional as F


class IncidenceEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.vocab = weight.shape[0]
        ctx.dtype = weight.dtype
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        # All zeros and ones are exact. FP32 gradient accumulation remains FP32;
        # no BF16 gradient rounding or sampled rows are introduced here.
        rows = indices.reshape(-1, 1)
        codes = torch.arange(ctx.vocab, device=indices.device).reshape(1, -1)
        incidence = (rows == codes).to(ctx.dtype)
        with torch.autocast(gradient.device.type, enabled=False):
            weight_gradient = incidence.transpose(0, 1) @ gradient.reshape(-1, gradient.shape[-1]).to(ctx.dtype)
        return None, weight_gradient


class PositionEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.shape = weight.shape
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        # Position lookup is unique in the model (one row for every position).
        # The broadcast-add backward has already summed the batch dimension.
        # Dedicated to the Transformer's arange(T) prefix lookup. For a full
        # context the gradient already has exactly the parameter's shape.
        missing = ctx.shape[0] - gradient.shape[0]
        return None, F.pad(gradient, (0, 0, 0, missing)) if missing else gradient


def install(model):
    """Keep identical parameter objects, initialization and state-dict names."""
    model.token.forward = lambda indices: IncidenceEmbedding.apply(indices, model.token.weight)
    model.position.forward = lambda indices: PositionEmbedding.apply(indices, model.position.weight)
    return model
