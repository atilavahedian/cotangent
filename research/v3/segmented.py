"""Stable segmented embedding gradients, authored Metal implementation."""
import torch
from torch.nn import functional as F
from research.v3.embedding import PositionEmbedding

SOURCE = r"""
#include <metal_stdlib>
using namespace metal;
kernel void embedding_segment_sum(
    device const float* gradient [[buffer(0)]],
    device const long* permutation [[buffer(1)]],
    device const long* offsets [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant uint& width [[buffer(4)]],
    constant uint& elements [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= elements) return;
    uint token = index / width;
    uint feature = index % width;
    float total = 0.0f;
    // Stable sorting makes the reduction order the ascending original row
    // order within each token. Each output has exactly one writing thread.
    for (long row = offsets[token]; row < offsets[token + 1]; ++row) {
        total += gradient[permutation[row] * width + feature];
    }
    output[index] = total;
}
"""
_library = None


def reduce(indices, gradient, vocab):
    global _library
    values, permutation = torch.sort(indices.reshape(-1), stable=True)
    offsets = torch.searchsorted(values, torch.arange(vocab + 1, device=indices.device))
    grad = gradient.reshape(-1, gradient.shape[-1]).contiguous()
    if grad.dtype != torch.float32:
        raise ValueError("The Metal reduction currently requires FP32 embedding gradients")
    output = torch.empty((vocab, grad.shape[-1]), device=grad.device, dtype=grad.dtype)
    if _library is None:
        _library = torch.mps.compile_shader(SOURCE)
    _library.embedding_segment_sum(grad, permutation, offsets, output, grad.shape[-1], output.numel(),
                                   threads=output.numel(), group_size=128)
    return output


class SegmentedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, weight):
        ctx.save_for_backward(indices)
        ctx.vocab = weight.shape[0]
        return F.embedding(indices, weight)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        if gradient.device.type != "mps":
            # CPU double reference is independently checked against nn.Embedding;
            # actual Metal comparisons are required before accepting the backend.
            rows = indices.reshape(-1, 1)
            codes = torch.arange(ctx.vocab, device=indices.device).reshape(1, -1)
            return None, (rows == codes).to(gradient.dtype).T @ gradient.reshape(-1, gradient.shape[-1])
        return None, reduce(indices, gradient, ctx.vocab)


def install(model):
    model.token.forward = lambda indices: SegmentedEmbedding.apply(indices, model.token.weight)
    model.position.forward = lambda indices: PositionEmbedding.apply(indices, model.position.weight)
    return model
