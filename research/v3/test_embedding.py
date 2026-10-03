import pytest
import torch
from torch.nn import functional as F
from research.v3.embedding import IncidenceEmbedding, PositionEmbedding


def test_repeated_byte_indices_full_gradient_and_gradcheck():
    torch.manual_seed(5)
    indices = torch.tensor([[0, 2, 2, 2], [2, 1, 0, 2]])
    weight = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w: IncidenceEmbedding.apply(indices, w), (weight,))
    signal = torch.randn(2, 4, 7, dtype=torch.float64)
    expected = torch.autograd.grad(F.embedding(indices, weight), weight, signal)[0]
    observed = torch.autograd.grad(IncidenceEmbedding.apply(indices, weight), weight, signal)[0]
    torch.testing.assert_close(expected, observed, atol=1e-14, rtol=1e-14)
    assert observed[3:].abs().max() == 0


def test_position_broadcast_full_gradient():
    weight = torch.randn(8, 6, dtype=torch.float64, requires_grad=True)
    indices = torch.arange(5)
    signal = torch.randn(3, 5, 6, dtype=torch.float64)
    expected = torch.autograd.grad(F.embedding(indices, weight).unsqueeze(0).expand(3, -1, -1), weight, signal)[0]
    observed = torch.autograd.grad(PositionEmbedding.apply(indices, weight).unsqueeze(0).expand(3, -1, -1), weight, signal)[0]
    torch.testing.assert_close(expected, observed, atol=1e-14, rtol=1e-14)


def test_native_norm_packed_updates():
    import copy
    from research.v3.optimizer import NativeNormPackedAdamW
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 2)).double()
    other = copy.deepcopy(model)
    opts = dict(lr=.003, betas=(.9, .95), weight_decay=.1)
    a, b = torch.optim.AdamW(model.parameters(), foreach=False, **opts), NativeNormPackedAdamW(other, **opts)
    for _ in range(8):
        x = torch.randn(7, 4, dtype=torch.float64)
        a.zero_grad(set_to_none=True)
        b.zero_grad()
        model(x).square().mean().backward()
        other(x).square().mean().backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False)
        a.step()
        other_norm = b.clip_and_step()
        torch.testing.assert_close(norm, other_norm, rtol=2e-14, atol=2e-14)
        for p, q in zip(model.parameters(), other.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-14, atol=2e-14)
