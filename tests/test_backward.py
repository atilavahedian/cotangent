import torch
from torch.nn import functional as F

from cotangent.backward import (Policy, ResearchLinear, probabilities,
                                estimate_weight_gradient, variance_bound, hadamard_basis)
from cotangent.model import ModelConfig, Transformer


def test_exact_custom_gradcheck_and_bias():
    policy = Policy(mode="exact_custom", collect=False)
    layer = ResearchLinear(5, 7, policy, "test").double()
    x = torch.randn(2, 4, 5, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda z: layer(z), (x,), atol=1e-6)
    upstream = torch.randn(2, 4, 7, dtype=torch.double)
    grads = torch.autograd.grad(layer(x), (x, layer.weight, layer.bias), upstream)
    refs = torch.autograd.grad(F.linear(x, layer.weight, layer.bias), (x, layer.weight, layer.bias), upstream)
    for actual, expected in zip(grads, refs):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_sampling_unbiasedness_and_variance():
    torch.manual_seed(42)
    x = torch.randn(7, 3, dtype=torch.double)
    delta = torch.randn(7, 4, dtype=torch.double)
    p = probabilities(x, delta).double()
    exact = delta.T @ x
    # Analytic enumeration verifies unbiasedness without a flaky sampling test.
    expectation = sum(p[i] * torch.outer(delta[i], x[i]) / p[i] for i in range(len(x)))
    torch.testing.assert_close(expectation, exact)
    second = (x.square().sum(-1) * delta.square().sum(-1) / p).sum()
    expected_variance = (second - exact.square().sum()) / 4
    generator = torch.Generator().manual_seed(81)
    draws = torch.stack([estimate_weight_gradient(x, delta, p, 4, generator) for _ in range(4000)])
    observed = (draws - exact).square().sum((-2, -1)).mean()
    assert abs(float(observed / expected_variance) - 1) < 0.07
    assert (draws.mean(0) - exact).norm() / exact.norm() < 0.06


def test_zero_signal_probabilities_are_valid():
    p = probabilities(torch.zeros(10, 3), torch.zeros(10, 4))
    torch.testing.assert_close(p, torch.full((10,), 0.1))


def test_full_rank_wht_is_exact():
    torch.manual_seed(12)
    layer = ResearchLinear(5, 7, Policy(mode="wht", fraction=1, collect=False), "test").double()
    x = torch.randn(2, 8, 5, dtype=torch.double, requires_grad=True)
    upstream = torch.randn(2, 8, 7, dtype=torch.double)
    actual = torch.autograd.grad(layer(x), (x, layer.weight, layer.bias), upstream)
    expected = torch.autograd.grad(F.linear(x, layer.weight, layer.bias), (x, layer.weight, layer.bias), upstream)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=1e-10, rtol=1e-10)


def test_transformer_exact_custom_matches_native():
    cfg = ModelConfig(width=32, layers=2, heads=4, sequence=16)
    torch.manual_seed(9)
    native = Transformer(cfg, Policy(mode="native")).double()
    torch.manual_seed(9)
    custom = Transformer(cfg, Policy(mode="exact_custom", collect=False)).double()
    tokens = torch.randint(0, 256, (2, 16))
    targets = torch.randint(0, 256, (2, 16))
    native(tokens, targets)[1].backward()
    custom(tokens, targets)[1].backward()
    for (n, a), (m, b) in zip(native.named_parameters(), custom.named_parameters()):
        assert n == m
        torch.testing.assert_close(a.grad, b.grad, atol=1e-9, rtol=1e-7)


def test_sampling_keeps_input_gradient_exact():
    torch.manual_seed(4)
    policy = Policy(mode="importance", fraction=0.25, min_rows=2, collect=False)
    layer = ResearchLinear(5, 7, policy, "test").double()
    x = torch.randn(2, 8, 5, dtype=torch.double, requires_grad=True)
    delta = torch.randn(2, 8, 7, dtype=torch.double)
    dx = torch.autograd.grad(layer(x), x, delta)[0]
    torch.testing.assert_close(dx, delta @ layer.weight, atol=1e-12, rtol=1e-12)


def test_adaptive_warmup_is_exact():
    policy = Policy(mode="adaptive", min_rows=2, collect=False)
    layer = ResearchLinear(5, 7, policy, "test").double()
    x = torch.randn(2, 8, 5, dtype=torch.double, requires_grad=True)
    delta = torch.randn(2, 8, 7, dtype=torch.double)
    dw = torch.autograd.grad(layer(x), layer.weight, delta)[0]
    torch.testing.assert_close(dw, delta.reshape(-1, 7).T @ x.reshape(-1, 5))
