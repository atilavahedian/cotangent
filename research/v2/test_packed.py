import copy

import pytest
import torch

from research.v2.packed import PackedAdamW


@pytest.mark.parametrize("max_norm", [1e-2, 1e6])
def test_multistep_adamw_and_moments_match(max_norm):
    torch.manual_seed(7)
    reference = torch.nn.Sequential(torch.nn.Linear(5, 9), torch.nn.GELU(),
                                    torch.nn.LayerNorm(9), torch.nn.Linear(9, 3)).double()
    packed_model = copy.deepcopy(reference)
    options = dict(lr=0.003, betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)
    native = torch.optim.AdamW(reference.parameters(), foreach=False, **options)
    packed = PackedAdamW(packed_model, **options)
    for step in range(8):
        x, y = torch.randn(11, 5, dtype=torch.float64), torch.randn(11, 3, dtype=torch.float64)
        native.zero_grad(set_to_none=True)
        packed.zero_grad()
        a, b = (reference(x) - y).square().mean(), (packed_model(x) - y).square().mean()
        torch.testing.assert_close(a, b, atol=2e-14, rtol=2e-14)
        a.backward()
        b.backward()
        for p, q in zip(reference.parameters(), packed_model.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-14, rtol=2e-14)
        norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm, foreach=False)
        native.step()
        packed_norm = packed.clip_and_step(max_norm)
        torch.testing.assert_close(norm, packed_norm, atol=2e-14, rtol=2e-14)
        for p, q in zip(reference.parameters(), packed_model.parameters()):
            torch.testing.assert_close(p, q, atol=2e-14, rtol=2e-14)
        for key in ("exp_avg", "exp_avg_sq"):
            joined = torch.cat([native.state[p][key].reshape(-1) for p in reference.parameters()])
            torch.testing.assert_close(joined, packed.optimizer.state[packed.master][key],
                                       atol=2e-14, rtol=2e-14)


def test_rejects_missing_gradient():
    model = torch.nn.Linear(2, 2)
    optimizer = PackedAdamW(model, lr=0.001)
    with pytest.raises(RuntimeError, match="every parameter"):
        optimizer.clip_and_step()
