"""Verify deterministic full-model derivatives on the measured target backend."""
import json
from pathlib import Path
import torch
import torch.utils.deterministic
from cotangent.model import ModelConfig
from cotangent.runtime import synchronize
from research.v3.pilot import build

torch.set_num_threads(4)
torch.utils.deterministic.fill_uninitialized_memory = False
checks = []
for precision in ("fp32", "bf16"):
    cfg = ModelConfig()
    native, aopt, ahash, _ = build(cfg, 503, "native-fused")
    candidate, bopt, bhash, _ = build(cfg, 503, "incidence-packed-flatnorm")
    assert ahash == bhash
    torch.manual_seed(509)
    x, y = torch.randint(256, (8, 256), device="mps"), torch.randint(256, (8, 256), device="mps")
    torch.use_deterministic_algorithms(False)
    with torch.autocast("mps", dtype=torch.bfloat16, enabled=precision == "bf16"):
        _, loss_a = native(x, y)
    loss_a.backward()
    torch.use_deterministic_algorithms(True)
    with torch.autocast("mps", dtype=torch.bfloat16, enabled=precision == "bf16"):
        _, loss_b = candidate(x, y)
    loss_b.backward()
    synchronize(torch.device("mps"))
    torch.testing.assert_close(loss_a, loss_b, rtol=0., atol=0.)
    scale = max(float(p.grad.abs().max().item()) for p in native.parameters())
    maximum = max(float((p.grad-q.grad).abs().max().item()) for p,q in zip(native.parameters(),candidate.parameters()))
    assert maximum / scale < 3e-6
    first = [p.grad.detach().cpu().clone() for p in candidate.parameters()]
    bopt.zero_grad()
    with torch.autocast("mps", dtype=torch.bfloat16, enabled=precision == "bf16"):
        _, repeat = candidate(x, y)
    repeat.backward()
    synchronize(torch.device("mps"))
    assert all(torch.equal(g, p.grad.cpu()) for g,p in zip(first,candidate.parameters()))
    checks.append(dict(precision=precision, matching_forward_loss=True, initialization_unchanged=True,
        max_gradient_absolute_difference=maximum, gradient_scale=scale,
        gradient_scaled_difference=maximum/scale, repeated_full_gradients_bitwise_identical=True))
    del native, candidate, aopt, bopt, x, y, loss_a, loss_b, repeat, first
    torch.mps.empty_cache()

torch.use_deterministic_algorithms(True)
try:
    embedding = torch.nn.Embedding(5, 7).to("mps")
    embedding(torch.tensor([0,0,2], device="mps")).sum().backward()
    native_rejection = None
except RuntimeError as error:
    assert "embedding_dense_backward_mps" in str(error)
    native_rejection = str(error)
result = dict(status="passed", checks=checks, native_deterministic_rejection=native_rejection,
    native_strict_mode_probe_accepted=native_rejection is None,
    runtime_caveat="Installed binary accepts the small native embedding strict-mode probe. Current main source's nondeterminism alert must not be treated as installed-binary behavior. Repeated full-model gradients are checked directly.",
    torch_version=str(torch.__version__), fill_uninitialized_memory=False,
    scope="Full 3.35M model initial FP32/BF16 derivatives and repeated exact gradients, not a trained-quality claim.")
root = Path(__file__).resolve().parents[2]
(root / "artifacts/v3/verification-incidence.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
