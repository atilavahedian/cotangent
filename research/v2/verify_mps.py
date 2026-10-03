"""Actual-model derivative, aliasing and first-update check on the target backend."""
import copy
import json
from pathlib import Path

import torch

from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer
from cotangent.runtime import model_hash, synchronize
from research.v2.packed import PackedAdamW

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
checks = []
for precision in ("fp32", "bf16"):
    torch.manual_seed(303)
    native = Transformer(ModelConfig(), Policy(mode="native")).to("mps")
    candidate = copy.deepcopy(native)
    initial = model_hash(candidate)
    opts = dict(lr=.001, betas=(.9, .95), weight_decay=.1, eps=1e-8)
    baseline = torch.optim.AdamW(native.parameters(), fused=True, foreach=False, **opts)
    packed = PackedAdamW(candidate, fused=True, **opts)
    assert model_hash(candidate) == initial == model_hash(native)
    x = torch.randint(256, (8, 256), device="mps")
    y = torch.randint(256, (8, 256), device="mps")
    with torch.autocast("mps", dtype=torch.bfloat16, enabled=precision == "bf16"):
        _, a = native(x, y)
        _, b = candidate(x, y)
    a.backward()
    b.backward()
    max_grad = max(float((p.grad - q.grad).abs().max().item()) for p, q in zip(native.parameters(), candidate.parameters()))
    gradient_scale = max(float(p.grad.abs().max().item()) for p in native.parameters())
    norm = torch.nn.utils.clip_grad_norm_(native.parameters(), 1., foreach=False)
    baseline.step()
    packed_norm = packed.clip_and_step()
    synchronize(torch.device("mps"))
    max_weight = max(float((p - q).abs().max().item()) for p, q in zip(native.parameters(), candidate.parameters()))
    norm_relative = abs(float(norm.item()) - float(packed_norm.item())) / float(norm.item())
    # Backend reductions need numerical equivalence, not a bitwise promise.
    assert max_grad / gradient_scale < 2e-6, (max_grad, gradient_scale)
    assert max_weight < 2e-7, max_weight
    assert norm_relative < 2e-6, norm_relative
    checks.append(dict(precision=precision, initialization_unchanged=True, max_gradient_absolute_difference=max_grad,
                       gradient_max_scale=gradient_scale, gradient_scaled_absolute_difference=max_grad / gradient_scale,
                       max_update_absolute_difference=max_weight, clipping_norm_relative_difference=norm_relative))
    del native, candidate, baseline, packed, x, y, a, b
    torch.mps.empty_cache()
result = dict(status="passed", checks=checks, torch_version=str(torch.__version__),
              scope="One full 3.35M-parameter transformer step per precision; CPU tests verify eight-step moments and updates.")
(ROOT / "artifacts/v2").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts/v2/verification-mps.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
