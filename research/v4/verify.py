"""Actual MPS gradients, native norms, clipped updates and AdamW moments."""
import json
from pathlib import Path
import torch
from cotangent.model import ModelConfig
from cotangent.runtime import batch, load_bytes, make_schedule, model_hash
from research.v4.pilot import build

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
data = load_bytes(ROOT / "data/train.bin")
training = data[:int(len(data)*.95)]
records = []
for size, cfg in (("small",ModelConfig()), ("medium",ModelConfig(width=512,layers=6,heads=8))):
    for precision in (torch.float32, torch.bfloat16):
        native, native_opt, initial, _ = build(cfg, 109, "incidence-fused")
        packed, packed_opt, other, _ = build(cfg, 109, "incidence-bucket")
        assert initial == other
        schedule = make_schedule(len(training),8,8,256,109)
        for step in range(8):
            native_opt.zero_grad(set_to_none=True); packed_opt.zero_grad()
            x,y = batch(training,schedule[step],256,"mps")
            with torch.autocast("mps",dtype=torch.bfloat16,enabled=precision==torch.bfloat16):
                _, native_loss = native(x,y); _, packed_loss = packed(x,y)
            assert torch.equal(native_loss,packed_loss)
            native_loss.backward(); packed_loss.backward()
            assert all(torch.equal(a.grad,b.grad) for a,b in zip(native.parameters(),packed.parameters()))
            norm = torch.nn.utils.clip_grad_norm_(native.parameters(),1.,foreach=False)
            native_opt.step(); other_norm = packed_opt.clip_and_step(1.)
            assert torch.equal(norm,other_norm), (size,str(precision),step,"norm")
            assert all(torch.equal(a,b) for a,b in zip(native.parameters(),packed.parameters()))
            offset=0
            for parameter in native.parameters():
                for key in ("exp_avg","exp_avg_sq"):
                    flat=packed_opt.optimizer.state[packed_opt.master][key]
                    assert torch.equal(native_opt.state[parameter][key],flat[offset:offset+parameter.numel()].reshape_as(parameter))
                offset += parameter.numel()
            records.append(dict(size=size,precision=str(precision),step=step+1,
                forward_gradient_norm_parameter_moments_bitwise_equal=True))
        assert model_hash(native)==model_hash(packed)
        del native,packed,native_opt,packed_opt
        torch.mps.empty_cache()
result=dict(status="passed",checks=len(records),checks_detail=records,
    scope="Native full-model MPS FP32/BF16 gradients, eight consecutive native-versus-bucket norm/clipped-update/moment checks per size and precision. Both arms use deterministic exact embedding cotangents solely to control this numerical verification; primary training retains native embeddings.",
    torch_version=str(torch.__version__),torch_revision=torch.version.git_version)
(ROOT/"artifacts/v4/verification-mps.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(dict(status="passed",checks=len(records))))
