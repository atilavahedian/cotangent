"""Finite-precision boundaries, model parity and explicit restriction checks."""
import gc
import json
import torch
from cotangent.runtime import Guard, batch, load_bytes, make_schedule, model_hash
from research.v3.embedding import install
from research.v4.bucketed import BucketedAdamW
from research.v6.models import make_model, CONFIGS
from research.v6.methods import Handler, bucket_norm, scalar_norm
from research.v6.runtime import ROOT, check_prior

def main():
    torch.set_num_threads(4)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required")
    output = ROOT / "artifacts/v6/verification.json"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    guard, boundaries, models, restrictions = Guard("mps"), [], [], []
    lengths = [1, 2, 3, 4, 5, 6, 8, 10, 31, 32, 33, 62, 64, 66, 126, 128, 130, 254, 256, 258, 510, 512, 514, 1022, 1024, 1026, 4094, 4096, 4098, 16384, 65536, 262144]
    torch.manual_seed(1701)
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        for length in lengths:
            for b in (1, 3, 8):
                gradients = [torch.randn(length, device="mps", dtype=dtype) for _ in range(b)]
                native = scalar_norm(gradients)
                other = bucket_norm(gradients, [list(range(b))])
                boundaries.append(dict(dtype=str(dtype), length=length, bucket_size=b, bitwise_equal=bool(torch.equal(native, other)),
                                       absolute_difference=float((native - other).abs().item()), fallback=length < 4 or length % 2 != 0))
    del gradients, native, other
    data = load_bytes(ROOT / "data/train.bin")
    training = data[:int(len(data) * .95)]
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    for name in CONFIGS:
        for bf16 in (False, True):
            for arm in ("bucket-native", "scalar-packed", "bucket-packed", "foreach-native"):
                native, other = make_model(name, 1702), make_model(name, 1702)
                install(native); install(other)
                optimizer = torch.optim.AdamW(native.parameters(), lr=.001, betas=(.9, .95), eps=1e-8, weight_decay=.1, fused=True, foreach=False)
                handler = Handler(other, arm)
                assert model_hash(native) == model_hash(other)
                schedule = make_schedule(len(training), 4, 8, 256, 1702)
                for step in range(4):
                    optimizer.zero_grad(set_to_none=True); handler.zero_grad()
                    x, y = batch(training, schedule[step], 256, "mps")
                    with torch.autocast("mps", dtype=torch.bfloat16, enabled=bf16):
                        _, loss = native(x, y); _, other_loss = other(x, y)
                    loss.backward(); other_loss.backward()
                    gradients_equal = all(torch.equal(a.grad, b.grad) for a, b in zip(native.parameters(), other.parameters()))
                    norm = torch.nn.utils.clip_grad_norm_(native.parameters(), 1., foreach=False)
                    optimizer.step(); other_norm = handler.step()
                    parameters_equal = all(torch.equal(a, b) for a, b in zip(native.parameters(), other.parameters()))
                    moments_equal, offset = True, 0
                    for a, b in zip(native.parameters(), other.parameters()):
                        for key in ("exp_avg", "exp_avg_sq"):
                            value = handler.optimizer.state[handler.wrapper.master][key][offset:offset + a.numel()].reshape_as(a) if handler.packed else handler.optimizer.state[b][key]
                            moments_equal &= bool(torch.equal(optimizer.state[a][key], value))
                        offset += a.numel()
                    models.append(dict(model=name, bf16=bf16, arm=arm, step=step + 1, loss_equal=bool(torch.equal(loss, other_loss)),
                                       gradients_equal=gradients_equal, norm_equal=bool(torch.equal(norm, other_norm)),
                                       parameters_equal=parameters_equal, moments_equal=moments_equal))
                    guard.check()
                del native, other, optimizer, handler, loss, other_loss, norm, other_norm, x, y, value, a, b
                gc.collect(); torch.mps.empty_cache()
    torch.use_deterministic_algorithms(False)
    # Verify the new combined treatment matches the previously frozen method.
    left, right = make_model("transformer-small", 1703), make_model("transformer-small", 1703)
    frozen, new = BucketedAdamW(left, fused=True, lr=.001, betas=(.9,.95), eps=1e-8, weight_decay=.1), Handler(right, "bucket-packed")
    for p, q in zip(left.parameters(), right.parameters()):
        p.grad = torch.randn_like(p); q.grad = p.grad.clone()
    old_norm, new_norm = frozen.clip_and_step(), new.step()
    frozen_match = bool(torch.equal(old_norm, new_norm)) and model_hash(left) == model_hash(right)
    # Reject missing gradients, mixed parameters and overlapping parameter objects.
    for kind in ("missing", "mixed", "overlap"):
        try:
            if kind == "overlap":
                module = torch.nn.Module(); storage = torch.ones(8)
                module.p = torch.nn.Parameter(storage[:6]); module.q = torch.nn.Parameter(storage[2:])
            elif kind == "mixed":
                module = torch.nn.Module(); module.p = torch.nn.Parameter(torch.ones(4)); module.q = torch.nn.Parameter(torch.ones(4, dtype=torch.float64))
            else:
                module = torch.nn.Linear(2, 2)
            h = Handler(module, "bucket-packed")
            h.step()
            restrictions.append(dict(case=kind, rejected=False))
        except (ValueError, RuntimeError) as exc:
            restrictions.append(dict(case=kind, rejected=True, reason=str(exc)))
    result = dict(fingerprints=check_prior(), torch=str(torch.__version__), torch_revision=torch.version.git_version,
                  model_checks=models, boundary_checks=boundaries, restriction_checks=restrictions, frozen_v4_update_match=frozen_match,
                  primary_dtype_boundary_pass=all(r["bitwise_equal"] for r in boundaries if r["dtype"] == "torch.float32"),
                  model_parity_pass=all(all(r[k] for k in ("loss_equal", "gradients_equal", "norm_equal", "parameters_equal", "moments_equal")) for r in models),
                  scope="Full-model checks use matched deterministic embedding derivatives in both arms to isolate update arithmetic; official training retains native derivatives. Boundary results are observations on the pinned backend, not universal proofs.", **guard.summary())
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("primary_dtype_boundary_pass", "model_parity_pass", "frozen_v4_update_match")}))

if __name__ == "__main__":
    main()
