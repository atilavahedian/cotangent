"""Serial matched language training, with immutable records and local weights."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback
import torch
from cotangent.runtime import Guard, batch, load_bytes, make_schedule, model_hash, schedule_hash, synchronize
from cotangent.train import evaluate
from research.v6.models import make_model, CONFIGS
from research.v6.methods import Handler, forward_callable, ARMS
from research.v6.runtime import ROOT, check_prior, digest

def run(model_name, arm, seed, steps, output, official=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    fp = check_prior()
    protocol_hash = None
    if official:
        protocol = ROOT / "artifacts/v6/frozen-study.json"
        frozen = json.loads(protocol.read_text())
        assert fp == frozen["fingerprints"]
        assert any(j["model"] == model_name and j["arm"] == arm and j["seed"] == seed and j["steps"] == steps for j in frozen["jobs"])
        protocol_hash = digest(protocol)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(False)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no fallback")
    model = make_model(model_name, seed)
    initial = model_hash(model)
    handler = Handler(model, arm)
    assert model_hash(model) == initial
    forward = forward_callable(model, arm)
    data = load_bytes(ROOT / "data/train.bin")
    training, diagnostic = data[:int(len(data) * .95)], data[int(len(data) * .95):]
    sequence = model.cfg.sequence
    schedule = make_schedule(len(training), steps, 8, sequence, seed)
    guard = Guard("mps")
    metadata = dict(model=model_name, config=CONFIGS[model_name], arm=arm, seed=seed, steps=steps, fingerprints=fp,
                    protocol_sha256=protocol_hash, initialization_sha256=initial, schedule_sha256=schedule_hash(schedule),
                    parameters=sum(p.numel() for p in model.parameters()), parameter_tensors=len(tuple(model.parameters())),
                    length_buckets=len(handler.buckets), torch=str(torch.__version__), torch_revision=torch.version.git_version,
                    official=official, dtype="BF16 autocast, FP32 weights/gradients", status="running")
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    durations, probes = [], []
    log = (output / "curve.jsonl").open("w", buffering=1)
    try:
        for _ in range(3):
            handler.zero_grad()
            x, y = batch(training, schedule[0], sequence, "mps")
            with torch.autocast("mps", dtype=torch.bfloat16):
                _, loss = forward(x, y)
            loss.backward()
        handler.zero_grad()
        synchronize("mps")
        setup = time.perf_counter() - start
        guard.check()
        for step in range(steps):
            t0 = time.perf_counter()
            handler.zero_grad()
            warm = min(100, max(20, steps // 10))
            ratio = min(1., (step + 1) / warm)
            if step >= warm:
                ratio = .1 + .9 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))) / 2
            handler.lr(.001 * ratio)
            x, y = batch(training, schedule[step], sequence, "mps")
            with torch.autocast("mps", dtype=torch.bfloat16):
                _, loss = forward(x, y)
            loss.backward()
            norm = handler.step()
            synchronize("mps")
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss")
            resource = guard.check() if step % 10 == 0 else {}
            value = float(loss.item()) / math.log(2) if step % 10 == 0 else None
            duration = time.perf_counter() - t0
            durations.append(duration)
            if step % 10 == 0:
                log.write(json.dumps(dict(step=step + 1, loss_bpb=value, norm=float(norm.item()), seconds=duration, **resource)) + "\n")
            if official and (step + 1) % 200 == 0:
                probe = evaluate(model, diagnostic, "mps", 8, sequence, "bf16", max_batches=32)
                probes.append(dict(step=step + 1, diagnostic_bpb=probe["bpb"], training_seconds=sum(durations), elapsed_seconds=time.perf_counter() - start))
                log.write(json.dumps(probes[-1]) + "\n")
        quality = evaluate(model, diagnostic, "mps", 8, sequence, "bf16", max_batches=32)
        result = dict(metadata, status="completed", setup_seconds=setup, training_seconds=sum(durations), step_seconds=durations,
                      warmed_training_seconds=sum(durations[20:]), diagnostic_bpb=quality["bpb"], diagnostic_targets=quality["tokens"],
                      probes=probes, final_model_sha256=model_hash(model), **guard.summary())
        if official:
            result["final_validation"] = evaluate(model, load_bytes(ROOT / "data/validation.bin"), "mps", 8, sequence, "bf16")
            result["final_test"] = evaluate(model, load_bytes(ROOT / "data/test.bin"), "mps", 8, sequence, "bf16")
            checkpoint = ROOT / "artifacts/checkpoints/v6" / output.name
            checkpoint = checkpoint.with_suffix(".pt")
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, model_name=model_name,
                            arm=arm, seed=seed, config=CONFIGS[model_name], final_model_sha256=result["final_model_sha256"]), checkpoint)
            result.update(checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_sha256=digest(checkpoint))
        synchronize("mps")
        result["total_seconds"] = time.perf_counter() - start
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("model", "arm", "seed", "training_seconds", "diagnostic_bpb")}), flush=True)
        return result
    except Exception as exc:
        result = dict(metadata, status="failed", error=repr(exc), traceback=traceback.format_exc(), training_seconds=sum(durations), **guard.summary())
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        raise
    finally:
        log.close()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=CONFIGS, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--official", action="store_true")
    a = p.parse_args()
    run(a.model, a.arm, a.seed, a.steps, a.output, a.official)
