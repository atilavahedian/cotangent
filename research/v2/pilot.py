"""Training-only diagnostics. V1 official held-out splits are never opened."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer
from cotangent.runtime import Guard, batch, load_bytes, make_schedule, model_hash, schedule_hash, synchronize
from cotangent.train import evaluate
from research.v2.packed import PackedAdamW

ROOT = Path(__file__).resolve().parents[2]


def source_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "research/v2").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def run(arm, seed, steps, size, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    torch.manual_seed(seed)
    cfg = ModelConfig() if size == "small" else ModelConfig(width=512, layers=6, heads=8)
    model = Transformer(cfg, Policy(mode="native")).to(device)
    initial = model_hash(model)
    opts = dict(lr=.001, betas=(.9, .95), weight_decay=.1)
    packed = arm.startswith("packed")
    fused = arm.endswith("fused")
    optimizer = PackedAdamW(model, fused=fused, **opts) if packed else torch.optim.AdamW(model.parameters(), fused=fused, foreach=False, **opts)
    assert model_hash(model) == initial
    data = load_bytes(ROOT / "data/train.bin")
    cutoff = int(len(data) * .95)
    training, diagnostic = data[:cutoff], data[cutoff:]
    schedule = make_schedule(len(training), steps, 8, cfg.sequence, seed)
    guard = Guard(device)
    times, losses = [], []
    metadata = dict(arm=arm, seed=seed, steps=steps, size=size, source_sha256=source_hash(),
                    initialization_sha256=initial, schedule_sha256=schedule_hash(schedule),
                    data_scope="V1 training bytes only; last 5 percent diagnostic holdout",
                    status="running", packed=packed, fused=fused)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    # Prime without updates so AdamW state initialization remains in measured step 1.
    for _ in range(3):
        x, y = batch(training, schedule[0], cfg.sequence, device)
        with torch.autocast("mps", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.zero_grad() if packed else optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    begin = time.perf_counter()
    for step in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad() if packed else optimizer.zero_grad(set_to_none=True)
        ratio = min(1., (step + 1) / 50)
        if step >= 50:
            ratio = .1 + .9 * (1 + math.cos(math.pi * (step - 50) / max(1, steps - 50))) / 2
        for group in optimizer.param_groups:
            group["lr"] = .001 * ratio
        x, y = batch(training, schedule[step], cfg.sequence, device)
        with torch.autocast("mps", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        if packed:
            optimizer.clip_and_step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False)
            optimizer.step()
        synchronize(device)
        times.append(time.perf_counter() - t0)
        if step % 10 == 0:
            guard.check()
        if step % 50 == 0:
            losses.append(float(loss.item()) / math.log(2))
        if not torch.isfinite(loss).item():
            raise RuntimeError("Nonfinite loss")
    wall = time.perf_counter() - begin
    evaluation = evaluate(model, diagnostic, device, 8, cfg.sequence, "bf16", max_batches=32)
    result = dict(metadata, status="completed", training_seconds=sum(times), loop_wall_seconds=wall,
                  step_seconds=times, diagnostic_bpb=evaluation["bpb"], train_loss_probes=losses,
                  final_model_sha256=model_hash(model), **guard.summary())
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("arm", "seed", "size", "training_seconds", "diagnostic_bpb")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["native-auto", "native-fused", "packed-auto", "packed-fused"], required=True)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--size", choices=["small", "medium"], default="small")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.arm, a.seed, a.steps, a.size, a.output)
