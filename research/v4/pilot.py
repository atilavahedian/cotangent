"""Training-only pilots for numerically matched bucketed clipping."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch
import torch.utils.deterministic

from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer
from cotangent.runtime import Guard, batch, load_bytes, make_schedule, model_hash, schedule_hash, synchronize
from cotangent.train import evaluate
from research.v2.packed import PackedAdamW
from research.v3.embedding import install
from research.v4.bucketed import BucketedAdamW
from research.v3.segmented import install as install_segmented
from research.v3.presorted import install as install_presorted, batch as presorted_batch

ROOT = Path(__file__).resolve().parents[2]


def source_hash():
    h = hashlib.sha256()
    for path in sorted((ROOT / "research/v4").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def build(cfg, seed, arm):
    torch.manual_seed(seed)
    model = Transformer(cfg, Policy(mode="native")).to("mps")
    before = model_hash(model)
    incidence = "incidence" in arm
    segmented = "segmented" in arm
    presorted = "presorted" in arm
    torch.use_deterministic_algorithms(incidence or segmented or presorted)
    # All model operations and our kernels fully initialize their outputs before
    # they are read. Avoid deterministic mode's redundant NaN fills, explicitly
    # permitted by PyTorch's reproducibility guidance for valid programs.
    torch.utils.deterministic.fill_uninitialized_memory = False
    if incidence:
        install(model)
    if segmented:
        install_segmented(model)
    if presorted:
        install_presorted(model)
    opts = dict(lr=.001, betas=(.9, .95), weight_decay=.1)
    packed = "bucket" in arm
    if packed:
        optimizer = BucketedAdamW(model, fused=True, **opts)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, foreach=False, **opts)
    assert model_hash(model) == before
    return model, optimizer, before, packed


def run(arm, seed, steps, size, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("mps")
    cfg = ModelConfig() if size == "small" else ModelConfig(width=512, layers=6, heads=8)
    model, optimizer, initial, packed = build(cfg, seed, arm)
    data = load_bytes(ROOT / "data/train.bin")
    cutoff = int(len(data)*.95)
    train, holdout = data[:cutoff], data[cutoff:]
    schedule = make_schedule(len(train), steps, 8, cfg.sequence, seed)
    guard = Guard(device)
    metadata = dict(arm=arm, seed=seed, size=size, steps=steps, source_sha256=source_hash(),
        initialization_sha256=initial, schedule_sha256=schedule_hash(schedule),
        data_scope="V1 training bytes only, last 5% diagnostic holdout", deterministic_required=any(k in arm for k in ("incidence","segmented","presorted")), deterministic_fill_uninitialized_memory=False)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    seconds = []
    try:
        for _ in range(3):
            x, y = presorted_batch(model, train, schedule[0], cfg.sequence, device) if "presorted" in arm else batch(train, schedule[0], cfg.sequence, device)
            with torch.autocast("mps", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            optimizer.zero_grad() if packed else optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        for step in range(steps):
            start = time.perf_counter()
            optimizer.zero_grad() if packed else optimizer.zero_grad(set_to_none=True)
            ratio = min(1., (step+1)/50)
            if step >= 50:
                ratio = .1 + .9*(1+math.cos(math.pi*(step-50)/max(1, steps-50)))/2
            for group in optimizer.param_groups:
                group["lr"] = .001*ratio
            x, y = presorted_batch(model, train, schedule[step], cfg.sequence, device) if "presorted" in arm else batch(train, schedule[step], cfg.sequence, device)
            with torch.autocast("mps", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            if packed:
                optimizer.clip_and_step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False)
                optimizer.step()
            synchronize(device)
            seconds.append(time.perf_counter()-start)
            if step%10 == 0:
                guard.check()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss")
        quality = evaluate(model, holdout, device, 8, cfg.sequence, "bf16", max_batches=32)
        result = dict(metadata, status="completed", training_seconds=sum(seconds), step_seconds=seconds,
                       diagnostic_bpb=quality["bpb"], final_model_sha256=model_hash(model), **guard.summary())
        (output / "summary.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({k: result[k] for k in ("arm","seed","size","training_seconds","diagnostic_bpb")}), flush=True)
    except Exception as exc:
        (output / "summary.json").write_text(json.dumps(dict(metadata, status="failed", error=repr(exc),
            traceback=traceback.format_exc(), training_seconds=sum(seconds)), indent=2))
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=["native-fused","native-bucket","incidence-fused","incidence-bucket"])
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--size", choices=["small","medium"], default="small")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    run(a.arm,a.seed,a.steps,a.size,a.output)
