"""Frozen deterministic-incidence and packed-gradient comparison."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch

from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer
from cotangent.runtime import Guard, batch, load_bytes, make_schedule, model_hash, schedule_hash, source_hash, synchronize
from cotangent.train import evaluate
from research.v3.pilot import build, source_hash as v3_source_hash
from research.v2.pilot import source_hash as v2_source_hash

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "artifacts/v3/frozen-study.json"


def run(arm, seed, size):
    frozen = json.loads(PROTOCOL.read_text())
    assert v3_source_hash() == frozen["source_sha256"]
    assert v2_source_hash() == frozen["v2_source_sha256"]
    assert source_hash(ROOT) == frozen["v1_source_sha256"]
    assert hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest() == frozen["data_manifest_sha256"]
    job = next(j for j in frozen["jobs"] if (j["arm"], j["seed"], j["size"]) == (arm, seed, size))
    config = frozen["configs"][size]
    output = ROOT / "artifacts/v3/final" / size / f"{arm}-{seed}"
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device("mps")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS required; no silent fallback")
    model, optimizer, initial, packed = build(ModelConfig(**config["model"]), seed, arm)
    data = load_bytes(ROOT / "data/train.bin")
    training = data[:int(len(data) * .95)]
    validation = load_bytes(ROOT / "data/validation.bin")
    schedule = make_schedule(len(training), config["steps"], 8, model.cfg.sequence, seed)
    guard = Guard(device)
    meta = {"arm": arm, "seed": seed, "size": size, "config": config, "status": "running",
            "source_sha256": frozen["source_sha256"], "v1_source_sha256": frozen["v1_source_sha256"],
            "v2_source_sha256": frozen["v2_source_sha256"],
            "deterministic_required": arm != "native-fused",
            "deterministic_fill_uninitialized_memory": False,
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
            "data_manifest_sha256": frozen["data_manifest_sha256"],
            "initialization_sha256": initial, "schedule_sha256": schedule_hash(schedule),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "tokens_budget": config["steps"] * 8 * model.cfg.sequence,
            "training_bytes_available": len(training), "torch_version": str(torch.__version__),
            "timing": frozen["timing"], "baseline_selection": frozen["baseline_selection"],
            "heldout_scope": frozen["heldout_scope"]}
    (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    for _ in range(3):
        x, y = batch(training, schedule[0], model.cfg.sequence, device)
        with torch.autocast("mps", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss.backward()
        optimizer.zero_grad() if packed else optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    setup = time.perf_counter() - start
    records, step_times = [], []
    training_seconds, evaluation_seconds = 0., 0.
    log = (output / "curve.jsonl").open("w", buffering=1)
    try:
        for step in range(config["steps"]):
            t0 = time.perf_counter()
            optimizer.zero_grad() if packed else optimizer.zero_grad(set_to_none=True)
            warm = config["lr_warmup"]
            ratio = min(1., (step + 1) / warm)
            if step >= warm:
                ratio = .1 + .9 * (1 + math.cos(math.pi * (step - warm) / max(1, config["steps"] - warm))) / 2
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * ratio
            x, y = batch(training, schedule[step], model.cfg.sequence, device)
            with torch.autocast("mps", dtype=torch.bfloat16):
                _, loss = model(x, y)
            loss.backward()
            if packed:
                norm = optimizer.clip_and_step(1.)
            else:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=False)
                optimizer.step()
            synchronize(device)
            # Include the same host checks and telemetry in measured training.
            if not torch.isfinite(loss).item():
                raise RuntimeError("Nonfinite loss")
            peak = guard.check() if step % 10 == 0 else {}
            loss_bpb = float(loss.item()) / math.log(2) if step % 10 == 0 else None
            norm_value = float(norm.item()) if step % 10 == 0 else None
            duration = time.perf_counter() - t0
            step_times.append(duration)
            training_seconds += duration
            if step % 10 == 0:
                log.write(json.dumps(dict(step=step + 1, loss_bpb=loss_bpb, gradient_norm=norm_value,
                                          step_seconds=duration, training_seconds=training_seconds, **peak)) + "\n")
            if (step + 1) % 100 == 0:
                t_eval = time.perf_counter()
                metric = evaluate(model, validation, device, 8, model.cfg.sequence, "bf16", max_batches=32)
                synchronize(device)
                evaluation_seconds += time.perf_counter() - t_eval
                record = dict(step=step + 1, validation_bpb=metric["bpb"], validation_tokens=metric["tokens"],
                              training_seconds=training_seconds, elapsed_seconds=time.perf_counter() - start)
                records.append(record)
                log.write(json.dumps(record) + "\n")
                print(json.dumps(dict(arm=arm, seed=seed, size=size, **record)), flush=True)
        t_eval = time.perf_counter()
        final_validation = evaluate(model, validation, device, 8, model.cfg.sequence, "bf16")
        final_test = evaluate(model, load_bytes(ROOT / "data/test.bin"), device, 8, model.cfg.sequence, "bf16")
        synchronize(device)
        evaluation_seconds += time.perf_counter() - t_eval
        evaluation_completed_seconds = time.perf_counter() - start
        checkpoint = ROOT / "artifacts/checkpoints/v3" / size / f"{arm}-{seed}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                        model_config=asdict(model.cfg), arm=arm, seed=seed, source_sha256=meta["source_sha256"]), checkpoint)
        result = dict(meta, status="completed", setup_seconds=setup, training_seconds=training_seconds,
                      step_seconds=step_times, evaluation_seconds=evaluation_seconds,
                      evaluation_completed_seconds=evaluation_completed_seconds,
                      total_elapsed_seconds=time.perf_counter() - start, validation_curve=records,
                      final_validation=final_validation, final_test=final_test,
                      final_model_sha256=model_hash(model), checkpoint=str(checkpoint.relative_to(ROOT)),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), **guard.summary())
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    except Exception as exc:
        (output / "summary.json").write_text(json.dumps(dict(meta, status="failed", error=repr(exc),
            traceback=traceback.format_exc(), training_seconds=training_seconds, **guard.summary()), indent=2))
        raise
    finally:
        log.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=["native-fused", "incidence-fused", "incidence-packed-flatnorm"], required=True)
    p.add_argument("--size", choices=["small", "medium"], required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    run(a.arm, a.seed, a.size)
