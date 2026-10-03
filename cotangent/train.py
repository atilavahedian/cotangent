"""Matched transformer training with immutable per-run evidence."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .backward import Policy
from .model import ModelConfig, Transformer
from .runtime import Guard, batch, load_bytes, make_schedule, model_hash, schedule_hash, source_hash, synchronize

ROOT = Path(__file__).resolve().parents[1]


def autocast(device, precision):
    return torch.autocast(str(device), dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


@torch.no_grad()
def evaluate(model, data, device, batch_size, sequence, precision, max_batches=None):
    """Contiguous fixed-shape windows; pad only the final batch and mask labels."""
    model.eval()
    nll, count = 0.0, 0
    stride = batch_size * sequence
    for j, start in enumerate(range(0, len(data) - 1, stride)):
        if max_batches is not None and j >= max_batches:
            break
        x = np.zeros((batch_size, sequence), dtype=np.int64)
        y = np.full((batch_size, sequence), -100, dtype=np.int64)
        for i in range(batch_size):
            offset = start + i * sequence
            length = min(sequence, len(data) - offset - 1)
            if length > 0:
                x[i, :length] = data[offset:offset + length]
                y[i, :length] = data[offset + 1:offset + length + 1]
        target = torch.from_numpy(y).to(device)
        with autocast(device, precision):
            _, loss = model(torch.from_numpy(x).to(device), target)
        valid = int((y != -100).sum())
        nll += float(loss.item()) * valid
        count += valid
    model.train()
    return {"bpb": nll / count / math.log(2), "nll_nats": nll / count, "tokens": count}


def train_run(config, mode, seed, output, phase="final", steps_override=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    clock_start = time.perf_counter()
    torch.set_num_threads(config.get("cpu_threads", 4))
    device = torch.device(config.get("device", "mps"))
    if str(device) == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS backend is unavailable; refusing silent CPU substitution")
    precision = config.get("precision", "fp32")
    cfg = ModelConfig(**config["model"])
    steps = steps_override or config["steps"]
    policy = Policy(mode=mode, **config.get("policy", {}))
    torch.manual_seed(seed)
    model = Transformer(cfg, policy).to(device)
    initial_hash = model_hash(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  betas=(0.9, 0.95), weight_decay=0.1, foreach=None)
    train = load_bytes(ROOT / "data/train.bin")
    cutoff = int(len(train) * 0.95)
    training = train[:cutoff]
    validation = train[cutoff:] if phase == "pilot" else load_bytes(ROOT / "data/validation.bin")
    schedule = make_schedule(len(training), steps, config["batch_size"], cfg.sequence, seed)
    guard = Guard(device)
    metadata = {"mode": mode, "seed": seed, "phase": phase, "config": config, "steps": steps,
                "parameter_count": sum(p.numel() for p in model.parameters()),
                "initialization_sha256": initial_hash, "schedule_sha256": schedule_hash(schedule),
                "source_sha256": source_hash(ROOT), "data_manifest_sha256": hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest(),
                "train_bytes_available": len(training), "diagnostic_holdout_bytes": len(train) - cutoff,
                "tokens_budget": steps * config["batch_size"] * cfg.sequence,
                "timing": "synchronized every step; data transfer, sampling, audits, gradients, clipping and AdamW included; setup and evaluation separately measured and included in elapsed clock",
                "status": "running"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    records = []
    sum_training = 0.0
    sum_eval = 0.0
    peak = {}
    # Prime kernels without weight updates, then clear gradients and policy state.
    # This shared startup is charged to setup and the elapsed clock.
    for _ in range(3):
        x, y = batch(training, schedule[0], cfg.sequence, device)
        policy.start_step(0)
        with autocast(device, precision):
            _, loss = model(x, y)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
    policy.states.clear()
    torch.manual_seed(seed + 200000)  # sampling RNG, separate from initialization and data schedule
    synchronize(device)
    setup = time.perf_counter() - clock_start
    log = (output / "curve.jsonl").open("w", buffering=1)
    try:
        for step in range(steps):
            t0 = time.perf_counter()
            policy.start_step(step)
            x, y = batch(training, schedule[step], cfg.sequence, device)
            # The schedule depends only on step, shared across arms.
            warmup = config.get("lr_warmup", 50)
            ratio = min(1.0, (step + 1) / warmup)
            if step >= warmup:
                progress = (step - warmup) / max(1, steps - warmup)
                ratio = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
            lr = config["learning_rate"] * ratio
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(device, precision):
                _, loss = model(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=None)
            optimizer.step()
            policy.finish_step()
            synchronize(device)
            duration = time.perf_counter() - t0
            sum_training += duration
            if step % 10 == 0:
                peak = guard.check()
            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite training loss")
            if (step + 1) % config.get("log_every", 10) == 0 or step == 0:
                record = {"step": step + 1, "tokens": (step + 1) * config["batch_size"] * cfg.sequence,
                          "loss_bpb": float(loss.item()) / math.log(2), "gradient_norm": float(norm.item()),
                          "step_seconds": duration, "training_seconds": sum_training, "learning_rate": lr,
                          "elapsed_seconds": time.perf_counter() - clock_start, **policy.aggregate(), **peak}
                log.write(json.dumps(record) + "\n")
            if (step + 1) % config.get("eval_every", 100) == 0 or step + 1 == steps:
                t_eval = time.perf_counter()
                metric = evaluate(model, validation, device, config["batch_size"], cfg.sequence,
                                  precision, config.get("eval_batches", 16))
                synchronize(device)
                sum_eval += time.perf_counter() - t_eval
                record = {"step": step + 1, "tokens": (step + 1) * config["batch_size"] * cfg.sequence,
                          "validation_bpb": metric["bpb"], "validation_tokens": metric["tokens"],
                          "training_seconds": sum_training, "elapsed_seconds": time.perf_counter() - clock_start,
                          **policy.aggregate(), **guard.check()}
                records.append(record)
                log.write(json.dumps(record) + "\n")
                print(json.dumps({"mode": mode, "seed": seed, "step": step + 1,
                                  "validation_bpb": round(metric["bpb"], 4),
                                  "training_seconds": round(sum_training, 2),
                                  "sample_fraction": record["sample_fraction"]}), flush=True)
        final_validation = None
        final_test = None
        if phase == "final":
            t_eval = time.perf_counter()
            final_validation = evaluate(model, validation, device, config["batch_size"], cfg.sequence, precision)
            # The official test split is evaluated only after the frozen training completes.
            final_test = evaluate(model, load_bytes(ROOT / "data/test.bin"), device,
                                  config["batch_size"], cfg.sequence, precision)
            synchronize(device)
            sum_eval += time.perf_counter() - t_eval
        checkpoint_dir = ROOT / "artifacts/checkpoints" / phase / output.parent.name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / f"{output.name}.pt"
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "model_config": asdict(cfg), "mode": mode, "seed": seed,
                    "data_manifest_sha256": metadata["data_manifest_sha256"],
                    "source_sha256": metadata["source_sha256"]}, checkpoint)
        result = {**metadata, "status": "completed", "setup_seconds": setup,
                  "training_seconds": sum_training, "evaluation_seconds": sum_eval,
                  "total_elapsed_seconds": time.perf_counter() - clock_start,
                  "validation_curve": records, "final_validation": final_validation, "final_test": final_test,
                  "checkpoint": str(checkpoint.relative_to(ROOT)),
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  "final_model_sha256": model_hash(model), **guard.summary()}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    except Exception as exc:
        result = {**metadata, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc(),
                  "training_seconds": sum_training, "evaluation_seconds": sum_eval,
                  "total_elapsed_seconds": time.perf_counter() - clock_start, **guard.summary()}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        torch.save({"model": model.cpu().state_dict(), "step": policy.step, "reason": repr(exc)}, output / "partial.pt")
        raise
    finally:
        log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", required=True, choices=["native", "exact_custom", "wht", "uniform", "importance", "adaptive"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["pilot", "final"], default="final")
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    train_run(json.loads(Path(args.config).read_text()), args.mode, args.seed,
              args.output, args.phase, args.steps)


if __name__ == "__main__":
    main()
