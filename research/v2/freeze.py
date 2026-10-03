"""Freeze V2 before its final runs. Refuse replacing an existing protocol."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from cotangent.runtime import source_hash
from research.v2.pilot import source_hash as v2_source_hash

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "artifacts/v2/frozen-study.json"
assert not path.exists()
assert json.loads((ROOT / "artifacts/v2/verification-mps.json").read_text())["status"] == "passed"
pairs = [("small", s, i % 2) for i, s in enumerate([211, 223, 227, 229, 233, 239, 241])]
pairs += [("medium", s, i % 2) for i, s in enumerate([251, 257, 263, 269, 271])]
random.Random(20261003).shuffle(pairs)
jobs = [dict(size=size, seed=seed, arm=arm) for size, seed, reverse in pairs
        for arm in (["packed-fused", "native-fused"] if reverse else ["native-fused", "packed-fused"])]
configs = {}
for size in ("small", "medium"):
    config = json.loads((ROOT / f"configs/final-{size}.json").read_text())
    configs[size] = {key: config[key] for key in ("model", "steps", "learning_rate", "lr_warmup")}
protocol = dict(
    version=2, frozen_at=datetime.now(timezone.utc).isoformat(), source_sha256=v2_source_hash(),
    v1_source_sha256=source_hash(ROOT),
    data_manifest_sha256=hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest(),
    method="Exact gradients, global gradient clipping and AdamW reindexed into a contiguous vector",
    arms=["native-fused", "packed-fused"], jobs=jobs, configs=configs,
    primary_size="small", replication_size="medium", quality_target_bpb=3.2,
    meaningful_elapsed_reduction_fraction=.10, quality_noninferiority_margin_bpb=.01,
    decision="Small-model win requires all target crossings, mean paired elapsed reduction >=10%, its two-sided 95% Student-t interval above zero, and upper paired full-test BPB difference <=0.01. Larger-model gate evaluated separately with the same rules.",
    inference="Paired Student-t intervals, n=7 small and n=5 medium. Target crossing uses first scheduled 100-step/65,536-byte official validation probe; no interpolation or censored-time filling. All planned seeds included.",
    timing="Serial paired AB/BA order, shuffled pair blocks before training. Synchronize MPS every optimizer step. Training time includes data batching/transfer, forward, full backward, gradient concatenation if packed, clipping, fused AdamW, finite-loss checks and resource checks. Target elapsed time additionally includes setup, both initialization hash checks, kernel priming and all preceding validation probes. Python interpreter launch is outside the clock equally in both arms. Final full validation/test and local checkpoint saving occur after target crossings and are reported separately.",
    baseline_selection="Native fused AdamW chosen from training-only diagnostics against automatic single-tensor and forced foreach AdamW. Both final arms use the same native forward/backward, BF16 autocast, FP32 parameters, lr/betas/decay/epsilon, global clip=1, model, initialization, batches and fixed step budgets.",
    heldout_scope="WikiText-2 official validation/test were already evaluated by V1. V2 pilots use only the last 5% of training bytes; no official held-out result informed V2 packing or baseline selection. New frozen seeds provide a timing replication; test BPB is a reused-corpus numerical/quality regression check, not a fresh unseen-dataset generalization claim.",
    limitations=["One Apple M5 Pro, PyTorch 2.14.1 MPS, one corpus, two model sizes; desktop timing is not a controlled multi-device systems study.",
                 "Packing is an established systems technique; this is an authored implementation and measured application, not a claimed invention of parameter flattening or a new chain rule.",
                 "No CUDA, compiled-training, distributed-training, very large model or global state-of-the-art superiority claim.",
                 "Gradient/optimizer groups must be homogeneous and every parameter active. One additional dense gradient vector is allocated; activation memory and gradient matrix multiplications are unchanged.",
                 "Exactness means mathematical equivalence; floating-point reduction/optimizer order can change trained bytes and loss slightly."],
    publication="Code, protocol, measurements, figures, HTML and incremental commits public; all weights stay local as explicitly requested.")
path.write_text(json.dumps(protocol, indent=2) + "\n")
print(json.dumps(dict(jobs=len(jobs), source_sha256=protocol["source_sha256"], protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
