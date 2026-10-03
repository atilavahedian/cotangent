"""Freeze the third experiment before any V3 official evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from cotangent.runtime import source_hash
from research.v2.pilot import source_hash as v2_source_hash
from research.v3.pilot import source_hash as v3_source_hash

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "artifacts/v3/frozen-study.json"
assert not path.exists()
assert json.loads((ROOT / "artifacts/v3/verification-incidence.json").read_text())["status"] == "passed"
small = [307,311,313,317,331,337,347,349,353,359,367,373,379,383,389]
ablation = [307,331,353,373,389]
medium = [401,409,419]
blocks = [("small", seed, i % 2) for i,seed in enumerate(small)]
blocks += [("medium", seed, i % 2) for i,seed in enumerate(medium)]
random.Random(2026100303).shuffle(blocks)
jobs = []
for size,seed,reverse in blocks:
    arms = ["native-fused","incidence-packed-flatnorm"]
    if reverse:
        arms.reverse()
    if size == "small" and seed in ablation:
        arms.insert(ablation.index(seed) % 3,"incidence-fused")
    jobs.extend(dict(size=size,seed=seed,arm=arm) for arm in arms)
configs = {}
for size in ("small","medium"):
    c = json.loads((ROOT/f"configs/final-{size}.json").read_text())
    configs[size] = {key:c[key] for key in ("model","steps","learning_rate","lr_warmup")}
protocol = dict(version=3,frozen_at=datetime.now(timezone.utc).isoformat(),
    source_sha256=v3_source_hash(),v2_source_sha256=v2_source_hash(),v1_source_sha256=source_hash(ROOT),
    data_manifest_sha256=hashlib.sha256((ROOT/"data/manifest.json").read_bytes()).hexdigest(),
    configs=configs,jobs=jobs,primary_seeds=small,ablation_seeds=ablation,medium_seeds=medium,
    primary_size="small",primary_control="native-fused",candidate="incidence-packed-flatnorm",
    quality_target_bpb=3.2,meaningful_elapsed_reduction_fraction=.10,quality_noninferiority_margin_bpb=.01,
    decision="Primary win requires all 15 paired target crossings, mean elapsed reduction >=10%, its two-sided 95% Student-t interval above zero, and upper paired full-test BPB difference <=0.01. All 15 pairs included. Five-seed incidence-fused ablation and three-seed larger model are descriptive, not alternate routes to declare primary success.",
    inference="Two-sided Student-t paired intervals: n=15 primary, n=5 ablation, n=3 medium. First scheduled 100-step/65,536-byte probe <=3.2 BPB defines the target crossing. No interpolation, imputation or seed removal. No optional stopping.",
    method="Exact FP32 embedding cotangents computed with a byte incidence matrix, exact position-prefix gradient, full native linear/attention backward, then global clipping and fused AdamW on a packed full-gradient vector. No sampled, low-rank, skipped or BF16-rounded embedding gradients.",
    timing="Same serial synchronized timing harness as V2, including data, forward, every derivative, incidence construction/GEMM, concatenation, clipping, AdamW, host finite checks and resources. Target elapsed additionally includes setup, two model-hash checks, kernel priming and preceding probes. Python process launch excluded equally. Pair blocks shuffled; AB/BA order balanced; five ablations inserted at predeclared rotated positions. Evaluation and checkpoint saving separately reported.",
    baseline_selection="Primary existing baseline is unmodified native BF16 autograd with native fused AdamW, selected by earlier training-only optimizer pilots. Both arms use identical forward function, architecture, parameter count, FP32 weights, batches, lr/betas/decay/epsilon and clip=1. Five deterministic-incidence/fused-AdamW control runs isolate the packing effect on selected predeclared seeds.",
    deterministic_configuration="Candidate and incidence-fused control request strict deterministic algorithms. All arms set fill_uninitialized_memory=False: kernels/standard operations fully initialize outputs before use. This removes redundant initialization work, as permitted by PyTorch's reproducibility guidance. Installed native embedding accepts a strict-mode probe despite atomic accumulation, so determinism is directly verified by derivative and full-training repeats, not inferred from the flag alone.",
    heldout_scope="The V3 hypothesis follows failed V2 gates and its native-repeat diagnosis. V3 implementation/performance tuning uses only training bytes and the last 5% diagnostic holdout. WikiText-2 official validation/test have already been evaluated in earlier studies. They are reused openly for a model-quality regression check; new frozen seeds replicate timing and quality on this same corpus, not fresh-dataset generalization.",
    limitations=["One Apple M5 Pro, PyTorch 2.14.1 MPS, one byte-level corpus; no CUDA, compiled/distributed training or large-scale superiority claim.",
        "Incidence-matrix differentiation, parameter packing and fused AdamW are established ideas. Contribution is authored tested code, diagnosis, backend measurements and reproducible evidence, not their invention.",
        "Dense incidence arithmetic scales with vocabulary size; the tested vocabulary has 256 bytes. This is not a demonstrated optimization for a 50k-token vocabulary.",
        "Additional incidence and packed-gradient buffers are allocated. Full activations and all transformer linear derivative products remain.",
        "Mathematical equivalence does not guarantee bitwise equivalence with native atomic embedding reduction or a different norm tree; final quality must pass the independent frozen gate.",
        "Desktop timing and small secondary sample sizes limit external validity; reused official corpus is disclosed."],
    publication="Only code, protocols, records, figures and HTML are public. Every weight file remains local.")
path.write_text(json.dumps(protocol,indent=2)+"\n")
print(json.dumps(dict(jobs=len(jobs),source_sha256=protocol["source_sha256"],protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
