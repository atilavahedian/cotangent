"""Freeze native-backward norm batching before any V4 official evaluation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from cotangent.runtime import source_hash
from research.v2.pilot import source_hash as v2_source_hash
from research.v3.pilot import source_hash as v3_source_hash
from research.v4.pilot import source_hash as v4_source_hash

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/"artifacts/v4/frozen-study.json"
assert not path.exists()
assert json.loads((ROOT/"artifacts/v4/verification-mps.json").read_text())["status"]=="passed"
assert all(r["deterministic_final_weights_identical"] for r in json.loads((ROOT/"artifacts/v4/pilot-comparison.json").read_text())["rows"])
primary=[503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659]
deterministic_small=[701,709,719,727,733]
medium=[751,757,761]
deterministic_medium=[809,811]
blocks=[("small",seed,"primary",i%2) for i,seed in enumerate(primary)]
blocks += [("small",seed,"equivalence",i%2) for i,seed in enumerate(deterministic_small)]
blocks += [("medium",seed,"descriptive",i%2) for i,seed in enumerate(medium)]
blocks += [("medium",seed,"equivalence",i%2) for i,seed in enumerate(deterministic_medium)]
random.Random(2026100304).shuffle(blocks)
jobs=[]
for size,seed,comparison,reverse in blocks:
    arms=["incidence-fused","incidence-bucket"] if comparison=="equivalence" else ["native-fused","native-bucket"]
    if reverse: arms.reverse()
    jobs.extend(dict(size=size,seed=seed,arm=arm,comparison=comparison) for arm in arms)
configs={}
for size in ("small","medium"):
    c=json.loads((ROOT/f"configs/final-{size}.json").read_text())
    configs[size]={key:c[key] for key in ("model","steps","learning_rate","lr_warmup")}
frozen=dict(version=4,frozen_at=datetime.now(timezone.utc).isoformat(),
    source_sha256=v4_source_hash(),v3_source_sha256=v3_source_hash(),v2_source_sha256=v2_source_hash(),v1_source_sha256=source_hash(ROOT),
    data_manifest_sha256=hashlib.sha256((ROOT/"data/manifest.json").read_bytes()).hexdigest(),
    configs=configs,jobs=jobs,primary_seeds=primary,medium_seeds=medium,
    equivalence_seeds=dict(small=deterministic_small,medium=deterministic_medium),
    primary_size="small",primary_control="native-fused",candidate="native-bucket",
    quality_target_bpb=3.2,meaningful_elapsed_reduction_fraction=.10,quality_noninferiority_margin_bpb=.01,
    confidence_level=.99,
    decision="All 25 primary pairs must cross the target. Mean paired target elapsed saving >=10%, its two-sided 99% Student-t lower bound >0, and the upper paired full-test difference <=0.01 BPB. Additionally all seven deterministic-gradient equivalence pairs must have identical final parameter hashes and full-test scores. All planned runs included; no optional stopping, seed removal, threshold relaxation or descriptive substitution. Failure remains failure.",
    inference="Primary two-sided 99% paired Student-t intervals, n=25 (df24 critical 2.796939504772804). More conservative than earlier 95% intervals after iterative development. Larger-model n=3 and deterministic-control comparisons are descriptive. First scheduled 100-step/65,536-byte probe <=3.2 BPB defines target, no interpolation or imputation. New source/method and new seeds; this is not an extension of a failed study.",
    method="Full native forward and backward, including native embeddings. Equal-length completed dense gradients are stacked into size buckets and viewed with two nontrivial reduced axes. This avoids MPS's different single-axis inner norm fast path and invokes the same generic reduction, element order and thread count as native per-parameter scalar norms. Scalar norms are reordered into native parameter order before the original final norm. Exact gradients are then concatenated, clipped and updated by packed native fused AdamW. Every copy and dispatch is charged. No cotangent approximation or embedding replacement in the primary arms.",
    timing="Serial synchronized harness from V2/V3, same data, checks and resource guards. Target elapsed includes setup, two model-hash checks, priming, batch transfer, full forward/backward, gradient bucket copies and reduction, concatenation, clipping, fused AdamW and all preceding probes. Every step synchronizes MPS; Python process launch excluded equally. All 35 pair blocks shuffled, AB/BA balanced separately within comparison types; no concurrent task GPU workload during campaign. Complete evaluation and weight-save time separately reported.",
    baseline_selection="Unmodified native BF16 transformer autograd and native fused AdamW, earlier selected as the fastest unmodified eager optimizer in training-only pilots. Architecture, parameter count, model forward, FP32 weights, native derivatives, initialization, batches, clip=1, AdamW betas(.9,.95), decay .1, eps1e-8 and learning-rate schedule are matched. Seven additional exact-incidence derivative-control pairs isolate update equivalence from native atomic embedding variability; they cannot replace the original native primary baseline.",
    deterministic_configuration="Primary control and candidate both retain native algorithms, including atomic embedding accumulation, and request no determinism. Additional equivalence pairs use the same directly verified deterministic embedding cotangents in both arms, solely to test norm/optimizer equivalence over full training. All arms disable redundant uninitialized fills; every operation fully initializes outputs before reading them.",
    heldout_scope="This new method follows failed V3 and source-level norm-kernel diagnosis. Implementation and performance selection use only training bytes and the last 5% diagnostic holdout. WikiText-2 official validation/test have already been exposed and are reused openly for a numerical/quality regression check, not unseen-dataset generalization. No V4 official evaluation precedes this freeze.",
    limitations=["One Apple M5 Pro, PyTorch 2.14.1 MPS and a byte-level transformer/corpus. Compiled, CUDA, distributed and global state-of-the-art comparisons remain untested.",
        "Kernel-specific reduction equivalence is verified for this pinned MPS version and these contiguous FP32 gradient shapes. Different backends, dtypes, versions or strides require independent checks.",
        "This optimizes gradient norms, layout and optimizer dispatch; full transformer derivative products and activation storage remain. Bucket copies and packed-gradient buffers add memory.",
        "Mathematical grouping and parameter packing are established ideas. Contribution is preserving actual reduction arithmetic while batching dispatch, authored code and independently frozen measurements.",
        "Native atomic embedding accumulation remains execution-sensitive in primary arms. Deterministic controls test update equivalence; their paired final hashes cannot certify native cross-execution equality.",
        "The corpus is reused; desktop timing can vary. All primary pairs and adverse measurements are retained, and secondary studies cannot rescue a failed primary gate."],
    publication="Only code, protocols, records, figures and HTML are public; every weight file remains local.")
path.write_text(json.dumps(frozen,indent=2)+"\n")
print(json.dumps(dict(jobs=len(jobs),source_sha256=frozen["source_sha256"],protocol_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
