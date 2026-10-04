"""Add complete V4 evidence to the report, preserving every failed protocol."""
from datetime import datetime, timezone
import json
from pathlib import Path
from analysis.make_report import source

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/"report/src/data.json"
snapshot=json.loads(path.read_text())
r=json.loads((ROOT/"artifacts/v4/analysis/results.json").read_text())
restored=json.loads((ROOT/"artifacts/v4/verification-snapshots.json").read_text())
assert r["runs"]==70 and r["integrity_checks_passed"]
assert restored["status"]=="passed" and restored["snapshots"]==70
snapshot["generatedAt"]=datetime.now(timezone.utc).isoformat()
snapshot["buildStatus"]="complete"
copy=snapshot["report"]["narratives"]

def ci(v,scale=1,digits=2):
    if v is None:return "censored"
    return f"{scale*v['mean']:.{digits}f} (99% CI {scale*v['lower']:.{digits}f} to {scale*v['upper']:.{digits}f})"

p,m=r["small"],r["medium"]
title="Cotangent beat native fused AdamW on this measured setup" if r["primary_success"] else "V4 did not meet its frozen win criteria"
copy["deck"]="A standalone investigation into faster transformer training: preserve native derivatives, batch gradient-norm work, verify numerical updates, and keep the evidence from unsuccessful attempts."
copy["batching-findings"]=f"""## {title}

Across **25 predeclared paired seeds**, the 3.35M model saved **{ci(p['elapsed_reduction_fraction'],100)}%** elapsed time to the same 3.2-BPB validation target. Fixed-budget training time fell **{ci(p['training_reduction_fraction'],100)}%**. Full-test change was **{ci(p['quality_difference_bpb'],digits=4)} BPB**; positive means worse. The frozen requirement was at least 10% mean time saving and an upper quality bound of at most 0.01 BPB.

The time gate **{'passed' if p['time_gate_pass'] else 'failed'}**, the quality gate **{'passed' if p['quality_gate_pass'] else 'failed'}**, and the full-training equivalence gate **{'passed' if r['equivalence_gate_pass'] else 'failed'}**. The combined primary decision **{'passed' if r['primary_success'] else 'failed'}**. Every primary pair is included below, with no discarded seeds or interpolated target times. These are **99% paired intervals**, stricter than the earlier studies' 95% intervals.

All **70 frozen V4 runs completed**, representing **{r['predicted_training_bytes']:,} predicted training bytes**. Five small-model and two larger-model pairs isolate update arithmetic with matched deterministic derivatives; their final parameter hashes and full-test scores were **{'identical in all seven pairs' if r['equivalence_gate_pass'] else 'not identical in every pair'}**. Those controls are additional exactness evidence and cannot replace the native-backprop primary comparison. All 70 local snapshots were restored, their trained parameter hashes matched, and CPU inference was finite. Weights remain local.

The larger 19.28M model is a **three-pair descriptive check**: time saved **{ci(m['elapsed_reduction_fraction'],100)}%**, test difference **{ci(m['quality_difference_bpb'],digits=4)} BPB**. Its sample is too small for a broad scaling claim. The supported result is bounded to this Apple M5 Pro, this pinned PyTorch MPS version, these models and this reused corpus. Compiled, CUDA, distributed and global state-of-the-art comparisons are untested. [Complete V4 evidence](https://github.com/atilavahedian/cotangent/tree/main/artifacts/v4).
"""
copy["batching-method"]="""## Preserve the sum. Batch the dispatch.

Native clipping computes each parameter norm `nₗ = sqrt(Σᵢ gₗᵢ²)`, then the final norm `n = sqrt(Σₗ nₗ²)` and coefficient `c = min(1, 1/(n+10⁻⁶))`. Flattening everything into one reduction is equal in real arithmetic, but changes the floating-point tree. Earlier experiments showed that tiny arithmetic differences can grow into different BF16 training trajectories.

Cotangent groups completed, equal-length gradients into buckets. Each bucket is viewed as `B × (K/2) × 2` and reduced over both trailing axes. This detail matters: the pinned [MPS norm dispatcher](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/ReduceOps.mm) uses a different fast kernel for a single reduced inner axis. Two nontrivial reduced axes retain the generic native reduction, its per-parameter element order and thread count. The scalar norms are restored to native parameter order before the original final norm. Odd or very short lengths use individual native norms.

The complete gradients are then concatenated, clipped with that coefficient and passed to packed **native fused AdamW**. Coordinatewise moment recurrences and decay commute with parameter reindexing when hyperparameters, active coordinates and step counts match. The model retains independent autograd parameter leaves sharing slices of the master vector. Full native forward and backward—including native embeddings—stay unchanged in the primary arms. No cotangents are sampled, projected, skipped or replaced.

The original flat-bucket attempt changed norms and remains in the record. The corrected generic-kernel probe matched every norm bitwise. **32 actual full-model Metal checks** cover two sizes, FP32/BF16 and eight consecutive updates: gradients, clipping norms, every parameter and both AdamW moments matched bitwise. Deterministic-gradient training-only pilots ended with identical whole-model hashes, followed by the seven independent full-length equivalence pairs above. Native atomic embedding variability remains present in the primary arms, so cross-execution native quality is assessed separately.

This is a backend-specific numerical and systems optimization. It saves norm and optimizer dispatch, charges every bucket copy, and adds temporary buffers. It retains full activations and all attention/linear derivative products. Grouping reductions and parameter flattening are established techniques; the contribution is the authored implementation that preserves actual reduction arithmetic and its tested application. [Derivation and code](https://github.com/atilavahedian/cotangent/blob/main/docs/BATCHING.md).
"""
copy["batching-design"]="""## A fresh freeze against the unchanged existing control

V1, V2 and V3 failed their combined gates and remain below. V4 is a new method with new seeds, frozen source hashes and a separately published protocol before any official evaluation. It does not extend a failed sample until a desired result appears. All 35 pair blocks are shuffled and native/candidate order is balanced within the comparison types. There is no optional stopping or removal of adverse measurements.

The primary control is native BF16 transformer autograd with native fused AdamW, selected by training-only optimizer pilots. Forward function, architecture, initialization, schedules, precision, clipping, learning rate, betas, decay, epsilon and step budgets are matched. Every step synchronizes Metal. Target elapsed includes model construction and hashing, priming, data transfer, full forward/backward, bucket copies, reduction, concatenation, clipping, AdamW, finite/resource checks and all preceding scheduled validation probes. Python process launch is excluded equally. Training, complete evaluation and checkpoint saving are separately recorded.

The primary requirements retain the **10% minimum time saving** and **0.01-BPB maximum upper quality difference**, using more conservative **99% two-sided paired Student-t intervals** over 25 pairs. Every pair must cross the same first scheduled 100-step, 65,536-target-byte probe at or below 3.2 BPB. There is no interpolation or imputation. Additionally, every one of the seven deterministic-gradient pairs must end with identical final weights and full-test scores. The three larger native-backprop pairs are descriptive and cannot rescue the primary decision.

WikiText-2 validation/test were already exposed in previous studies. They are openly reused as a quality regression check; implementation and performance selection used training bytes and their separate last-5% diagnostic holdout. New timing seeds are not a fresh dataset. Primary arms retain native algorithms, including atomic embedding accumulation. Equivalence controls use the same directly verified deterministic embedding derivative in both arms solely to isolate update arithmetic. All kernels fully initialize their outputs; redundant uninitialized-memory fills are disabled in every arm. [Immutable V4 protocol](https://github.com/atilavahedian/cotangent/blob/main/artifacts/v4/frozen-study.json).
"""
definitions=[
    ("Time saved","100*(1 - candidate target elapsed/native fused target elapsed), paired by seed. Includes setup and preceding probes. Positive is faster. First scheduled target crossing; no interpolation."),
    ("Full-test difference","Candidate minus native full WikiText-2 test BPB; positive is worse. Reused corpus, 1,292,012 target bytes per run. Final full validation covers 1,148,006 targets."),
    ("Intervals and decision","Two-sided 99% paired Student-t intervals over 25 primary pairs. Mean time saving >=10%, timing lower bound >0, quality upper bound <=0.01 BPB. All seven full-training deterministic-gradient pairs must match final hashes and test scores."),
    ("Equivalence controls","Five small and two larger pairs with the same deterministic derivatives in both arms. They isolate norm/optimizer arithmetic; they cannot replace the unchanged native primary baseline."),
]
tables=(
    ("batching_summary",r["aggregates"],["artifacts/v4/analysis/method-means.csv"],["batching-method-table"]),
    ("batching_pairs",p["pairs"]+m["pairs"],["artifacts/v4/analysis/paired-results.csv"],["batching-findings","batching-paired-table"]),
    ("batching_equivalence",r["equivalence"]["small"]["pairs"]+r["equivalence"]["medium"]["pairs"],["artifacts/v4/analysis/paired-equivalence.csv"],["batching-equivalence-table"]),
    ("batching_curves",r["mean_learning_curves"],["artifacts/v4/final/*/*/curve.jsonl"],["batching-small-curve","batching-medium-curve"]),
    ("batching_integrity",[dict(runs=70,source_sha256=r["source_sha256"],protocol_sha256=r["protocol_sha256"],equivalence_gate_pass=r["equivalence_gate_pass"],restored_snapshots=70)],
     ["artifacts/v4/frozen-study.json","artifacts/v4/verification-mps.json","artifacts/v4/verification-snapshots.json","artifacts/v4/analysis/results.json"],["batching-method","batching-design"]),
)
for key,rows,files,ids in tables:
    provenance=source("V4 native backward and numerically matched norm batching",files,ids,definitions,r["limitations"]+[r["heldout_scope"]])
    provenance["evidenceFlow"][0]["detail"]="Public V4 freeze → all 70 raw runs → analysis/v4/analyze.py → local snapshot restoration → reviewed report. Every seed and earlier failure is preserved."
    snapshot["queries"][key]=dict(rows=rows,source=provenance)
if "For the latest independent protocol," not in copy["reproduce"]:
    copy["reproduce"] += "\n\nFor the latest independent protocol, use `protocol-v4` in a separate checkout and follow [the reproduction guide](https://github.com/atilavahedian/cotangent/blob/main/docs/REPRODUCING.md). Public statistics can be recomputed with `python -m analysis.v4.analyze --records-only`, without weights. That mode explicitly does not claim checkpoint-file verification. Strict local audits verify the actual snapshots."
path.write_text(json.dumps(snapshot,indent=2)+"\n")
print(json.dumps(dict(id=snapshot["id"],buildStatus=snapshot["buildStatus"],primary_success=r["primary_success"])))
