"""Bind V3's complete evidence to the existing report; preserve V1 and V2."""
from datetime import datetime, timezone
import json
from pathlib import Path

from analysis.make_report import source

ROOT = Path(__file__).resolve().parents[2]
path = ROOT / "report/src/data.json"
snapshot = json.loads(path.read_text())
result = json.loads((ROOT / "artifacts/v3/analysis/results.json").read_text())
assert result["runs"] == 41 and result["integrity_checks_passed"]
verification = json.loads((ROOT / "artifacts/v3/verification-snapshots.json").read_text())
assert verification["status"] == "passed" and verification["snapshots"] == 41
snapshot["generatedAt"] = datetime.now(timezone.utc).isoformat()
snapshot["buildStatus"] = "complete"
copy = snapshot["report"]["narratives"]

def ci(value, scale=1, digits=2):
    if value is None:
        return "censored"
    return f"{value['mean']*scale:.{digits}f} (95% CI {value['lower']*scale:.{digits}f} to {value['upper']*scale:.{digits}f})"

small, medium, ablation = result["small"], result["medium"], result["ablation"]
title = "V3 beat native fused AdamW on this measured setup" if result["primary_success"] else "V3 did not meet its frozen win criteria"
copy["deck"] = "A measured investigation into efficient transformer training: approximate cotangents, exact packing, deterministic embedding adjoints, and preserved evidence from every attempt."
copy["deterministic-findings"] = f"""## {title}

Across **15 predeclared paired seeds**, the 3.35M model saved **{ci(small['elapsed_reduction_fraction'],100)}%** elapsed time to the same 3.2-BPB validation target. Fixed-budget training time fell **{ci(small['training_reduction_fraction'],100)}%**. Final full-test difference was **{ci(small['quality_difference_bpb'],digits=4)} BPB**; positive means worse. The time gate **{'passed' if small['time_gate_pass'] else 'failed'}**, the 0.01-BPB quality gate **{'passed' if small['quality_gate_pass'] else 'failed'}**, and the combined primary gate **{'passed' if small['success'] else 'failed'}**.

Every primary seed is shown below, including the slow candidate run at seed 389. All **41 frozen runs completed**, consuming **{result['predicted_training_bytes']:,} predicted training bytes**. Each final validation/test evaluation covers its complete split. All 41 local snapshots were restored, their parameter hashes matched, and CPU inference was finite. Weights remain local.

The larger 19.28M model is a **three-pair descriptive check**: elapsed saving **{ci(medium['elapsed_reduction_fraction'],100)}%**, test difference **{ci(medium['quality_difference_bpb'],digits=4)} BPB**. Its combined criterion **{'passed' if medium['success'] else 'failed'}**. It cannot substitute for the primary outcome. The **five-seed deterministic-embedding control** isolates packing: elapsed saving **{ci(ablation['elapsed_reduction_fraction'],100)}%**, test difference **{ci(ablation['quality_difference_bpb'],digits=4)} BPB**. Its sample and control differ from the primary comparison.

This is a backend-specific training-efficiency study against an existing optimized baseline. It does not establish universally faster backpropagation, large-vocabulary scaling, or superiority to untested CUDA, compiled or distributed training. V1 and V2 remain inspectable negative results below. [Complete V3 records and audit](https://github.com/atilavahedian/cotangent/tree/main/artifacts/v3).
"""
copy["deterministic-method"] = """## Exact cotangents, deterministic accumulation

V2's unchanged native larger-model repeat moved from 2.692846 to 2.828499 test BPB with the same seed, source, initial parameters and batch schedule. This single diagnostic demonstrates execution sensitivity; it does not identify every cause of every quality gap. The installed PyTorch revision's [MPS embedding shader](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels/Embedding.metal) uses floating-point atomic addition. Its installed binary accepts our small strict-determinism probe, so a flag alone is insufficient evidence of reproducibility.

For byte indices `zᵢ`, define the incidence matrix `Hᵢᵥ = 1[zᵢ = v]`. An embedding is `E = HW`. Given incoming cotangents `D`, the exact adjoint is **`dL/dW = HᵀD`**. Cotangent constructs this matrix with integer comparisons and evaluates the full product in FP32, including all repeated-index contributions and unused rows. Native forward lookup stays unchanged. The position table uses the exact unique-prefix adjoint; its broadcast backward has already summed the batch dimension.

The complete gradients are then concatenated, globally clipped, and passed to fused AdamW on a contiguous master parameter vector. With homogeneous hyperparameters, `‖concat(g₁,…,gL)‖² = Σₗ‖gₗ‖²`; coordinatewise AdamW commutes with this reindexing in real arithmetic. Native atomic reductions and norm trees can differ in floating point, so the independent trained-quality gate remains essential. All attention and linear derivatives remain native and exact; no gradient rows are sampled or skipped.

CPU double tests compare exact derivatives and clipped/unclipped optimizer moments. Actual Metal checks compare complete-model FP32/BF16 losses and gradients; repeated candidate gradients are bitwise identical. Initial failed assumptions, redundant-initialization pilots, and GPU/CPU-sorted Metal reduction prototypes remain in the record. The Metal prototypes passed correctness checks but showed no convincing speed advantage over incidence GEMM, so they were excluded from the final candidate. [Derivation, assumptions and preserved prototypes](https://github.com/atilavahedian/cotangent/blob/main/docs/DETERMINISM.md).

This combines established incidence-matrix differentiation, GEMM, parameter packing and fused AdamW. The contribution is the authored implementation, numerical diagnosis and measured application; these ideas are credited as prior art.
"""
repeat_path = ROOT / "artifacts/v3/diagnostics/candidate-repeat-comparison.json"
if repeat_path.exists():
    repeat = json.loads(repeat_path.read_text())
    copy["deterministic-method"] += f"\n\nA post-study 19.28M candidate repeat matched source, initialization, schedule and configuration. Full-test difference was **{repeat['repeat_minus_original_bpb']:.8f} BPB** and the final parameter hashes were **{'identical' if repeat['identical_final_model_hash'] else 'different'}**. This is one repeat on this machine, excluded from every frozen gate; it does not prove cross-platform determinism."
copy["deterministic-design"] = """## A new freeze with the same demanding gates

The third protocol froze source hashes, 15 new small-model pairs, five deterministic-control ablations, three larger-model pairs and a balanced, shuffled run order before any V3 official evaluation. The primary win requires every pair to reach the 3.2-BPB target, at least **10% mean paired elapsed reduction** with its two-sided 95% Student-t interval above zero, and the upper paired full-test quality difference at most **0.01 BPB**. These thresholds were retained from failed V2. There is no optional stopping, seed removal or reinterpretation of a descriptive result as primary success.

Native fused AdamW is the existing control, selected by training-only optimizer pilots. Forward function, architecture, initialization, batch schedules, precision, clipping, hyperparameters, step budgets and evaluation are matched. Every step synchronizes Metal. Target elapsed includes setup, model-hash checks, priming, batching, full forward/backward, incidence work, packing, clipping, optimization, finite/resource checks and all preceding scheduled probes. First 100-step probe crossing is used without interpolation; training and complete-evaluation durations are separately reported.

Candidates request deterministic algorithms; all arms disable redundant uninitialized-memory fills because every operation initializes its complete outputs before reading them, as permitted by [PyTorch's guidance](https://docs.pytorch.org/docs/main/notes/randomness.html#filling-uninitialized-memory). Determinism is directly tested, rather than inferred from this configuration alone.

WikiText-2 validation/test were already evaluated in V1 and V2 and are openly reused for a quality regression check. Method tuning used training bytes and their separate diagnostic holdout. New seeds replicate timing and quality on this corpus; they do not create an unseen dataset. Dense incidence arithmetic costs `O(NVd)` and additional buffers, and is tested only with **V=256 bytes**. No activation-memory reduction or 50k-token-vocabulary gain is established. The measured scope is one Apple M5 Pro, PyTorch 2.14.1 MPS, and these models. [Immutable V3 protocol](https://github.com/atilavahedian/cotangent/blob/main/artifacts/v3/frozen-study.json).
"""

definitions = [
    ("Elapsed reduction", "100*(1 - candidate elapsed / matched control elapsed) at first 65,536-target-byte probe <=3.2 BPB. Includes setup and probes; positive is faster."),
    ("Quality difference", "Candidate full-test BPB minus matched control BPB. Positive is worse. Entire reused WikiText-2 test split; 1,292,012 target bytes."),
    ("Inference", "Two-sided 95% paired Student-t intervals: 15 primary, three larger-model descriptive, five deterministic-control ablation pairs. Primary mean saving >=10%, timing lower bound >0, quality upper bound <=0.01 BPB."),
]
tables = (
    ("deterministic_summary", result["aggregates"], ["artifacts/v3/analysis/method-means.csv"], ["deterministic-method-table"]),
    ("deterministic_pairs", small["pairs"]+medium["pairs"], ["artifacts/v3/analysis/paired-results.csv"], ["deterministic-findings","deterministic-paired-table"]),
    ("deterministic_ablation", ablation["pairs"], ["artifacts/v3/analysis/paired-ablation.csv"], ["deterministic-ablation-table"]),
    ("deterministic_curves", result["mean_learning_curves"], ["artifacts/v3/final/*/*/curve.jsonl"], ["deterministic-small-curve","deterministic-medium-curve"]),
    ("deterministic_integrity", [dict(runs=41, source_sha256=result["source_sha256"], protocol_sha256=result["protocol_sha256"], integrity_checks_passed=True, restored_snapshots=41)],
     ["artifacts/v3/frozen-study.json","artifacts/v3/verification-incidence.json","artifacts/v3/verification-segmented.json","artifacts/v3/verification-snapshots.json","artifacts/v3/analysis/results.json"], ["deterministic-method","deterministic-design"]),
)
for key, rows, files, ids in tables:
    provenance = source("V3 deterministic exact-gradient study", files, ids, definitions,
                        result["limitations"]+[result["heldout_scope"]])
    provenance["evidenceFlow"][0]["detail"] = "Frozen V3 protocol → all 41 raw runs → analysis/v3/analyze.py → source-bound report. No seed discarded; secondary comparisons remain descriptive."
    snapshot["queries"][key] = dict(rows=rows, source=provenance)
path.write_text(json.dumps(snapshot, indent=2)+"\n")
print(json.dumps(dict(id=snapshot["id"],buildStatus=snapshot["buildStatus"],primary_success=result["primary_success"])))
