"""Build the complete replication report; retain every earlier failed study."""
from datetime import datetime, timezone
import json
from pathlib import Path
from analysis.make_report import source

ROOT=Path(__file__).resolve().parents[2]
path=ROOT/'report/src/data.json'
snapshot=json.loads(path.read_text())
r=json.loads((ROOT/'artifacts/v5/analysis/results.json').read_text())
restored=json.loads((ROOT/'artifacts/v5/verification-snapshots.json').read_text())
assert r['runs']==200 and r['integrity_checks_passed'] and r['checkpoint_file_hashes_verified']
assert restored['status']=='passed' and restored['snapshots']==200
assert r['equivalence_prerequisite_pass']
p=r['small']
copy=snapshot['report']['narratives']
snapshot['generatedAt']=datetime.now(timezone.utc).isoformat()
snapshot['buildStatus']='complete'

def ci(v,scale=1,digits=2):
    if v is None:return 'censored'
    return f"{scale*v['mean']:.{digits}f} (99.5% CI {scale*v['lower']:.{digits}f} to {scale*v['upper']:.{digits}f})"

title='Cotangent beat native fused AdamW on the measured MPS setup' if r['primary_success'] else 'V5 did not meet its frozen combined win criteria'
copy['deck']='Exact transformer training, cheaper gradient handling. An authored numerical optimization, 100 fresh paired comparisons, and an inspectable record of what worked and what failed.'
copy['replication-findings']=f"""## {title}

Across **100 fresh, predeclared paired seeds**, Cotangent saved **{ci(p['elapsed_reduction_fraction'],100)}%** elapsed time to the same 3.2-BPB validation target. Full-test change was **{ci(p['quality_difference_bpb'],digits=5)} BPB**; positive is worse. Fixed-budget training time fell **{ci(p['training_reduction_fraction'],100)}%**.

The unchanged gate required **at least 10% mean time saving**, a positive time lower bound, and **at most 0.01 BPB upper quality degradation**. The time gate **{'passed' if p['time_gate_pass'] else 'failed'}**, the quality gate **{'passed' if p['quality_gate_pass'] else 'failed'}**, and the numerical-equivalence prerequisite **passed**. The combined decision **{'passed' if r['primary_success'] else 'failed'}**. These are **99.5% two-sided paired Student-t intervals**, computed only after all planned observations completed.

All **200 runs completed**, totaling **{r['predicted_training_bytes']:,} predicted training bytes**. Every snapshot was restored locally with its trained parameter hash verified and finite CPU inference. Every one of the 100 pairs remains in the searchable table below. No seeds were dropped and no target times interpolated.

The supported result is a training-efficiency comparison against **unmodified native BF16 backpropagation and native fused AdamW** on one Apple M5 Pro, a 3.35M byte transformer and a disclosed reused corpus. This does not establish a general breakthrough in backpropagation or superiority to untested compiled, CUDA or distributed systems. All weights remain local. [Complete evidence](https://github.com/atilavahedian/cotangent/tree/main/artifacts/v5).
"""
copy['replication-method']="""## Why the method can save time without approximating gradients

Backpropagation still computes every native derivative, including embeddings, attention and linear layers. Cotangent changes what happens **after** gradients are complete: gradient-norm reductions, memory layout and optimizer dispatch.

Native clipping computes one norm for each parameter, then their joint norm. A single flat norm is mathematically equivalent but changes floating-point reduction order. Cotangent instead groups equal-length gradients and views each bucket as `B × (K/2) × 2`. Reducing both trailing axes keeps the pinned MPS generic norm kernel, its linear element order and thread count. Scalar norms return to native parameter order before the original final reduction. Four buckets replace 37 separately dispatched parameter norms in this model; all copies count in measured time.

The exact gradients are concatenated, multiplied by the same clipping coefficient and passed to native fused AdamW on one contiguous master vector. AdamW's coordinatewise moment and decay recurrences commute with this reindexing under identical hyperparameters, active coordinates and step counts. Independent model parameter leaves share slices of that vector; native autograd remains intact.

**32 actual Metal numerical checks** covered two sizes, FP32/BF16 and eight consecutive updates: loss, every gradient, clipping norm, parameter and both AdamW moments matched bitwise. **Seven complete-training equivalence pairs** also finished with identical whole-model hashes and validation/test scores, using the same deterministic embedding derivative in both arms to isolate optimizer arithmetic. V5 uses the unchanged V4 source and retains those proofs as frozen prerequisites.

Primary comparisons keep native atomic embedding accumulation in both arms. Cross-execution trajectories can therefore differ even with equivalent optimizer arithmetic; the independent quality interval measures that variability. The numerical equivalence is specific to the pinned Torch MPS version and tested dense contiguous FP32 gradient shapes. Temporary buckets add memory; full activations and derivative products remain. [Mathematical argument, restrictions and implementation](https://github.com/atilavahedian/cotangent/blob/main/docs/BATCHING.md).
"""
copy['replication-design']="""## One fixed-size replication, frozen before outcomes

V1–V4 failed their combined gates and remain below. V4 had a real time advantage but its upper 99% quality bound, 0.01160035 BPB, missed the original 0.01 margin. It remains failed. V5 keeps the **same V4 algorithm, original native baseline, architecture, data, precision, budgets, hyperparameters and thresholds**. Only fresh seeds, fixed sample size and the stricter interval level change.

All 25 V4 quality differences supplied the planning SD, 0.0209261 BPB. At zero planned mean degradation, 90% power and a 99.5% two-sided interval, the normal approximation calls for 73.2 pairs. **100 pairs were publicly preregistered before any V5 evaluation** to allow extra room for tails. This is one independent powered replication, rather than adding observations to V4. Power is approximate, not a guarantee. The final analysis uses Student t, df99, critical 2.8713076612147663.

The 100 pair blocks have a fixed shuffled order: exactly 50 run native first and 50 candidate first. Every pair matches initialization and batch schedule. Every step synchronizes Metal. Target elapsed charges setup, hashing, priming, data transfer, full forward/backward, all bucketing/concatenation/clipping/AdamW work, host/resource checks and preceding validation probes. Python launch is excluded equally. No concurrent task GPU work ran during the campaign. The first scheduled 100-step, 65,536-target-byte probe at or below 3.2 BPB defines target time.

There is no optional stopping, seed removal, post-result sample extension or threshold relaxation. Prior failures are not overwritten. Stronger intervals do not provide universal family-wise coverage for every exploratory choice during development. Official WikiText-2 splits have been exposed previously and are openly reused for a numerical/quality regression check; fresh timing seeds are not an unseen dataset. Full validation covers 1,148,006 targets and full test covers 1,292,012 targets once after each training run. [Protocol](https://github.com/atilavahedian/cotangent/blob/main/artifacts/v5/frozen-study.json) · [Power planning](https://github.com/atilavahedian/cotangent/blob/main/docs/REPLICATION.md).
"""
copy['replication-pairs']='## Every fresh pair\n\nAll 100 frozen pairs are retained, including adverse quality differences and slower outcomes. Search by seed or sort a column to inspect the measurements. The decision uses the complete sample; this table is not a seed-selection tool.'
copy['reproduce'] += '\n\nV5 is the final fixed-size replication of unchanged V4. Recompute its public statistics with `python -m analysis.v5.analyze --records-only`; that mode explicitly leaves local checkpoint-file verification unperformed. Fresh training uses a separate checkout at `protocol-v5` and `python -m research.v5.campaign`. [Exact reproduction instructions](https://github.com/atilavahedian/cotangent/blob/main/docs/REPRODUCING.md).'
definitions=[
    ('Time saved','100*(1 - candidate target elapsed/native fused target elapsed), paired by seed. Positive is faster. Includes setup and preceding probes. First scheduled crossing; no interpolation.'),
    ('Full-test difference','Candidate minus native full-test BPB; positive is worse. Reused corpus, 1,292,012 targets per run. Quality noninferiority does not mean a superior model.'),
    ('Intervals and gate','Two-sided 99.5% paired Student-t over exactly 100 fresh pairs, df99 critical 2.8713076612147663. Mean time saving >=10%, lower time bound >0, upper quality bound <=0.01 BPB. Immutable V4 equivalence proof required.'),
    ('Replication','Same frozen V4 method and original native baseline. 100 independent fresh pairs planned from complete V4 variability before any V5 outcomes; no optional stopping or extensions.'),
]
metrics=[]
for label,key,scale,unit in [('Elapsed time saved','elapsed_reduction_fraction',100,'%'),('Training time saved','training_reduction_fraction',100,'%'),('Test quality change','quality_difference_bpb',1,'BPB')]:
    value=p[key]
    metrics.append(dict(metric=label,mean=scale*value['mean'] if value else None,lower=scale*value['lower'] if value else None,upper=scale*value['upper'] if value else None,unit=unit))
tables=[
    ('replication_summary',r['aggregates'],['artifacts/v5/analysis/method-means.csv'],['replication-method-table']),
    ('replication_metrics',metrics,['artifacts/v5/analysis/results.json'],['replication-findings','replication-metrics-table']),
    ('replication_pairs',p['pairs'],['artifacts/v5/analysis/paired-results.csv'],['replication-pairs','replication-paired-table']),
    ('replication_curves',[dict(point,target_bpb=3.2) for point in r['mean_learning_curves']],['artifacts/v5/analysis/learning-curves.csv'],['replication-elapsed-curve','replication-step-curve']),
    ('replication_integrity',[dict(runs=200,source_sha256=r['source_sha256'],method_source_sha256=r['v4_source_sha256'],protocol_sha256=r['protocol_sha256'],restored_snapshots=200,equivalence_prerequisite_pass=True)],
     ['artifacts/v5/frozen-study.json','artifacts/v5/power-planning.json','artifacts/v5/verification-snapshots.json','artifacts/v5/analysis/results.json','artifacts/v4/verification-mps.json','artifacts/v4/analysis/paired-equivalence.csv'],['replication-method','replication-design']),
]
for key,rows,files,ids in tables:
    provenance=source('V5 independent powered replication of unchanged V4',files,ids,definitions,r['limitations']+[r['heldout_scope']])
    provenance['evidenceFlow'][0]['detail']='Public protocol-v5 freeze → all 200 raw runs → analysis/v5/analyze.py → 200 local restorations → final report. All four prior failed studies remain disclosed.'
    snapshot['queries'][key]=dict(rows=rows,source=provenance)
path.write_text(json.dumps(snapshot,indent=2)+'\n')
print(json.dumps(dict(id=snapshot['id'],buildStatus='complete',primary_success=r['primary_success'])))
