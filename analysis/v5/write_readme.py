"""Generate the final overview from the complete audited replication, never partial runs."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
r=json.loads((ROOT/'artifacts/v5/analysis/results.json').read_text())
restored=json.loads((ROOT/'artifacts/v5/verification-snapshots.json').read_text())
assert r['runs']==200 and r['integrity_checks_passed'] and r['checkpoint_file_hashes_verified']
assert restored['status']=='passed' and restored['snapshots']==200
p=r['small']

def ci(v,scale=1,digits=2):
    if v is None:return 'censored'
    return f"{scale*v['mean']:.{digits}f} (99.5% CI {scale*v['lower']:.{digits}f} to {scale*v['upper']:.{digits}f})"

headline='Beat native fused AdamW on the measured MPS setup' if r['primary_success'] else 'Did not meet the frozen combined win criteria'
text=f"""# Cotangent

**Exact transformer training through numerically matched gradient handling.**

**{headline}.** In the independently frozen V5 replication, **100 fresh paired
seeds** saved **{ci(p['elapsed_reduction_fraction'],100)}%** elapsed time to the
same validation-quality target. Fixed-budget training time fell
**{ci(p['training_reduction_fraction'],100)}%**. Full-test change was
**{ci(p['quality_difference_bpb'],digits=5)} BPB**; positive is worse.

The unchanged gate required 10% mean time saving, a positive timing lower bound,
and an upper quality-degradation bound at most 0.01 BPB. V5 uses **99.5% paired
Student-t intervals**. Time **{'passed' if p['time_gate_pass'] else 'failed'}**,
quality **{'passed' if p['quality_gate_pass'] else 'failed'}**, and the immutable
numerical-equivalence prerequisite **passed**. The combined decision
**{'passed' if r['primary_success'] else 'failed'}**. All **200 planned runs completed**;
no seeds were discarded and no target times interpolated.

[Interactive report](https://atilavahedian.github.io/cotangent/) ·
[Portable HTML release](https://github.com/atilavahedian/cotangent/releases/tag/v0.5.0-research) ·
[Complete results](artifacts/v5/analysis/results.json) ·
[Mathematics](docs/BATCHING.md) · [Preregistered replication](docs/REPLICATION.md)

![Learning curves from all 100 fresh pairs](artifacts/v5/figures/learning-curves.png)

## What changes

The primary model retains **all native forward and backward operations**, including
embeddings, attention and linear derivatives. After backward, Cotangent batches
norm calculations and applies native fused AdamW to a contiguous parameter vector.
No gradient is approximated, sampled, projected, skipped or replaced.

Native clipping computes each parameter norm `n_l = sqrt(sum_i g_l[i]^2)`,
then `n = sqrt(sum_l n_l^2)` and `c = min(1,1/(n+1e-6))`. Flattening everything
into one reduction is equal in real arithmetic but changes floating-point order.
An early flat-norm experiment showed different BF16 training trajectories.

The corrected implementation groups equal-length gradients and views each bucket
as `B × (K/2) × 2`. Reducing both trailing axes avoids MPS's different single-axis
inner fast path, keeping the same generic native reduction, linear element order
and thread count. Norms return to native parameter order before the original final
reduction. Four buckets replace 37 separate parameter-norm reductions in this model.
Every copy is charged to the measured time and memory.

Complete gradients are concatenated and clipped, then native fused AdamW updates
one master vector. AdamW's coordinatewise moment/decay recurrences commute with
parameter reindexing under the same hyperparameters and step counts. Independent
model parameter leaves share slices of that vector; native autograd stays intact.

The unchanged method is [`research/v4/bucketed.py`](research/v4/bucketed.py),
with its full [numerical argument and restrictions](docs/BATCHING.md). It requires
active, dense, homogeneous contiguous parameters and one hyperparameter group.
Construct it before any forward, and do not move or reassign parameters afterward.
Bitwise equivalence is checked for this pinned MPS version and tested contiguous
FP32 gradient shapes. Extra buffers add memory; full activations and derivative
products remain. This is an authored systems/numerical optimization, not a new
chain rule or an invention of parameter flattening.

## Evidence

- **32 actual Metal full-model numerical checks**: two sizes, FP32/BF16, eight
  consecutive updates. Loss, every gradient, clipping norm, parameter and both
  AdamW moments matched native fused AdamW bitwise.
- **Seven full-training equivalence pairs**: same deterministic derivatives in
  both arms to isolate optimizer arithmetic. Final model hashes and full-test
  scores were identical in all seven. They are additional proofs; the primary
  comparison retains the original native backpropagation baseline.
- **100 fresh native-backprop pairs**: independently frozen seeds, matched
  architecture, initialization, data schedules, precision, clipping, optimizer
  hyperparameters and training budgets. Fixed shuffled block order, 50 AB/50 BA.
- **Every one of 200 local snapshots restored**, with trained parameter hashes
  verified and finite CPU inference. Public records contain hashes; all weights
  remain local. Restoration does not re-evaluate the official test data.
- **{r['predicted_training_bytes']:,} predicted training bytes** in the replication.
  The 3,346,944-parameter model trains from scratch for 2,000 steps per run.

The target clock charges setup, hashing, priming, transfer, full forward/backward,
bucket copies, concatenation, clipping, fused AdamW, host/resource checks and all
preceding probes. Every step synchronizes Metal. The first scheduled 100-step,
65,536-target-byte validation probe at or below 3.2 BPB defines the crossing.
Full validation covers 1,148,006 targets and full test covers 1,292,012 targets once
after training. Complete evaluation and checkpoint-save costs are also recorded.

![All paired observations and 99.5% confidence intervals](artifacts/v5/figures/paired-outcomes.png)

## Failed attempts stay public

| Study | Candidate | Planned runs | Combined decision |
| --- | --- | ---: | --- |
| [V1](artifacts/analysis/results.json) | Adaptive sampled weight gradients | 36 | Failed time and quality gates |
| [V2](artifacts/v2/analysis/results.json) | Exact packing, flat norm | 24 | Failed combined gates |
| [V3](artifacts/v3/analysis/results.json) | Deterministic adjoints, flat norm | 41 | Failed combined gates |
| [V4](artifacts/v4/analysis/results.json) | Native backward, matched norm buckets | 70 | Time/equivalence passed; quality gate failed |
| [V5](artifacts/v5/analysis/results.json) | Same V4 method, powered replication | 200 | {'Passed' if r['primary_success'] else 'Failed'} |

V4 saved 17.83% target time, but its upper 99% quality bound was 0.01160035 BPB
against the original 0.01 margin. It remains failed. The new fixed size uses all
V4 quality-difference variance: approximately 73.2 pairs for 90% planning power,
rounded up to 100 before any V5 outcome. Method, baseline and thresholds are
unchanged; intervals become stricter. V5 does not add observations to V4 or erase
it. There is no optional stopping, adverse-seed removal or post-result extension.
The intervals do not claim universal family-wise coverage for every exploratory
choice during development. [Planning and protocol](docs/REPLICATION.md).

## Reproduce

Measured environment: Apple M5 Pro, 24 GiB, macOS 27.0, Python 3.12.14,
PyTorch 2.14.1 MPS. Fresh pinned byte-level WikiText-2; sequence 256, batch eight,
BF16 autocast, FP32 weights, clip=1, AdamW betas(.9,.95), decay .1, eps1e-8.
No code, weights or results were inherited from the author's other projects.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m analysis.v5.analyze --records-only
```

The final command recomputes all statistics from public records without weights
or dataset downloads. It explicitly leaves local checkpoint-file verification
unperformed. For new training, use a separate checkout at **`protocol-v5`**, which
predates every official V5 result. Retrieve pinned data and run the serial
campaign. [Complete instructions](docs/REPRODUCING.md). Do not overwrite published
measurements or edit any frozen training sources. Local snapshots contain model
weights and metadata, not optimizer states for resuming training.

## Scope and attribution

This demonstrates **{'a measured' if r['primary_success'] else 'an investigated'}
eager-MPS efficiency advantage against native fused AdamW**, bounded to the tested
hardware/model/backend. Official corpus splits were exposed before and are openly
reused for quality regression, not unseen-dataset generalization. Compiled, CUDA,
distributed, billion-parameter and global state-of-the-art comparisons remain
untested. Native atomic embedding accumulation is execution-sensitive, so independent
primary trajectories can differ even with equivalent update arithmetic.

Parameter flattening, grouping reductions and fused AdamW are established ideas.
Cotangent contributes the authored numerical/kernel diagnosis, corrected
implementation, proofs and frozen measurements. See the pinned
[MPS norm dispatcher](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/ReduceOps.mm),
[native AdamW documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html),
and [FSDP flat parameters](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_flat_param.py).
Earlier derivations and related work remain in [MATHEMATICS.md](docs/MATHEMATICS.md),
[PACKING.md](docs/PACKING.md) and [DETERMINISM.md](docs/DETERMINISM.md).

The new implementation, analysis and authored explanation are MIT licensed.
Dependencies/report infrastructure retain their terms; dataset content is not
redistributed. No paid/cloud compute was used. **All weight files remain local.**
"""
(ROOT/'README.md').write_text(text)
print(headline)
