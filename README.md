# Cotangent

**Efficient exact transformer training through numerically matched gradient-norm batching.**

Cotangent is a standalone ML research and engineering project: derive the method,
inspect backend arithmetic, verify gradients and optimizer moments, freeze the
experiment, actually train models, and preserve every result. No implementation,
weights, preprocessing or results were inherited from the author's other projects.

**V4 result: did not meet the frozen combined win criteria.** Across **25 predeclared paired seeds**, the
3.35M transformer saved **17.83 (99% CI 16.07 to 19.59)%** elapsed time
to the same validation-quality target. Fixed-budget training time fell
**18.17 (99% CI 17.31 to 19.02)%**. Full-test change was
**-0.0001 (99% CI -0.0118 to 0.0116) BPB**; positive is worse.

The frozen minimum was 10% mean time saving with a positive interval lower bound,
and at most 0.01 BPB upper quality degradation. V4 uses **99% paired intervals**.
Its statistical gate **failed** and its additional
full-training update-equivalence gate **passed**.
All **70 V4 runs completed**; all planned seeds are retained.

[Interactive research report](https://atilavahedian.github.io/cotangent/) ·
[Standalone HTML release](https://github.com/atilavahedian/cotangent/releases/tag/v0.4.0-research) ·
[Complete results](artifacts/v4/analysis/results.json) ·
[Mathematical and numerical argument](docs/BATCHING.md)

## What changes

The primary model uses the **same native forward and backward**, including
embeddings, attention and every linear derivative, as native fused AdamW.
After backward, Cotangent groups complete equal-length gradients into buckets,
preserves the native per-parameter reduction arithmetic, restores norm order,
then clips and applies native fused AdamW on a contiguous parameter/gradient vector.

An ordinary flat norm or batched single-axis norm changes floating-point trees.
The new implementation views each tested bucket as `B × (K/2) × 2` and reduces
both trailing axes, preserving MPS's generic native reduction rather than its
different inner-axis fast path. The tested models use four length buckets instead
of 37 or 53 separately dispatched parameter norms. Every bucket copy is charged.

For gradients `g_l`, native clipping uses
`n_l = sqrt(sum_i g_l[i]^2)`, `n = sqrt(sum_l n_l^2)`,
`c = min(1,1/(n+1e-6))`. AdamW is coordinatewise, so parameter reindexing preserves
its recurrence under the same hyperparameters and step counts. Numerical
equivalence is checked on the actual hardware, rather than inferred from real
arithmetic alone. No gradient is sampled, skipped, projected or replaced.

The implementation is [`research/v4/bucketed.py`](research/v4/bucketed.py).
It requires dense, active, homogeneous contiguous parameters and one set of
optimizer hyperparameters. Construct it before any forward; do not move or
reassign parameters afterward. The bitwise claim is specific to the pinned MPS
version and tested contiguous FP32 gradient shapes, not every backend or dtype.
Full activations and derivative matrix products remain; temporary buffers add memory.

## Evidence

- **32 actual full-model MPS checks**: two sizes, FP32/BF16, eight consecutive
  updates each. Gradients, clipping norms, all parameters and both AdamW moments
  matched native fused AdamW bitwise.
- **Seven full-training equivalence pairs**: five small and two larger pairs
  use the same deterministic derivatives in both arms to isolate optimizer
  arithmetic. Final weights and test scores **matched in all seven pairs**.
  They cannot replace the native-backprop primary comparison.
- **25 independent native-backprop primary pairs**: identical architecture,
  initialization, data schedules, precision and hyperparameters; shuffled pair
  blocks and balanced execution order. Every step synchronizes Metal.
- **Three larger native-backprop pairs**, descriptive: time saved
  **26.70 (99% CI 21.30 to 32.10)%**, test change
  **-0.0269 (99% CI -1.1763 to 1.1225) BPB**. This small sample cannot
  establish broad scaling.
- All 70 local V4 snapshots restored with matching trained parameter hashes
  and finite CPU inference. Public records include their hashes; weights stay local.

The target clock includes setup, model hashing, priming, batch transfer, full
forward/backward, norm buckets, packing, clipping, fused AdamW, resource/finite
checks and preceding validation probes. First scheduled 100-step probe at or
below 3.2 BPB defines the crossing, without interpolation. Each probe contains
65,536 targets; final validation and test cover their complete splits.

## Preserve the failed experiments

V1, V2 and V3 failed their frozen combined criteria. They remain public:

| Study | Candidate | Frozen runs | Primary result |
| --- | --- | ---: | --- |
| [V1](artifacts/analysis/results.json) | Adaptive sampled weight gradients | 36 | Slower; time and quality gates failed |
| [V2](artifacts/v2/analysis/results.json) | Exact packing with a flat norm | 24 | Training faster; combined gates failed |
| [V3](artifacts/v3/analysis/results.json) | Deterministic adjoints + flat norm | 41 | Training faster; combined gates failed |
| [V4](artifacts/v4/analysis/results.json) | Native backward + matched norm buckets | 70 | Did not meet the frozen combined win criteria |

V4 is a new method and independent freeze, not additional seeds appended to a
failed study. Its time and quality thresholds were retained and its intervals
made stricter. Initial numerical failures, slower prototypes and adverse runs
remain inspectable. Native MPS atomic embedding accumulation was execution-sensitive
in a repeated larger-model diagnostic; deterministic full-training repeats and
controlled equivalence checks are reported separately.

## Reproduce

Measured environment: Apple M5 Pro, 24 GiB, macOS 27.0, Python 3.12.14,
PyTorch 2.14.1, Metal. Models train from scratch on pinned byte-level WikiText-2.
The main model has 3,346,944 parameters and trains 2,000 steps; the larger model
has 19,280,896 parameters and trains 1,500 steps. Sequence length 256, batch eight,
BF16 autocast, FP32 parameters, clip=1, AdamW betas(.9,.95), decay .1, eps1e-8.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m analysis.v4.analyze --records-only
```

That last command recomputes all statistics from public records, without weights
or data downloads, and explicitly marks weight-file verification unperformed.
To retrain, use a separate checkout at **`protocol-v4`**, which predates official
evaluation, retrieve the pinned data, run verification and the serial campaign.
Follow [the complete reproduction guide](docs/REPRODUCING.md). Do not overwrite
published measurements or edit any tagged training sources. Snapshots contain
model weights and metadata, not optimizer states for resuming training.

## Scope and prior art

This is a **scoped eager-MPS training-efficiency result against native fused AdamW**.
The official corpus was already exposed in earlier studies and is openly reused
as a quality regression check. It is not unseen-dataset generalization. Compiled,
CUDA, distributed, billion-parameter and global state-of-the-art comparisons are
untested. Performance and reduction behavior require new verification on another
hardware/backend/version.

Grouping reductions, flat parameters and fused AdamW are established techniques.
Cotangent contributes authored numerical/kernel diagnosis, implementation and
measured evidence, not their invention or a new chain rule. See the pinned
[MPS norm dispatcher](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/ReduceOps.mm),
[native AdamW documentation](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html),
and [FSDP flat parameters](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_flat_param.py).
Earlier mathematical investigations and related-work attribution remain in
[`MATHEMATICS.md`](docs/MATHEMATICS.md), [`PACKING.md`](docs/PACKING.md) and
[`DETERMINISM.md`](docs/DETERMINISM.md).

The new research implementation, analysis and authored explanation are MIT
licensed. Dependencies and the general-purpose report runtime retain their own
terms. WikiText retains its source license; dataset content is not redistributed.
No paid or cloud compute was used. All weight files remain local.
