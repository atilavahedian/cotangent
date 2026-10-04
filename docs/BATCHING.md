# Batch the dispatch without changing the reduction

V1's approximate cotangents, V2's flat norm and V3's deterministic embedding
candidate all failed their frozen combined gates. V4 tests a different method:
**the full native backward stays unchanged**, while gradient-norm work and
AdamW parameter layout are batched. The native fused optimizer remains the
primary existing baseline. Earlier outcomes are preserved.

## The numerical issue

For completed parameter gradients `g_l`, global clipping computes

```
n_l = sqrt(sum_i g_l[i]^2)
n = sqrt(sum_l n_l^2)
c = min(1, 1 / (n + 1e-6))
```

A flat global norm is equal in real arithmetic, but its floating-point reduction
tree can differ. BF16 transformer training can amplify small differences over
thousands of steps. The new implementation preserves a separate norm for every
parameter and preserves their original order in the final norm.

The pinned [MPS norm dispatcher](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/ReduceOps.mm)
selects a different inner reduction when an output has multiple elements and
only the innermost axis is reduced. Simply stacking flat gradients and reducing
each row therefore changes arithmetic. That first failed numerical probe is in
`artifacts/v4/probe-bucket-norms.json`.

## Preserve the generic kernel

Group equal-length, contiguous gradients into buckets. A bucket of `B` gradients
with `K` elements is stacked and viewed as `B × (K/2) × 2`. Reduce both trailing
axes. For the tested even lengths, both are nontrivial: the inner fast-path
condition is false, so the existing generic norm kernel is used.

Each parameter still has `K` reduced elements in the same linear order and the
same threads per group. Different parameters occupy independent thread groups.
Restore scalar norms to native parameter order before the final reduction.
Odd or very short lengths fall back to individual native norms. The tested
transformers have four length buckets; native clipping otherwise launches 37
small-model or 53 larger-model parameter reductions. All bucket copies are
included in timing and memory measurements.

Concatenate the complete gradients, multiply by the original clipping coefficient
and apply native fused AdamW to the packed master vector. With the same learning
rate, moments, decay, epsilon and step count, each coordinate follows the same
update. Independent model parameter leaves share slices of the master; backward
does not traverse slicing through that master. No derivative is approximated,
skipped or replaced in the primary experiment.

This is a backend-specific implementation argument. It is not a promise of
bitwise equivalence on every device, shape, dtype or future Torch version. The
published dependency is pinned and actual numerical behavior is checked.

## Check numbers before claiming speed

The generic-bucket probe matched every native norm bitwise. The full MPS check
compares FP32 and BF16 models at both sizes for eight consecutive steps: native
gradients, clipping norms, all parameters and both AdamW moments match bitwise in
all 32 checks. The verification uses the same deterministic embedding adjoint in
both arms to isolate update arithmetic from native atomic accumulation.

Training-only pilots retain native embeddings for performance selection. Two
500-step small-model derivative-controlled pairs and one 300-step larger pair
also ended with identical complete trained parameter hashes. They are diagnostic
evidence, not official quality results. The primary frozen campaign keeps the
unmodified native embedding backward in both arms.

## The independent decision

V4 freezes 25 new primary pairs, three larger-model descriptive pairs and seven
full-training deterministic-gradient equivalence pairs: 70 runs. Its win still
requires at least 10% mean elapsed saving to the same target and at most 0.01 BPB
upper quality degradation; it uses **99%** paired intervals, more conservative
than earlier studies. Every equivalence pair must also finish with identical
trained parameter hashes and test scores. All runs and failures remain included.

The reused WikiText-2 corpus is a disclosed quality regression check. This
optimizes clipping and optimizer dispatch, retains full activation storage and
all derivative products, adds temporary buckets and a gradient vector, and does
not establish superiority to untested compiled, CUDA or distributed training.
Grouping reductions and parameter flattening are established techniques; the
contribution is preserving the actual arithmetic while reducing dispatch, plus
authored code and frozen measurements.

## The adjoint and coordinate-invariance argument

For model output `y = f_theta(x)` and loss cotangent `r = dL/dy`, reverse mode
computes `g = (D_theta f_theta(x))^T r`. The primary Cotangent arms evaluate this
same native vector–Jacobian product. They materialize every coordinate of `g`
before doing any batching. The speedup therefore concerns gradient handling and
optimizer application, rather than changing the chain rule or estimating the
adjoint.

Let `P` denote concatenation/reindexing of all active parameter coordinates into
the master vector. As a coordinate permutation, `P` is an isometry:
`||P g||_2 = ||g||_2`. The global clipping coefficient is consequently invariant
in real arithmetic. For the clipped gradient `h = c g`, packed moment recurrences
satisfy

```
P m_t = beta1 * P m_(t-1) + (1-beta1) * P h_t
P v_t = beta2 * P v_(t-1) + (1-beta2) * (P h_t)^2
```

because elementwise square, square root, division, addition and scalar
multiplication commute with a coordinate permutation. Bias corrections use the
same scalar step count. The parameter update therefore also commutes with `P`.
Induction from the same initial parameters and zero moments establishes
coordinate-equivalent AdamW iterates in real arithmetic, under the implementation's
single-group, all-active-coordinate assumptions.

The finite-precision issue is separate: floating-point addition is not associative,
so an isometry proof cannot establish bitwise equality of parallel sums. V4
preserves the native two-level norm structure and chooses its same actual generic
reduction kernel. Numerical checks on the pinned backend then test the part that
real-arithmetic algebra alone cannot establish. Native embedding atomics still
make independently executed primary derivatives variable; the controlled
adjoint checks isolate the optimizer, and the independent quality interval
assesses the complete native training process.

## V5: unchanged method, independently powered replication

The complete V4 result passed time and all seven equivalence pairs, but failed
quality uncertainty: upper 99% bound 0.01160035 BPB exceeds the original 0.01.
V5 preserves this implementation byte for byte. Its single fixed-size replication
uses 100 fresh pairs and stricter 99.5% intervals, with the same original native
baseline and thresholds. The complete V4 variance determines sample-size planning;
no observations are added after outcomes. See [REPLICATION.md](REPLICATION.md).
