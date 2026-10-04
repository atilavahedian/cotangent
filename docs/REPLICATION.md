# Cotangent V5: one powered replication of the unchanged V4 method

V4 passed its time and numerical-equivalence gates but failed the combined decision:
its 99% upper quality bound was 0.01160035 BPB against the unchanged 0.01 margin.
It remains a failed combined study, with every one of its 70 runs public.

V5 uses the **same frozen V4 algorithm and original native fused AdamW baseline**.
It changes only the independent seed set, fixed sample size and confidence level.
All 100 pairs (200 runs) are committed before any new official evaluation. There is
no extension of V4 and no stopping when a favorable result appears.

## Sample-size planning

The full 25-pair V4 quality-difference sample standard deviation was
0.020926128365766364 BPB. For planning, use zero mean degradation (supported by
coordinate/update equivalence, but not assumed in the decision), a 0.01 BPB margin,
90% power, and a 99.5% two-sided interval. The normal approximation is

`n = ((z(0.9975) + z(0.90)) * SD / margin)^2 = 73.2022`.

The fixed size is **100 fresh pairs**, allowing extra room for tails and finite-sample
uncertainty. The final interval uses Student t with 99 degrees of freedom and critical
value 2.8713076612147663, independently computed by inversion of its beta-integral
survival probability. Power planning is approximate and does not guarantee success.

## Decision, frozen before outcomes

- All 100 pairs must reach the identical 3.2 BPB validation target.
- Mean paired elapsed-time saving must be at least 10%, with positive 99.5% lower bound.
- The upper 99.5% bound on candidate-minus-native full-test BPB must be at most 0.01.
- The unchanged V4 source, all seven full-training numerical-equivalence pairs,
  completed integrity audit and local checkpoint restoration remain prerequisites.

Exactly 50 pair blocks run native first and 50 candidate first, with fixed shuffled
block order. Architecture, budgets, precision, learning rate, clipping, optimizer
hyperparameters, data schedules and timing harness are otherwise unchanged. Every
planned observation is retained. There is no optional stopping, seed removal,
post-result addition or threshold relaxation. This is the single fixed-size powered
replication planned here; a failure remains a failure.

The stronger intervals do not supply universal family-wise coverage for every
exploratory choice in project development. Prior failed studies are disclosed.
Official corpus splits have been exposed before; this is a numerical/quality
regression comparison, not fresh-dataset generalization. One M5 Pro and eager MPS
cannot establish superiority over compiled CUDA or distributed training.

See [the protocol](../artifacts/v5/frozen-study.json),
[planning values](../artifacts/v5/power-planning.json), and
[unchanged implementation](../research/v4/bucketed.py).
