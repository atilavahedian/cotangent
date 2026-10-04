# Experimental record

| Study | Question | Planned runs | Outcome |
| --- | --- | ---: | --- |
| V1 | Sampled weight gradients | 36 | Failed time and quality gates |
| V2 | Packing with a flat clipping norm | 24 | Failed combined gate |
| V3 | Deterministic adjoints and flat norm | 41 | Failed combined gate |
| V4 | Numerically matched norm buckets | 70 | Time/equivalence passed; quality uncertainty failed |
| V5 | Independent powered replication of V4 | 200 | Passed the original time and quality thresholds |
| V6 | Factorial ablation and compiled architecture comparisons | 128 | Descriptive extension; complete per-configuration intervals |

V4 remains failed: its upper 99% quality bound was 0.01160035 BPB against the
original 0.01 margin. The independent V5 sample size was fixed before its
outcomes, using the complete V4 variance. V5 changed neither the implementation
nor the original thresholds. All 100 pairs completed and remain in the result.

V6 screened executable baselines using two training-only diagnostic seeds.
Compilation was fastest in all four configurations. Each candidate uses the
same compiled model as its baseline and changes only gradient handling.
Eight additional four-treatment blocks isolate norm batching and parameter packing.
All 128 official extension runs are retained. They are descriptive comparisons,
not a replacement for the powered V5 decision.

Sources under cotangent/, scripts/, tests/, configs/, and research/v2/ through
research/v6/ retain their original frozen fingerprints. Tags protocol-v5 and
protocol-v6 precede their official outcomes. Reproduction must use a separate
checkout and must never overwrite published observations.

Numerical controls, training-only pilots, official quality checks and snapshot
restorations are separately labelled in their records. Official corpus splits
have been exposed before and are reused for quality regression.

[Original replication report](archive/v5.html) ·
[V5 results](../artifacts/v5/analysis/results.json) ·
[V6 results](../artifacts/v6/analysis/results.json)
