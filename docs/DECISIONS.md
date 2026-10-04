# Decision log

## Before training

- The project began in a clean directory. No previous author project was read or
  copied. New model and algorithm implementations were written here.
- The machine was measured live: Apple M5 Pro, 24 GiB unified RAM, PyTorch 2.14.1,
  Metal/MPS available, CUDA unavailable. Only about 8.3 GiB RAM was available during
  initial inspection; the resource protocol is therefore conservative.
- The public WikiText-2 raw corpus was downloaded afresh and pinned to a repository
  revision and split hashes. Original datasets are not committed.
- A Dataset Viewer filename does not equal the raw repository filename. The first
  download failed with HTTP 404; the downloader was repaired to resolve the pinned
  repository tree. No experiment results existed at that point.
- Existing low-rank training methods already project both activations and gradients.
  The primary candidate instead importance-samples the weight-gradient product,
  preserving exact activation-gradient propagation. This sacrifices activation
  memory savings to isolate the variance/compute tradeoff.
- A separate WHT-style projection control retains the more aggressive mechanism.
  Official INSTANT source was read for its equations; no official performance
  reproduction or superiority claim is implied.
- Seven CPU mathematical and derivative tests passed before the first MPS training
  pilot. These tests establish implementation properties, not a training advantage.

## Diagnostic refinement, before official held-out evaluation

- The first MPS run failed because explicitly forced foreach gradient clipping is
  unsupported on this backend. The failed run is retained. Standard clipping and
  AdamW now use PyTorch's automatic backend selection, equally for every arm.
- At 120 steps, the training-only pilot's native BF16 run used about 2.15 seconds
  of measured training versus 2.78 seconds for native FP32, with essentially equal
  diagnostic BPB. BF16 is selected for all final arms.
- The first adaptive BF16 controller used 5.48 seconds. It synchronized once per
  layer to read variance statistics. This unfavorable result is preserved.
- The revised controller selects layer budgets only at periodic audits and batches
  their host reads. Dense fallbacks use native autograd directly. Its new small
  pilot used 2.55 seconds versus 2.22 seconds for native BF16; the larger pilot used
  7.85 seconds versus 7.11 seconds. These short runs still do not show a speed win.
- Both FP32 and BF16 exact-custom transformer gradients matched native gradients
  on MPS in the diagnostic check. Eight CPU tests passed after the refinement.
- The final study is deliberately retained despite unfavorable short pilots, to
  test whether longer learning dynamics change the result. It uses five paired
  seeds for the small-model controls and three paired seeds for a larger-model
  robustness comparison. The method is no longer tuned after the study freeze.

## Subsequent frozen studies and the numerical diagnosis

- V1's approximate weight-gradient candidate failed time and quality gates.
  All 36 runs, unfavorable pilots and local snapshots were retained.
- V2 selected the fastest unmodified eager native optimizer in training-only
  pilots, then tested exact parameter/gradient packing against native fused
  AdamW. Flat clipping changes the floating-point reduction tree. Fixed-budget
  training was faster, but the frozen 24-run combined decision failed.
- A post-study native MPS repeat exposed execution-sensitive embedding
  accumulation. Native source uses floating-point atomic additions; a requested
  deterministic flag alone did not prove reproducibility on the pinned build.
- V3 authored exact deterministic embedding adjoints and verified actual Metal
  repeatability. Its new 41-run primary study still failed. The failed study and
  successful isolated full-training repeat remain separate records.
- Source-level MPS norm-dispatch diagnosis showed why single-axis batched norms
  differ from native scalar norms. V4's two-nontrivial-axis buckets preserve the
  generic native reduction. Full native backward remains unchanged. All 32
  numerical checks and seven full-training equivalence pairs matched bitwise.
- V4's 70-run study saved 17.83% primary target elapsed time. Its quality upper
  99% bound, 0.01160035 BPB, narrowly missed the original 0.01 requirement. The
  combined result remains **failed**, despite time and equivalence passing.
- Before any further evaluation, one V5 replication of the unchanged V4 method
  was frozen publicly: 100 independent fresh pairs, original baseline and
  thresholds, stricter 99.5% intervals. The size uses the complete V4 variance,
  not favorable subset selection. No seeds can be added after outcomes and
  there is no optional stopping. See `REPLICATION.md` and `protocol-v5`.

Each study has a separate immutable source fingerprint and publicly tagged
pre-evaluation protocol. Analysis is outside those training-source scopes.
All weights remain local, and no implementation came from another author project.
