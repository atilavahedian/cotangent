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
