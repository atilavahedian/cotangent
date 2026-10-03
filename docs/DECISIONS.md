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
