# Reproducing Cotangent

## Environment

The measured training backend is Apple Metal on an M5 Pro with 24 GiB RAM,
macOS 27.0, Python 3.12.14 and PyTorch 2.14.1. Exact dependency versions are in
`requirements-research.txt`; actual hardware observations are in
`artifacts/environment.json`. CPU tests are supported. Frozen timing conclusions
must not be transplanted to another backend without a new experiment.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest tests analysis/test_analysis.py -q
```

## Pinned data and mathematical verification

```sh
.venv/bin/python analysis/fetch_pinned_data.py
.venv/bin/python -m scripts.verify_mps
```

The reproduction downloader refuses to replace an existing preparation. It downloads the
public pinned revision recorded in `data/manifest.json`, verifies each downloaded
parquet and prepared split, and does not alter the manifest. It does not require a
Hugging Face account. The exact byte preparation and split hashes are recorded.
The tests check dense derivatives, unbiased sampling, the variance expression,
exact activation-gradient propagation, full-rank projection and controller logic.
Metal verification checks actual FP32/BF16 custom and native gradients.

## Fresh frozen-study reproduction

Use a fresh Git worktree at `protocol-v1`, then copy the analysis directory from the
research release. The protocol tag includes no final result directories.
Do not overwrite this repository's published measurements. New results belong in
a separate checkout. Run the exact tagged source and its existing frozen protocol:

```sh
.venv/bin/python -m scripts.campaign
```

The runner trains 36 arms serially, validates the source/data freeze before and
after each run, and preserves failed or incomplete runs. It never silently falls
back from Metal to CPU. Completed runs may be preserved on restart; failed runs
require investigation rather than overwrite. Changing source/configuration
requires a new explicitly labeled exploratory study, not replacement of this one.

## Recompute analysis and figures

```sh
.venv/bin/python analysis/analyze.py
.venv/bin/python analysis/figures.py
.venv/bin/python analysis/make_report.py
```

Analysis deliberately refuses missing or failed planned arms, unmatched schedules,
changed frozen source, inconsistent validation curves and bad checkpoint hashes.
It retains censored target crossings. The public JSON/CSV results can be inspected
without model files. Recompute their statistics with
`python analysis/analyze.py --records-only`; this writes a separate output directory
and explicitly marks local checkpoint-file verification as unperformed. To rerun
the complete checkpoint integrity audit, retrain the protocol or obtain the exact
local snapshot archives from the author. All 36 snapshots are retained locally;
weights are not part of the public release. These snapshots support inference and
verification, not resuming AdamW training.

```sh
.venv/bin/python analysis/infer.py artifacts/checkpoints/final/small/adaptive-11.pt
```

Generated text is a qualitative smoke test, not evidence of general language
capability. Mathematical derivations are in `docs/MATHEMATICS.md`; method choices
and preserved development failures are in `docs/DECISIONS.md`.
