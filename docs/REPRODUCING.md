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
# V2: exact gradient and optimizer packing

The second experiment has its own immutable `protocol-v2` tag and
`artifacts/v2/frozen-study.json`. It compares exact packing against native **fused
AdamW**, with seven small-model pairs and five larger-model pairs. Its source lives
in `research/v2/`; it does not modify V1's frozen implementation. Read
[`PACKING.md`](PACKING.md) for the equivalence argument and restrictions.

Recompute V2 from public records without data downloads or local weights:

```sh
python -m analysis.v2.analyze --records-only
```

This writes `artifacts/v2/records-only-analysis` and explicitly marks local
checkpoint verification as unperformed. The published strict analysis verifies
the actual local checkpoint files as well as source/protocol/data hashes, paired
initialization and batch schedules, coverage and raw curve consistency.

For fresh training, use a separate checkout at `protocol-v2`, which precedes all
final V2 records. The campaign intentionally refuses to overwrite existing runs:

```sh
git worktree add ../cotangent-v2-replication protocol-v2
cd ../cotangent-v2-replication
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python analysis/fetch_pinned_data.py
.venv/bin/python -m pytest tests analysis/test_analysis.py research/v2/test_packed.py -q
.venv/bin/python -m research.v2.verify_mps
.venv/bin/python -m research.v2.campaign
```

Copy the release's `analysis/v2/` directory into the replication checkout if you
want its post-freeze statistics and figures; analysis is outside both frozen
training-source hash scopes. Then run `python -m analysis.v2.analyze` and
`python -m analysis.v2.figures`. Keep the exact protocol and training files intact.

The official WikiText-2 held-out corpus was already used by V1. V2 explicitly
reuses it as a numerical/quality regression check. Its new seeds and frozen run
order test timing replication; they do not create a fresh unseen dataset.
All V2 model snapshots remain local, as requested, alongside V1's snapshots.
