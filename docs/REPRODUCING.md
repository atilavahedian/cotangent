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

## Compiled comparisons and factorial ablation: V6

The separately frozen extension contains 128 official runs: eight four-treatment
factorial blocks and twelve paired seeds for each of four model configurations.
Its training-only pilot screening selected TorchInductor for every breadth
baseline. Baseline and candidate use the same compiled model; gradient handling
and native fused AdamW run after its backward call. Compiler caches are warm.

Audit the public release with only Python's standard library:

```sh
python3 analysis/v6/public_audit.py
```

This checks source and observation hashes, independently recomputes paired
intervals from raw times and quality scores, and checks the paper/report hashes.
It requires no dataset, PyTorch installation, GPU or weight download. It does
not verify local weight files or execute training. Complete record analysis uses
the pinned research dependencies:

```sh
python -m analysis.v6.analyze --records-only
```

For new measurements, use a separate checkout at **protocol-v6**. That tag
predates the official outcomes. Install dependencies, retrieve the pinned data,
run numerical checks and reproduce the pilot cache before the official campaign:

```sh
git worktree add ../cotangent-v6-replication protocol-v6
cd ../cotangent-v6-replication
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python analysis/fetch_pinned_data.py
.venv/bin/python -m research.v6.verify
.venv/bin/python -m research.v6.pilots
.venv/bin/python -m research.v6.compose
.venv/bin/python -m research.v6.campaign
```

The tag contains the original pilot and verification evidence. The verification
and pilot runners refuse to overwrite those observations. Before rerunning
these diagnostics in the replication checkout, preserve the tagged files and
move **artifacts/v6/pilots/**, **verification-final.json** and
**verification-compiled.json** out of their expected output locations. The
official frozen protocol identifies their original hashes. Restore the tagged
evidence before the official campaign; leave the regenerated compiler cache in
place. Do not regenerate the frozen study specification.

Alternatively, reproduce only the official campaign from the tagged protocol;
its first model calls populate missing compiler cache entries, so setup timing
will differ from the published warm-cache experiment. Training refuses changed
frozen source, mismatched protocol jobs and overwritten observations. All GPU
jobs must run serially on the designated device.

Copy the release's **analysis/v6/** directory into that replication checkout
afterward for post-freeze statistics and snapshot restoration:

```sh
python -m analysis.v6.analyze
python -m analysis.v6.restore
```

Strict analysis checks actual checkpoint-file hashes. Restoration reconstructs
all 128 models, checks their parameter hashes and runs finite CPU inference.
Records-only analysis writes a separate directory and leaves weight-file checks
unperformed. Every weight file stays local. V6 intervals are descriptive per
configuration; they do not replace V5's powered quality decision.

Do not modify frozen source under **cotangent/**, **scripts/**, **tests/**,
**configs/** or Python files under **research/v2/** through **research/v6/**.
Analysis and document generation are outside those fingerprint scopes.

## Paper and project page

The manuscript is standalone LaTeX: figure coordinates and bibliography entries
are embedded, so compilation needs no additional project files. The published
source and PDF are in **paper/**. With strict analysis and restored snapshots,
rebuild its measured tables and scientific figures using:

```sh
MPLCONFIGDIR=./.mpl-cache python -m analysis.v6.paper
```

Open **paper/manuscript.tex** in the Codex LaTeX editor for its live PDF preview,
or export **paper/manuscript.pdf** using an existing Tectonic/TeX installation.
After exporting the PDF, build the repository overview and portable report:

```sh
python -m analysis.v6.site
python analysis/v6/public_audit.py
```

The page uses embedded figures and local JavaScript; it needs no server-side
runtime. **docs/index.html** is the GitHub Pages entry point,
**docs/cotangent-paper.pdf** is its paper link, and **docs/archive/v5.html**
preserves the preceding report. Rebuilding the manuscript changes its hash;
export its current PDF before rebuilding the page and publication manifest.

## Independent fixed-size replication: V5

V5 is the single independent powered replication of unchanged V4. Its
[protocol](../artifacts/v5/frozen-study.json) preregisters 100 fresh pairs (200 runs),
99.5% intervals, and the original 10%/0.01 thresholds.
[Sample-size planning](REPLICATION.md) uses all V4 variability.

Recompute its complete public statistics without data or weights:

```sh
python -m analysis.v5.analyze --records-only
```

For fresh measurements, create a separate checkout at **`protocol-v5`**, install
the dependencies and retrieve pinned data as above. That tag contains no V5
final observations. Copy the release's `analysis/v5/` into that checkout afterward
for post-freeze analysis. Training checks all five immutable source fingerprints,
refuses overwrite and runs every planned job serially.

```sh
python analysis/fetch_pinned_data.py
python -m research.v5.campaign
python -m analysis.v5.analyze
python -m analysis.v5.verify_snapshots
MPLCONFIGDIR=./.mpl-cache python -m analysis.v5.figures
```

Strict analysis verifies actual local weight-file hashes. Records-only analysis
writes a separate output and explicitly leaves that check unperformed. The V4
full-training equivalence proof is unchanged and remains a required prerequisite.
Do not edit any frozen `cotangent/`, `scripts/`, `tests/`, `configs/` source or
Python under `research/v2/` through `research/v5/`. Use separate checkouts for new
results; never replace published observations. Weights remain local.

## Earlier independent study: V4

The latest method and numerical assumptions are in [`BATCHING.md`](BATCHING.md).
V4's source is frozen at `protocol-v4`, before all official evaluation. Its
70-run protocol is in `artifacts/v4/frozen-study.json`: 25 primary native-backprop
pairs, three larger-model descriptive pairs and seven complete-training numerical
equivalence pairs. Primary intervals are 99%; time and quality thresholds retain
the earlier 10% and 0.01-BPB requirements, with an additional exactness gate.

Recompute from public records without any dataset or weight download:

```sh
python -m analysis.v4.analyze --records-only
```

For fresh measurements use a separate checkout at `protocol-v4`. Install the
pinned research dependencies as above and obtain the data, then run:

```sh
python analysis/fetch_pinned_data.py
python -m pytest tests analysis/test_analysis.py research/v2/test_packed.py research/v3/test_embedding.py -q
python -m research.v4.verify
python -m research.v4.campaign
```

The campaign checks all four source freezes, refuses to replace existing runs,
and performs all 70 jobs serially. Copy the release's `analysis/v4/` into the
separate checkout for post-freeze statistics, restoration and scientific figures:

```sh
python -m analysis.v4.analyze
python -m analysis.v4.verify_snapshots
MPLCONFIGDIR=./.mpl-cache python -m analysis.v4.figures
```

Analysis is outside every frozen training-source hash scope. Do not edit tagged
`cotangent/`, `scripts/`, `tests/`, `configs/` or any `research/v2/`, `v3/`, `v4/`
Python source. The strict audit checks every local weight-file hash; records-only
mode explicitly marks that part unverified and writes a separate output directory.
Weights remain local. The official WikiText-2 corpus is openly reused as a quality
regression check; new seeds do not create a fresh unseen dataset.

## Earlier protocols: V1 data and mathematical verification

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

# V3: deterministic exact embedding adjoints

V3 uses its own `protocol-v3` tag, frozen source in `research/v3/`, and
`artifacts/v3/frozen-study.json`. Its 41 runs comprise 15 small-model primary
pairs, five deterministic-control ablations and three larger-model descriptive
pairs. The descriptive comparisons cannot replace the primary decision.
[`DETERMINISM.md`](DETERMINISM.md) explains the exact incidence adjoint, numerical
diagnosis, restrictions and retained Metal prototypes.

Recompute the primary decision from public records, without weights or downloads:

```sh
python -m analysis.v3.analyze --records-only
```

For new measurements, create a separate checkout at `protocol-v3`, install the
same dependencies and retrieve the pinned data as above. Copy the release's
`analysis/v3/` into that checkout for post-freeze analysis. Do not modify the
tagged `cotangent/`, `scripts/`, `tests/`, `configs/`, `research/v2/` or
`research/v3/` files: the campaign verifies all three source freezes.

```sh
python -m pytest tests analysis/test_analysis.py research/v2/test_packed.py research/v3/test_embedding.py -q
python -m research.v3.verify_incidence
python -m research.v3.verify_segmented
python -m research.v3.campaign
python -m analysis.v3.analyze
python -m analysis.v3.verify_snapshots
MPLCONFIGDIR=./.mpl-cache python -m analysis.v3.figures
```

The optional `python -m analysis.v3.repeat_candidate` is a post-study diagnostic.
It runs the unchanged frozen candidate in a separate local mirror, matches its
inputs, and compares full trained parameter hashes. It cannot add a primary
pair or change the frozen gates. It refuses to overwrite an earlier diagnostic.
Published model files remain local; public summaries contain their hashes.
The official corpus was already evaluated in V1 and V2, so the quality result
is a reused-corpus regression check rather than unseen-domain generalization.
