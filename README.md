# Cotangent

**Can adaptive gradient sampling make transformer training faster?**

Cotangent is a standalone study of variance-aware weight-gradient approximation.
It includes newly written custom autograd operations, mathematical derivations,
matched transformer training, frozen success criteria, preserved negative results,
and an interactive HTML research report. No implementation, weights, preprocessing,
configuration, or results were inherited from the author's other projects.

**Primary result: no demonstrated advantage over native BF16 backpropagation.**
Across five paired seeds, the adaptive candidate's elapsed time to a common quality
target was **5.2% worse on average**. The paired 95% interval for time reduction was
**−11.4% to +1.0%**; the frozen requirement was at least 10% improvement with its
interval above zero. Final test degradation averaged **0.0089 bits per byte**,
with a 95% interval **−0.0133 to +0.0311**. Its upper bound narrowly exceeded the
predeclared 0.03-BPB tolerance. Both gates failed. These results apply to the
measured local setup; they do not prove universal inferiority.

All **36 frozen runs completed**. The larger-model comparison also showed no speed
advantage: mean paired time reduction was **−11.9%** (95% CI −35.0% to +11.2%).
Every saved model was restored and its trained parameter hash verified; all 11
derivative and analysis tests passed.

[Read the interactive report](https://atilavahedian.github.io/cotangent/) ·
[Download the standalone HTML](https://github.com/atilavahedian/cotangent/releases/tag/v0.1.0-research) ·
[Inspect the complete results](artifacts/analysis/results.json)

## Method

For a linear layer, the weight gradient is `G = DᵀX`, a sum of rank-one row
contributions. Cotangent samples those contributions with replacement, weighting
by inverse sampling probability. Probabilities depend on the product of activation
and incoming-gradient row norms. This estimator is conditionally unbiased before
floating-point rounding, and its variance has an analytic expression.

An audit controller estimates the row budget required for a relative RMS error
of 0.5. It audits every 32 steps, batches its device-to-host statistics, and uses
native dense backward when the requested budget is at least 75% of the rows.
Activation gradients remain exact. This preserves the chain-rule signal but
retains full activations and limits the available compute saving.

The study tests six arms: native exact, exact custom backward, uniform 25% row
sampling, importance 25% sampling, a 25% Hadamard projection control, and the
adaptive candidate. Classical sampling and low-rank backpropagation are prior art;
this project contributes the investigated controller, implementation and evidence,
without claiming a new general backpropagation algorithm.

## Evidence and project structure

- [`docs/MATHEMATICS.md`](docs/MATHEMATICS.md): estimator, variance, compute ceiling,
  controller assumptions, and a limited SGD convergence statement.
- [`artifacts/frozen-study.json`](artifacts/frozen-study.json): outcome definitions,
  paired seeds, resource limits, and source/data hashes frozen before evaluation.
- [`cotangent/backward.py`](cotangent/backward.py): custom backward and controller.
- [`cotangent/model.py`](cotangent/model.py), [`cotangent/train.py`](cotangent/train.py):
  fresh byte-level transformer and matched training/evaluation harness.
- [`artifacts/final`](artifacts/final): per-seed raw curves, summaries and logs.
- [`analysis`](analysis): integrity audit, paired analysis, figures, report bindings,
  data reproduction, inference and checkpoint packaging.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): pilot-driven choices and retained failures.

The main model has **3.35M parameters**, trained for **2,000 steps** across five
paired seeds. The larger model has **19.28M parameters**, trained for **1,500 steps**
across three paired seeds. All arms share initialization, batch schedules, data,
precision, optimizer settings and evaluation windows. Official WikiText-2 validation
and test splits were reserved until the final protocol freeze.

Timing includes setup, sampling, audits, synchronization, AdamW, clipping, guards
and scheduled validation. The primary target is the first scheduled 65,536-byte
validation probe at or below 3.2 bits per byte, without interpolation. Missing
crossings remain censored. Final test quality uses the entire official test split.

## Reproduce

Measured hardware: Apple M5 Pro, 24 GiB RAM, macOS 27.0, PyTorch 2.14.1, Metal.
Training is local and serial. The study does not establish CUDA, distributed or
large-model performance. See the [complete reproduction instructions](docs/REPRODUCING.md).

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest tests analysis/test_analysis.py -q
.venv/bin/python analysis/fetch_pinned_data.py
```

Retrain from the `protocol-v1` tag in a separate checkout, with the release's
analysis directory copied in for pinned data retrieval. Do not overwrite published
results. Model snapshots contain weights and metadata, not optimizer states.
All 36 verified snapshots are retained locally; their hashes and restoration
results are public. Weight files are not included in the public release.

Recompute statistics from the public records without retraining or weight files:

```sh
.venv/bin/python analysis/analyze.py --records-only
```

This writes a separate analysis directory and explicitly marks checkpoint-file
verification as unperformed. The published strict audit verified all local files.

## Related work and scope

- [Randomized matrix multiplication, Drineas–Kannan–Mahoney 2006](https://doi.org/10.1137/S0097539704442684)
- [LBP-WHT: low-rank backpropagation for adaptation, NeurIPS 2023](https://arxiv.org/abs/2309.15275)
- [INSTANT: gradient and activation compression, ICLR 2026](https://github.com/hieu-trannn/INSTANT)
- [APOLLO: memory-efficient optimizer states](https://arxiv.org/abs/2412.05270)
- [Moonwalk: inverse-forward differentiation, AISTATS 2026](https://proceedings.mlr.press/v300/krylov26a.html)

The Hadamard arm is an independently written mechanism control inspired by LBP-WHT,
not an official reproduction. INSTANT, APOLLO and Moonwalk were not directly
benchmarked. This study does not support a claim of beating those methods or the
current state of the art.

## License

The newly written ML implementation, analysis, tests and authored explanation are
MIT licensed. General-purpose report UI infrastructure and external dependencies
retain their own terms. The [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext)
retains its source license and is not redistributed in this repository.
