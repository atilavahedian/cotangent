"""Publish the paper-facing project page and repository overview."""
import base64
import html
import json
import shutil
from research.v6.runtime import ROOT, digest
from analysis.v6.paper import NAMES

def image_data(name):
    return "data:image/png;base64," + base64.b64encode((ROOT/"artifacts/v6/figures"/name).read_bytes()).decode()

def main():
    result = json.loads((ROOT/"artifacts/v6/analysis/results.json").read_text())
    publication_path = ROOT/"artifacts/v6/publication.json"
    publication = json.loads(publication_path.read_text())
    assert digest(ROOT/"paper/manuscript.tex") == publication["tex_sha256"]
    assert (ROOT/"paper/manuscript.pdf").is_file()
    rows, factorial = result["breadth"], result["factorial"]
    minimum = min(r["training_reduction"]["mean"] for r in rows)*100
    maximum = max(r["training_reduction"]["mean"] for r in rows)*100
    breadth, quality, markdown = [], [], []
    for r in rows:
        s, q = r["training_reduction"], r["test_difference"]
        breadth.append("<tr><td>"+NAMES[r["model"]]+"</td><td class='num'>"+f"{r['parameters']/1e6:.2f}M"+"</td><td class='num'>"+
                       f"{r['native_training_seconds']:.2f} / {r['candidate_training_seconds']:.2f} s"+"</td><td class='num'>"+
                       f"{100*s['mean']:.2f}%"+"</td><td class='num'>"+f"{100*s['lower']:.2f}–{100*s['upper']:.2f}%"+"</td></tr>")
        quality.append("<tr><td>"+NAMES[r["model"]]+"</td><td class='num'>"+f"{r['native_test_bpb']:.4f} / {r['candidate_test_bpb']:.4f}"+
                       "</td><td class='num'>"+f"{q['mean']:+.5f}"+"</td><td class='num'>"+f"{q['lower']:+.5f} to {q['upper']:+.5f}"+
                       "</td><td class='num'>"+f"{100*r['total_reduction']['mean']:.2f}%"+"</td></tr>")
        markdown.append(f"| {NAMES[r['model']]} | {r['parameters']/1e6:.2f}M | {100*s['mean']:.2f}% | {100*s['lower']:.2f}–{100*s['upper']:.2f}% | {q['mean']:+.5f} BPB |")
    outside = [NAMES[r["model"]] for r in rows if r["test_difference"]["upper"]>.01]
    quality_note = ("The descriptive quality interval extends above 0.01 BPB for "+", ".join(outside)+". The narrow tolerance is therefore not established for every configuration."
                    if outside else "All four descriptive upper quality bounds fall below 0.01 BPB. This is reported with the complete intervals rather than promoted to a new powered quality decision.")
    text = (ROOT/"analysis/v6/site.template.html").read_text()
    replacements = {
        "@@COMPILED_RANGE@@": f"{minimum:.2f}–{maximum:.2f}%",
        "@@BREADTH_ROWS@@": "".join(breadth),
        "@@QUALITY_ROWS@@": "".join(quality),
        "@@QUALITY_NOTE@@": html.escape(quality_note),
        "@@MEMORY_NOTE@@": html.escape("Sampled driver allocation includes allocator/cache effects. " +
            " ".join(f"{NAMES[r['model']]} peaks: {r['native_peak_driver_mib']:.0f} MiB baseline / {r['candidate_peak_driver_mib']:.0f} MiB Cotangent."
                     for r in rows if r["model"] in ("transformer-small", "causal-conv")) +
            " The complete memory and completion-time table is in the paper."),
        "@@COMPILED_IMAGE@@": image_data("compiled-comparison.png"),
        "@@FACTORIAL_IMAGE@@": image_data("factorial-ablation.png"),
        "@@COMPILED_CAPTION@@": f"All 48 paired observations are retained, including {sum(r['slower_pairs'] for r in rows)} slower candidate pairs and {sum(r['worse_test_pairs'] for r in rows)} pairs with worse full-test quality.",
        "@@FACTORIAL_CAPTION@@": f"Norm batching alone saves {100*factorial['norm_only']['mean']:.2f}%, parameter packing {100*factorial['packing_only']['mean']:.2f}%, and their combination {100*factorial['combined']['mean']:.2f}%.",
        "@@PAIRS_JSON@@": json.dumps(result["pairs"], separators=(",",":")),
    }
    for marker, value in replacements.items():
        assert marker in text
        text = text.replace(marker, value)
    assert "@@" not in text
    archive = ROOT/"docs/archive/v5.html"
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        assert digest(ROOT/"docs/index.html") == "ea233fee1d4c0dfaf34d2a085d06952d8bf5fb33f4fd5200b0c852ecbe38131f"
        shutil.copyfile(ROOT/"docs/index.html", archive)
    (ROOT/"docs/index.html").write_text(text)
    (ROOT.parent/"cotangent.html").write_text(text)
    shutil.copyfile(ROOT/"paper/manuscript.pdf", ROOT/"docs/cotangent-paper.pdf")
    shutil.copyfile(ROOT/"paper/manuscript.pdf", ROOT.parent/"cotangent-paper.pdf")
    fence = chr(96)*3
    overview = f"""# Cotangent

Numerically matched gradient batching for efficient training on Apple Metal.

Cotangent studies the work between backward and the next parameter update:
gradient norms, clipping and fused AdamW dispatch. It retains the full gradients,
batches equal-length norm reductions, and consistently packs optimizer coordinates.

[Paper](paper/manuscript.pdf) · [Abstract and citation](https://atilavahedian.github.io/cotangent/paper.html) · [Project page](https://atilavahedian.github.io/cotangent/) ·
[LaTeX](paper/manuscript.tex) · [Results](artifacts/v6/analysis/results.json) ·
[Reproduce](docs/REPRODUCING.md)

## Results

The independent eager replication contains **100 paired seeds / 200 runs**.
Time to the fixed quality target fell **15.03% (99.5% CI, 13.50–16.56%)**;
fixed-budget training time fell **16.97% (16.26–17.69%)**. The upper full-test
degradation bound, **0.00728 BPB**, is below the predeclared **0.01-BPB** tolerance.

The stronger extension adds **128 runs**: 32 factorial treatments and 96
architecture comparisons. Its baseline screening selected TorchInductor for
all four configurations. Both arms therefore use the **same compiled model**;
Cotangent changes gradient handling after backward.

| Configuration | Parameters | Training-time reduction | 99% interval | Mean test change |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(markdown)}

These are per-configuration descriptive paired intervals from twelve seeds,
1,200 steps per arm. Positive test change means worse quality. Full intervals,
absolute times, memory observations and every adverse pair are in the
[paper](paper/manuscript.pdf) and [raw records](artifacts/v6/analysis/pairs.csv).
{quality_note}

![Compiled comparisons and quality differences](artifacts/v6/figures/compiled-comparison.png)

## What accounts for the saving

The eight-block factorial experiment separates norm batching from parameter
packing. Norm batching alone saves **{100*factorial['norm_only']['mean']:.2f}%**,
packing alone **{100*factorial['packing_only']['mean']:.2f}%**, and their
combination **{100*factorial['combined']['mean']:.2f}%** on the eager small model.
The complete main effects and interaction retain all four treatments.

![Factorial ablation](artifacts/v6/figures/factorial-ablation.png)

Native clipping first computes a norm for every parameter, then combines those
scalar norms. A bucket view of **B × (K/2) × 2** keeps the pinned MPS generic
reduction path for tested even lengths. Scalar norms return to parameter order
before the original joint norm. Short and odd lengths use native fallbacks.
Complete gradients are then concatenated, clipped and updated with native
fused AdamW. Every copy and temporary buffer counts.

[Method and mathematical argument](docs/BATCHING.md) ·
[Extension design](docs/EXTENSION.md) · [Related work](docs/RELATED_WORK.md)

## Numerical checks

- 128 full-model extension checks: four configurations, FP32/BF16 forward
  precision, four treatments and four consecutive updates. Losses, gradients,
  clipping norms, parameters and both moment buffers match the reference bitwise.
- 288 norm cases: 32 lengths, three bucket sizes and three gradient dtypes.
- 16 compiler composition checks with matched deterministic derivatives.
- Seven earlier full-training controlled pairs finish with identical model
  hashes and evaluation scores.
- Every one of 128 extension snapshots is restored and checked locally.
  **All weight files remain local.**

Controlled checks isolate update arithmetic. Performance runs retain native
embedding backward, whose accumulation can vary across executions.

## Reproduce the public evidence

The standard-library audit requires no GPU, dataset download or weights:

{fence}sh
python3 analysis/v6/public_audit.py
{fence}

It verifies immutable source and record hashes and independently recomputes the
paired statistics. It does not execute Metal training or inspect local snapshots.
For complete record-based analysis:

{fence}sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-research.txt
.venv/bin/python -m analysis.v5.analyze --records-only
.venv/bin/python -m analysis.v6.analyze --records-only
{fence}

Fresh training uses a separate checkout at **protocol-v6**, which predates its
official outcomes. Published observations and frozen sources must remain intact.
[Training and paper reproduction](docs/REPRODUCING.md).

## Scope

One Apple M5 Pro, PyTorch 2.14.1 MPS, byte language models from 3.35M to 19.28M
parameters, and reused WikiText-2 splits. Compiler measurements use warm caches
and per-step synchronization. Gradient handling and native fused AdamW remain
outside the compiled model call. Dense contiguous complete gradients and one
hyperparameter group are required. Full activations and derivative products
remain; additional buffers consume memory.

Other Apple devices, CUDA, distributed training and larger model scales remain
untested. The V5 quality decision and V6 descriptive comparisons have different
roles. [The experimental record](docs/EXPERIMENTS.md) preserves earlier failures.

## Repository

| Path | Contents |
| --- | --- |
| paper/ | Standalone LaTeX manuscript and PDF |
| research/v4/ | Frozen reference implementation |
| research/v6/ | Frozen ablations, architectures and compiler composition |
| analysis/v6/ | Statistics, independent audit, figures and publication tools |
| artifacts/v5/ and artifacts/v6/ | Protocols and complete measurement records |
| docs/ | Method, reproduction instructions and project page |

Research by **Atila Vahedian**. [Citation](CITATION.cff).
Code and authored report are MIT licensed. Dependencies and dataset retain their
terms; corpus content and model weights are not redistributed.
"""
    (ROOT/"README.md").write_text(overview)
    experiments = """# Experimental record

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
"""
    (ROOT/"docs/EXPERIMENTS.md").write_text(experiments)
    citation = """cff-version: 1.2.0
message: "Please cite the accompanying Cotangent technical report."
title: "Cotangent: Preserving Reduction Arithmetic in Batched Gradient Handling"
authors:
  - family-names: Vahedian
    given-names: Atila
version: 0.6.0
date-released: 2026-10-04
url: "https://github.com/atilavahedian/cotangent"
repository-code: "https://github.com/atilavahedian/cotangent"
license: MIT
preferred-citation:
  type: unpublished
  title: "Cotangent: Preserving Reduction Arithmetic in Batched Gradient Handling"
  authors:
    - family-names: Vahedian
      given-names: Atila
  year: 2026
  url: "https://atilavahedian.github.io/cotangent/cotangent-paper.pdf"
"""
    (ROOT/"CITATION.cff").write_text(citation)
    publication.update(pdf_sha256=digest(ROOT/"paper/manuscript.pdf"), html_sha256=digest(ROOT/"docs/index.html"),
                       pdf_bytes=(ROOT/"paper/manuscript.pdf").stat().st_size,
                       html_bytes=(ROOT/"docs/index.html").stat().st_size)
    publication_path.write_text(json.dumps(publication,indent=2)+"\n")
    print(json.dumps(publication))

if __name__ == "__main__":
    main()
