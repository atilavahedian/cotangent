"""Build the standalone manuscript and scientific figures from audited records."""
import json
import math
from pathlib import Path
from research.v6.runtime import ROOT, digest

NAMES = {
    "transformer-small": "Small transformer",
    "transformer-medium": "Medium transformer",
    "gated-gqa": "Gated GQA",
    "causal-conv": "Causal convolution",
}

def pct(value):
    return f"{100 * value:.2f}"

def ci(value, scale=100):
    return f"{scale * value['lower']:.2f}--{scale * value['upper']:.2f}"

def figures(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.facecolor": "white"})
    out = ROOT / "artifacts/v6/figures"
    out.mkdir(parents=True, exist_ok=True)
    rows = result["breadth"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for i, row in enumerate(rows):
        s, q = row["training_reduction"], row["test_difference"]
        axes[0].errorbar(100*s["mean"], i, xerr=[[100*(s["mean"]-s["lower"])], [100*(s["upper"]-s["mean"])]],
                        fmt="o", color="#126764", capsize=4, markersize=6)
        axes[1].errorbar(q["mean"], i, xerr=[[q["mean"]-q["lower"]], [q["upper"]-q["mean"]]],
                        fmt="o", color="#6c5880", capsize=4, markersize=6)
    for ax in axes:
        ax.set_yticks(range(4), [NAMES[r["model"]] for r in rows])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.16)
        ax.axvline(0, color="#a7a7a7", linewidth=.8)
    axes[0].set_title("Same compiled model, less gradient-handling time", loc="left", fontweight="bold")
    axes[0].set_xlabel("Paired training-time reduction (%)")
    axes[1].set_title("Full-test quality differences", loc="left", fontweight="bold")
    axes[1].set_xlabel("Candidate minus baseline (bits per byte)")
    axes[1].axvline(.01, color="#ae6f40", linestyle="--", linewidth=1, label="V5 tolerance, for reference")
    axes[1].legend(loc="lower right", frameon=False, fontsize=8)
    fig.suptitle("Twelve pairs per configuration · descriptive 99% confidence intervals", fontsize=10, y=1.02)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"compiled-comparison.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    keys = ("norm_only", "packing_only", "combined")
    labels = ("Norm batching", "Parameter packing", "Both")
    for i, (key, label) in enumerate(zip(keys, labels)):
        s = result["factorial"][key]
        ax.barh(i, 100*s["mean"], color=("#75aaa5", "#a8b3c2", "#126764")[i], height=.5)
        ax.errorbar(100*s["mean"], i, xerr=[[100*(s["mean"]-s["lower"])], [100*(s["upper"]-s["mean"])]],
                    fmt="none", ecolor="#343c43", capsize=4)
        ax.text(max(0, 100*s["upper"])+.5, i, f"{100*s['mean']:.2f}%", va="center", fontsize=9)
    lower = min(result["factorial"][k]["lower"]*100 for k in keys)
    upper = max(result["factorial"][k]["upper"]*100 for k in keys)
    ax.set_xlim(min(-1, lower-2), max(5, upper+6))
    ax.set_yticks(range(3), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#a7a7a7", linewidth=.8)
    ax.grid(axis="x", alpha=.16)
    ax.set_xlabel("Paired training-time reduction from native scalar handling (%)")
    ax.set_title("Eight factorial blocks · 800 steps per treatment · descriptive 99% intervals", loc="left", fontsize=10)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"factorial-ablation.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)

def main():
    result = json.loads((ROOT / "artifacts/v6/analysis/results.json").read_text())
    assert result["runs"] == 128 and result["integrity_checks_passed"]
    assert result["checkpoint_file_hashes_verified"]
    restored = json.loads((ROOT / "artifacts/v6/verification-snapshots.json").read_text())
    assert restored["status"] == "passed" and restored["snapshots"] == 128
    rows = result["breadth"]
    factorial = result["factorial"]
    minimum = min(r["training_reduction"]["mean"] for r in rows)
    maximum = max(r["training_reduction"]["mean"] for r in rows)
    all_positive = all(r["training_reduction"]["lower"] > 0 for r in rows)
    model_table = []
    for r in rows:
        cfg = json.loads((ROOT / "artifacts/v6/frozen-study.json").read_text())["configs"][r["model"]]
        model_table.append(f"{NAMES[r['model']]} & {r['parameters']:,} & {cfg['width']} & {cfg['layers']} & {r['parameter_tensors']} / {r['buckets']} " + r"\\")
    labels = {
        "norm_only": "Norm batching only", "packing_only": "Packing only",
        "combined": "Combined", "norm_main_effect": "Norm main effect",
        "packing_main_effect": "Packing main effect", "interaction": "Signed interaction",
    }
    fact_table = "\n".join(f"{labels[k]} & {pct(s['mean'])} & {ci(s)} " + r"\\" for k, s in factorial.items())
    breadth_table = "\n".join(f"{NAMES[r['model']]} & {r['native_training_seconds']:.2f} / {r['candidate_training_seconds']:.2f} & {pct(r['training_reduction']['mean'])} & {ci(r['training_reduction'])} " + r"\\" for r in rows)
    quality_table = "\n".join(f"{NAMES[r['model']]} & {r['native_test_bpb']:.4f} / {r['candidate_test_bpb']:.4f} & {r['test_difference']['mean']:+.5f} & {r['test_difference']['lower']:+.5f} to {r['test_difference']['upper']:+.5f} " + r"\\" for r in rows)
    plot = []
    for i, row in enumerate(rows, 1):
        s = row["training_reduction"]
        plot.append(r"\addplot+[only marks,mark=*,color=blue!50!black,error bars/.cd,x dir=both,x explicit] coordinates {" +
                    f"({100*s['mean']:.6f},{i}) +- ({100*(s['upper']-s['mean']):.6f},0)" + "};")
    tolerance_rows = [r for r in rows if r["test_difference"]["upper"] > .01]
    quality_abstract = ("The extension's descriptive quality intervals have upper bounds below "
                        "0.01 BPB, without introducing a new powered quality decision.")
    if tolerance_rows:
        quality_abstract = ("The extension's descriptive quality intervals do not establish the "
                            "0.01-BPB tolerance for every configuration.")
    quality_text = ("All four descriptive upper quality bounds are below the original 0.01-BPB tolerance. "
                    "The extension was not powered or preregistered as a new noninferiority decision, so this "
                    "observation is reported alongside the full intervals rather than promoted to a broader gate.")
    if tolerance_rows:
        names = ", ".join(NAMES[r["model"]] for r in tolerance_rows)
        quality_text = (f"The descriptive quality interval extends above the original 0.01-BPB tolerance for {names}. "
                        "The extension therefore does not establish that narrow quality tolerance for every "
                        "configuration. The controlled update checks and the V5 quality result remain separate evidence.")
    adverse = sum(r["slower_pairs"] for r in rows)
    quality_text += (f" Across the 48 breadth pairs, {adverse} have slower candidate training and "
                     f"{sum(r['worse_test_pairs'] for r in rows)} have worse candidate full-test BPB. "
                     "Both counts remain in the analysis.")
    quality_text = quality_text.replace("%", r"\%")
    norm, packing, both = [factorial[k] for k in ("norm_only", "packing_only", "combined")]
    fact_text = (f"Norm batching alone reduces training time by {pct(norm['mean'])}\\%, "
                 f"packing alone by {pct(packing['mean'])}\\%, and their combination by {pct(both['mean'])}\\%. "
                 "The main-effect and interaction contrasts in Table~\\ref{tab:factorial} use the complete "
                 "four-cell block. They describe this architecture and execution regime; an additive decomposition "
                 "is not assumed.")
    largest_factor = "norm batching" if norm["mean"] >= packing["mean"] else "parameter packing"
    discussion = (f"The factorial evidence attributes the larger single-factor saving to {largest_factor}. "
                  "This locates the benefit within the measured training step rather than inferring it from "
                  "a synthetic optimizer benchmark. "
                  "Compiler screening also changes the relevant baseline: on the small-model pilot, compiled "
                  "native training is faster than the original eager candidate. The extension therefore composes "
                  "the gradient layout with that same compiler path. Its comparison measures the added effect "
                  "of handling completed gradients, while leaving compiler arithmetic common to both arms.")
    resource = (f"The largest sampled driver allocation across breadth baselines is "
                f"{max(r['native_peak_driver_mib'] for r in rows)/1024:.2f} GiB, versus "
                f"{max(r['candidate_peak_driver_mib'] for r in rows)/1024:.2f} GiB for candidates. "
                "These are sampled allocation peaks, not a continuous memory trace. "
                "The additional buffers are part of the implementation and may matter more at larger scales. "
                "Total completion times, including setup, diagnostics, full evaluation and checkpoint writing, "
                "are also reported in the public paired records.")
    breadth_text = (f"With the same compiled model in each pair, mean training-time reductions range from "
                    f"{pct(minimum)}\\% to {pct(maximum)}\\%. " +
                    ("Every per-configuration descriptive timing interval has a positive lower bound. " if all_positive else
                     "The individual intervals indicate where the timing evidence is conclusive and where it remains uncertain. ") +
                    "Table~\\ref{tab:breadth} reports both absolute times and paired reductions. "
                    "These results extend the implementation evidence across layouts; they do not replace "
                    "the independent time-to-quality decision in V5.")
    replacements = {
        "@@EXTENSION_ABSTRACT@@": (f"A separately frozen 128-run extension isolates norm batching from packing "
                                   f"and evaluates four configurations against training-only-selected compiled baselines. "
                                   f"Using the same compiled model, mean training-time reductions range from "
                                   f"{pct(minimum)}\\% to {pct(maximum)}\\%. "),
        "@@EXTENSION_QUALITY_ABSTRACT@@": quality_abstract,
        "@@ARCH_TABLE@@": "\n".join(model_table),
        "@@FACTORIAL_TEXT@@": fact_text,
        "@@FACTORIAL_TABLE@@": fact_table,
        "@@BREADTH_TEXT@@": breadth_text,
        "@@BREADTH_TABLE@@": breadth_table,
        "@@QUALITY_TABLE@@": quality_table,
        "@@QUALITY_TEXT@@": quality_text,
        "@@RESOURCE_TEXT@@": resource,
        "@@DISCUSSION_TEXT@@": discussion,
        "@@BREADTH_PLOT@@": "\n".join(plot),
        "@@PLOT_MIN@@": str(min(0, math.floor(min(r["training_reduction"]["lower"]*100 for r in rows)-3))),
        "@@PLOT_MAX@@": str(math.ceil(max(r["training_reduction"]["upper"]*100 for r in rows)+3)),
        "@@RELEASE_LINK@@": r"The paper and portable report are included in release \code{v0.6.0-research}.",
    }
    text = (ROOT / "analysis/v6/manuscript.template.tex").read_text()
    for marker, replacement in replacements.items():
        assert marker in text
        text = text.replace(marker, replacement)
    assert "@@" not in text
    text = text.replace(r"\begin{document}", r"\hypersetup{pdftitle={Cotangent: Preserving Reduction Arithmetic in Batched Gradient Handling},pdfauthor={Atila Vahedian}}" + "\n" + r"\begin{document}")
    paper = ROOT / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    (paper / "manuscript.tex").write_text(text)
    figures(result)
    manifest = dict(tex_sha256=digest(paper/"manuscript.tex"), v5_result_sha256=digest(ROOT/"artifacts/v5/analysis/results.json"),
                    v6_result_sha256=digest(ROOT/"artifacts/v6/analysis/results.json"), author="Atila Vahedian",
                    paper_status="Technical report", weights_published=False)
    (ROOT / "artifacts/v6/publication.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))

if __name__ == "__main__":
    main()
