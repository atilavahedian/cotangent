"""Publication-ready figures generated only from the audited frozen results."""
import json
from collections import defaultdict
from pathlib import Path
import statistics
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze import T95, LABELS

ROOT=Path(__file__).resolve().parents[1]
COLORS={"native":"#16324f","exact_custom":"#62869f","adaptive":"#cb5c32",
        "uniform":"#89974a","importance":"#7257a0","wht":"#9e8990"}


def main():
    r=json.loads((ROOT/"artifacts/analysis/results.json").read_text())
    out=ROOT/"artifacts/figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"svg.fonttype":"none","savefig.dpi":180})
    def save(fig,name):
        fig.savefig(out/f"{name}.svg",bbox_inches="tight")
        fig.savefig(out/f"{name}.png",bbox_inches="tight")
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4.7))
    points=defaultdict(list)
    for p in r["learning_curves"]:
        if p["group"]=="small":
            points[p["mode"],p["step"]].append(p["validation_bpb"])
    for mode in LABELS:
        steps=sorted(step for (m,step) in points if m==mode)
        means=[statistics.mean(points[mode,step]) for step in steps]
        bands=[T95[5]*statistics.stdev(points[mode,step])/math.sqrt(5) for step in steps]
        ax.plot(steps,means,label=LABELS[mode],color=COLORS[mode],lw=1.8)
        ax.fill_between(steps,[v-e for v,e in zip(means,bands)],[v+e for v,e in zip(means,bands)],color=COLORS[mode],alpha=.10)
    ax.axhline(3.2,color="#777",ls=":",lw=1,label="Frozen probe target")
    ax.set(xlabel="Optimizer step",ylabel="Validation bits per byte (lower is better)",
           title="Matched transformer training · 3.35M parameters · five seeds")
    ax.legend(ncol=2,fontsize=9,frameon=False)
    ax.grid(alpha=.12)
    fig.text(.12,-.025,"Bands: two-sided 95% Student-t intervals across seeds; 65,536-byte validation probes.",fontsize=8)
    save(fig,"learning-curves")
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.9))
    p=r["primary"]
    seeds=[str(row["seed"]) for row in p["pairs"]]
    gain=[100*row["elapsed_reduction_fraction"] if row["both_reached"] else float("nan") for row in p["pairs"]]
    diff=[row["test_difference_bpb"] for row in p["pairs"]]
    axes[0].barh(seeds,gain,color=COLORS["adaptive"])
    axes[0].axvline(0,color="#333",lw=.8)
    axes[0].axvline(10,color="#777",ls=":",label="10% minimum")
    axes[0].set(xlabel="Time reduction (%) · positive is faster",ylabel="Paired seed",title="Elapsed time to 3.2 BPB")
    axes[0].legend(frameon=False,fontsize=8)
    axes[1].barh(seeds,diff,color=COLORS["adaptive"])
    axes[1].axvline(0,color="#333",lw=.8)
    axes[1].axvline(.03,color="#777",ls=":",label="0.03 BPB margin")
    axes[1].set(xlabel="Cotangent − native test BPB · lower is better",title="Full test-split quality")
    axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    save(fig,"paired-outcomes")
    fig,ax=plt.subplots(figsize=(8.5,3.8))
    for seed in sorted({p["seed"] for p in r["sampling_observations"] if p["group"]=="small"}):
        points=[p for p in r["sampling_observations"] if p["group"]=="small" and p["seed"]==seed]
        ax.plot([p["step"] for p in points],[p["sample_fraction"] for p in points],lw=1.1,alpha=.75,label=f"Seed {seed}")
    ax.set(ylim=(0,1.04),xlabel="Optimizer step",ylabel="Logged mean row fraction",
           title="Controller diagnostic · one means dense computation")
    ax.legend(ncol=5,fontsize=8,frameon=False)
    ax.grid(alpha=.12)
    fig.text(.12,-.025,"Unweighted layer means at logged steps, including exact audits; not a full-run FLOP saving.",fontsize=8)
    save(fig,"controller-budget")
    print(f"Saved three SVG/PNG figure pairs in {out.relative_to(ROOT)}")


if __name__=="__main__":
    main()
