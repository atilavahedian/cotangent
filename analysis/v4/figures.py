"""Publication figures from the complete frozen V4 analysis, without refitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
result = json.loads((ROOT / "artifacts/v4/analysis/results.json").read_text())
assert result["runs"] == 70 and result["integrity_checks_passed"]
out = ROOT / "artifacts/v4/figures"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
colors = {"native-fused": "#595e68", "native-bucket": "#006c67"}

def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), layout="constrained")
for ax, size, label in zip(axes, ("small", "medium"),
        ("3.35M · 25 paired seeds · primary", "19.28M · three pairs · descriptive")):
    for arm in colors:
        points = [p for p in result["mean_learning_curves"] if (p["size"], p["arm"]) == (size, arm)]
        ax.plot([p["elapsed_seconds"] for p in points], [p["validation_bpb"] for p in points],
                color=colors[arm], label=points[0]["method"], linewidth=2)
    ax.axhline(3.2, color="#ac5c10", linestyle="--", linewidth=1, label="Frozen target: 3.2 BPB")
    ax.set(title=label, xlabel="Elapsed seconds incl. setup + probes", ylabel="Validation BPB (lower is better)")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
fig.suptitle("Cotangent V4 · native backward + numerically matched norm buckets")
save(fig, "learning-curves")

fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.6), layout="constrained")
for index, (group,label) in enumerate((("small","3.35M · 25 pairs · primary"),("medium","19.28M · three pairs · descriptive"))):
    c=result[group]
    for ax,key,pairkey,scale,threshold,xlabel in (
        (axes[index,0],"elapsed_reduction_fraction","elapsed_reduction_fraction",100,10,"Paired time saved (%) · right is faster"),
        (axes[index,1],"quality_difference_bpb","test_difference_bpb",1,.01,"Candidate − native BPB · left is better")):
        value=c[key]
        if value is None:
            ax.text(.5,.5,"Target censored",transform=ax.transAxes,ha="center")
        else:
            ax.scatter([scale*p[pairkey] for p in c["pairs"]],[0]*len(c["pairs"]),marker="|",s=150,color="#8eaea8",alpha=.8)
            ax.errorbar(scale*value["mean"],0,xerr=[[scale*(value["mean"]-value["lower"])],[scale*(value["upper"]-value["mean"])]],fmt="o",color="#006c67",capsize=5)
        ax.axvline(0,color="#777",linewidth=.8)
        ax.axvline(threshold,color="#ac5c10",linestyle="--",linewidth=1)
        ax.set(yticks=[],ylim=(-.6,.6),xlabel=xlabel,title=label)
        ax.grid(axis="x",alpha=.15)
axes[0,0].set_title("Primary · elapsed to the same 3.2-BPB target")
axes[0,1].set_title("Primary · full reused-corpus test quality")
axes[1,0].set_title("Larger model · descriptive timing, n=3")
axes[1,1].set_title("Larger model · descriptive quality, n=3")
fig.suptitle("Means and two-sided 99% paired Student-t intervals · ticks: every pair\nSeparate axes preserve the primary detail and larger-model uncertainty")
save(fig,"paired-outcomes")
print(str(out.relative_to(ROOT)))
