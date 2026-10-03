"""Export scientific figures directly from the strict V2 analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
result = json.loads((ROOT / "artifacts/v2/analysis/results.json").read_text())
out = ROOT / "artifacts/v2/figures"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})
colors = {"native-fused": "#595e68", "packed-fused": "#006c67"}
fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), layout="constrained")
for ax, size, label in zip(axes, ("small", "medium"), ("3.35M · seven paired seeds", "19.28M · five paired seeds")):
    for arm in ("native-fused", "packed-fused"):
        points = [p for p in result["mean_learning_curves"] if (p["size"], p["arm"]) == (size, arm)]
        ax.plot([p["elapsed_seconds"] for p in points], [p["validation_bpb"] for p in points],
                color=colors[arm], label=points[0]["method"], linewidth=2)
    ax.axhline(3.2, color="#ac5c10", linestyle="--", linewidth=1, label="Frozen target: 3.2 BPB")
    ax.set(title=label, xlabel="Elapsed seconds incl. setup + probes", ylabel="Validation bits per byte (lower better)")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
fig.suptitle("Cotangent V2 · measured exact packing versus native fused AdamW")
for extension in ("png", "svg"):
    fig.savefig(out / f"learning-curves.{extension}")
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), layout="constrained")
for index, size in enumerate(("small", "medium")):
    c = result[size]
    timing, quality = c["elapsed_reduction_fraction"], c["quality_difference_bpb"]
    axes[0].scatter([100*p["elapsed_reduction_fraction"] for p in c["pairs"]],
                    [index]*len(c["pairs"]), marker="|", s=150, color="#8eaea8", alpha=.7)
    axes[0].errorbar(100*timing["mean"], index,
                    xerr=[[100*(timing["mean"]-timing["lower"])], [100*(timing["upper"]-timing["mean"])]],
                    fmt="o", color="#006c67", capsize=5)
    axes[1].scatter([p["test_difference_bpb"] for p in c["pairs"]],
                    [index]*len(c["pairs"]), marker="|", s=150, color="#8eaea8", alpha=.7)
    axes[1].errorbar(quality["mean"], index,
                    xerr=[[quality["mean"]-quality["lower"]], [quality["upper"]-quality["mean"]]],
                    fmt="o", color="#006c67", capsize=5)
for ax in axes:
    ax.set_yticks([0, 1], ["3.35M · n=7", "19.28M · n=5"])
    ax.set_ylim(-.6, 1.6)
    ax.axvline(0, color="#777", linewidth=.8)
    ax.grid(axis="x", alpha=.15)
axes[0].axvline(10, color="#ac5c10", linestyle="--", linewidth=1, label="Minimum mean: 10%")
axes[0].set(title="Elapsed time to the same 3.2-BPB target", xlabel="Paired time saved (%) · right is faster")
axes[1].axvline(.01, color="#ac5c10", linestyle="--", linewidth=1, label="Maximum upper bound: 0.01 BPB")
axes[1].set(title="Full reused-corpus test quality", xlabel="Packed − native BPB · left is better")
for ax in axes:
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("Dots: paired means · bars: two-sided 95% Student-t intervals · ticks: individual pairs")
for extension in ("png", "svg"):
    fig.savefig(out / f"paired-outcomes.{extension}")
plt.close(fig)
print(str(out.relative_to(ROOT)))
