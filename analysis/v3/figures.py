"""Publication figures from the complete frozen V3 analysis, without refitting."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
result = json.loads((ROOT / "artifacts/v3/analysis/results.json").read_text())
assert result["runs"] == 41 and result["integrity_checks_passed"]
out = ROOT / "artifacts/v3/figures"
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
colors = {"native-fused": "#595e68", "incidence-packed-flatnorm": "#006c67"}

def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4), layout="constrained")
for ax, size, label in zip(axes, ("small", "medium"),
        ("3.35M · 15 paired seeds · primary", "19.28M · three pairs · descriptive")):
    for arm in colors:
        points = [p for p in result["mean_learning_curves"] if (p["size"], p["arm"]) == (size, arm)]
        ax.plot([p["elapsed_seconds"] for p in points], [p["validation_bpb"] for p in points],
                color=colors[arm], label=points[0]["method"], linewidth=2)
    ax.axhline(3.2, color="#ac5c10", linestyle="--", linewidth=1, label="Frozen target: 3.2 BPB")
    ax.set(title=label, xlabel="Elapsed seconds incl. setup + probes", ylabel="Validation BPB (lower is better)")
    ax.grid(alpha=.15)
    ax.legend(fontsize=8)
fig.suptitle("Cotangent V3 · exact deterministic embedding gradients + packed AdamW")
save(fig, "learning-curves")

fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), layout="constrained")
groups = ("small", "medium", "ablation")
labels = ["3.35M · n=15 · primary", "19.28M · n=3 · descriptive",
          "3.35M · n=5 · deterministic control"]
for index, group in enumerate(groups):
    c = result[group]
    for ax, key, pairkey, scale in (
        (axes[0], "elapsed_reduction_fraction", "elapsed_reduction_fraction", 100),
        (axes[1], "quality_difference_bpb", "test_difference_bpb", 1)):
        value = c[key]
        if value is None:
            ax.text(0, index, "Target censored", va="center")
            continue
        ax.scatter([scale*p[pairkey] for p in c["pairs"]], [index]*len(c["pairs"]),
                   marker="|", s=150, color="#8eaea8", alpha=.8)
        ax.errorbar(scale*value["mean"], index,
            xerr=[[scale*(value["mean"]-value["lower"])], [scale*(value["upper"]-value["mean"])]],
            fmt="o", color="#006c67", capsize=5)
for ax in axes:
    ax.set_yticks(range(3), labels)
    ax.set_ylim(-.6, 2.6)
    ax.axvline(0, color="#777", linewidth=.8)
    ax.grid(axis="x", alpha=.15)
axes[0].axvline(10, color="#ac5c10", linestyle="--", linewidth=1, label="Minimum primary mean: 10%")
axes[0].set(title="Elapsed time to the same 3.2-BPB target", xlabel="Paired time saved (%) · right is faster")
axes[1].axvline(.01, color="#ac5c10", linestyle="--", linewidth=1, label="Maximum primary upper bound: 0.01 BPB")
axes[1].set(title="Full reused-corpus test quality", xlabel="Candidate − control BPB · left is better")
for ax in axes:
    ax.legend(loc="upper left", fontsize=8)
fig.suptitle("Means and two-sided 95% paired Student-t intervals; ticks show every pair")
save(fig, "paired-outcomes")
print(str(out.relative_to(ROOT)))
