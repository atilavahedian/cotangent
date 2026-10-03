"""Write a one-time immutable study definition before official held-out access."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from cotangent.runtime import source_hash

ROOT = Path(__file__).resolve().parents[1]


def main():
    dest = ROOT / "artifacts/frozen-study.json"
    if dest.exists():
        raise SystemExit("Study already frozen; refusing to overwrite")
    small = json.loads((ROOT / "configs/pilot-v3-small.json").read_text())
    small.update(steps=2000, lr_warmup=100, eval_every=100, eval_batches=32)
    medium = json.loads((ROOT / "configs/pilot-v3-medium.json").read_text())
    medium.update(steps=1500, lr_warmup=100, eval_every=100, eval_batches=32)
    for name, cfg in [("small", small), ("medium", medium)]:
        (ROOT / "configs" / f"final-{name}.json").write_text(json.dumps(cfg, indent=2) + "\n")
    jobs = []
    arms = ["native", "exact_custom", "uniform", "importance", "wht", "adaptive"]
    for seed in [11, 23, 37, 53, 71]:
        order = list(np.random.default_rng(seed + 300000).permutation(arms))
        jobs.extend({"group": "small", "mode": str(mode), "seed": seed,
                     "config": "configs/final-small.json"} for mode in order)
    for i, seed in enumerate([11, 23, 37]):
        order = ["native", "adaptive"] if i % 2 == 0 else ["adaptive", "native"]
        jobs.extend({"group": "medium", "mode": mode, "seed": seed,
                     "config": "configs/final-medium.json"} for mode in order)
    frozen = {"name": "Cotangent", "frozen_on": "2026-10-03", "phase": "before-official-validation-or-test",
              "source_sha256": source_hash(ROOT),
              "data_manifest_sha256": hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest(),
              "code_commit_before_freeze": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "primary_comparison": "adaptive versus native BF16 exact backpropagation, small model, five paired seeds",
              "quality_target_bpb": 3.2,
              "quality_target_definition": "First scheduled 65,536-byte official-validation probe at or below 3.2 bits per byte; no interpolation. Missing crossings remain censored.",
              "quality_noninferiority_margin_bpb": 0.03,
              "meaningful_time_reduction_fraction": 0.10,
              "confidence_interval": "Paired two-sided 95% Student-t interval across seeds, descriptive assumption of independent seed differences. Five seeds are limited evidence.",
              "primary_success_rule": "All paired arms reach target; upper paired 95% CI for final test BPB difference <= 0.03; mean paired elapsed-time reduction >= 10%; lower paired 95% CI for that reduction > 0.",
              "timing_endpoint": "Elapsed time from process-local setup start through first scheduled validation crossing, including kernel priming, sampling, audits, synchronization, guards, optimizer and earlier evaluations. Dependency installation and common data download excluded.",
              "secondary_scope": "Fixed controls and the larger-model comparison are descriptive robustness checks, not additional confirmatory discovery tests.",
              "jobs": jobs}
    dest.write_text(json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"jobs": len(jobs), "source_sha256": frozen["source_sha256"], "quality_target_bpb": 3.2}, indent=2))


if __name__ == "__main__":
    main()
