"""Recompute frozen study conclusions from immutable per-run evidence.

No training, model selection, interpolation of crossings, or missing-value filling.
Run from the repository root: python analysis/analyze.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
# Two-sided 95% Student-t critical values, df=2 and df=4.
T95 = {3: 4.302652729911275, 5: 2.7764451051977987}
LABELS = {"native": "Native exact", "exact_custom": "Custom exact",
          "uniform": "Uniform 25%", "importance": "Importance 25%",
          "wht": "Hadamard 25%", "adaptive": "Cotangent adaptive"}


def interval(values):
    n = len(values)
    if n not in T95:
        raise ValueError(f"Unplanned sample size: {n}")
    mean = statistics.mean(values)
    half = T95[n] * statistics.stdev(values) / math.sqrt(n)
    return {"n": n, "mean": mean, "lower": mean - half, "upper": mean + half}


def crossing(curve, target):
    return next((p for p in curve if p["validation_bpb"] <= target), None)


def paired(rows, group, frozen):
    seeds = sorted({r["seed"] for r in rows if r["group"] == group})
    pairs = []
    for seed in seeds:
        arms = {r["mode"]: r for r in rows if r["group"] == group and r["seed"] == seed}
        base, candidate = arms["native"], arms["adaptive"]
        reached = base["target_seconds"] is not None and candidate["target_seconds"] is not None
        pairs.append({"group": group, "seed": seed,
                      "native_target_seconds": base["target_seconds"],
                      "adaptive_target_seconds": candidate["target_seconds"],
                      "elapsed_reduction_fraction": 1 - candidate["target_seconds"] / base["target_seconds"] if reached else None,
                      "native_test_bpb": base["test_bpb"], "adaptive_test_bpb": candidate["test_bpb"],
                      "test_difference_bpb": candidate["test_bpb"] - base["test_bpb"],
                      "training_reduction_fraction": 1 - candidate["training_seconds"] / base["training_seconds"],
                      "both_reached": reached})
    quality = interval([p["test_difference_bpb"] for p in pairs])
    all_reached = all(p["both_reached"] for p in pairs)
    time = interval([p["elapsed_reduction_fraction"] for p in pairs]) if all_reached else None
    quality_pass = quality["upper"] <= frozen["quality_noninferiority_margin_bpb"]
    time_pass = bool(time and time["mean"] >= frozen["meaningful_time_reduction_fraction"] and time["lower"] > 0)
    return {"pairs": pairs, "all_reached": all_reached,
            "quality_difference_bpb": quality, "elapsed_reduction_fraction": time,
            "training_reduction_fraction": interval([p["training_reduction_fraction"] for p in pairs]),
            "quality_gate_pass": quality_pass, "time_gate_pass": time_pass,
            "success": all_reached and quality_pass and time_pass,
            "confirmatory": group == "small"}


def main(records_only=False, output_dir=None):
    from cotangent.runtime import source_hash
    frozen = json.loads((ROOT / "artifacts/frozen-study.json").read_text())
    assert source_hash(ROOT) == frozen["source_sha256"], "Frozen source has changed"
    assert hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest() == frozen["data_manifest_sha256"]
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    rows, evidence, hashes, learning, sampling = [], [], defaultdict(set), [], []
    expected = {(j["group"], j["mode"], j["seed"]) for j in frozen["jobs"]}
    found = set()
    for job in frozen["jobs"]:
        group, mode, seed = job["group"], job["mode"], job["seed"]
        path = ROOT / "artifacts/final" / group / f"{mode}-{seed}" / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Study incomplete; missing {path.relative_to(ROOT)}")
        s = json.loads(path.read_text())
        assert s["status"] == "completed", path
        assert s["mode"] == mode and s["seed"] == seed and s["phase"] == "final"
        config = json.loads((ROOT / job["config"]).read_text())
        assert s["config"] == config and s["steps"] == config["steps"]
        assert s["source_sha256"] == frozen["source_sha256"]
        assert s["data_manifest_sha256"] == frozen["data_manifest_sha256"]
        assert s["final_test"]["tokens"] == manifest["splits"]["test"]["bytes"] - 1
        assert s["final_validation"]["tokens"] == manifest["splits"]["validation"]["bytes"] - 1
        assert s["tokens_budget"] == s["steps"] * config["batch_size"] * config["model"]["sequence"]
        assert s["peak_swap_delta_bytes"] <= 512 * 2**20
        checkpoint = ROOT / s["checkpoint"]
        if not records_only:
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == s["checkpoint_sha256"]
        hashes[group, seed].add((s["initialization_sha256"], s["schedule_sha256"], s["tokens_budget"], s["parameter_count"]))
        found.add((group, mode, seed))
        hit = crossing(s["validation_curve"], frozen["quality_target_bpb"])
        curve = [json.loads(line) for line in (path.parent / "curve.jsonl").read_text().splitlines()]
        probes = [r for r in curve if "validation_bpb" in r]
        assert probes == s["validation_curve"]
        assert [r["step"] for r in probes] == list(range(config["eval_every"], s["steps"] + 1, config["eval_every"]))
        train_logs = [r for r in curve if "loss_bpb" in r]
        observed_fraction = statistics.mean(r["sample_fraction"] for r in train_logs)
        row = {"group": group, "mode": mode, "method": LABELS[mode], "seed": seed,
               "parameters": s["parameter_count"], "steps": s["steps"], "tokens": s["tokens_budget"],
               "training_seconds": s["training_seconds"], "total_elapsed_seconds": s["total_elapsed_seconds"],
               "target_seconds": hit["elapsed_seconds"] if hit else None,
               "target_step": hit["step"] if hit else None, "target_reached": bool(hit),
               "censor_seconds": probes[-1]["elapsed_seconds"],
               "test_bpb": s["final_test"]["bpb"], "validation_bpb": s["final_validation"]["bpb"],
               "logged_sample_fraction": observed_fraction,
               "peak_rss_mib": s["peak_rss_bytes"] / 2**20,
               "peak_driver_mib": s["peak_mps_driver_bytes"] / 2**20,
               "peak_swap_mib": s["peak_swap_delta_bytes"] / 2**20}
        rows.append(row)
        evidence.append({"group": group, "mode": mode, "seed": seed,
                         "summary": str(path.relative_to(ROOT)),
                         "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "checkpoint": s["checkpoint"], "checkpoint_sha256": s["checkpoint_sha256"],
                         "initialization_sha256": s["initialization_sha256"], "schedule_sha256": s["schedule_sha256"]})
        for p in probes:
            learning.append({"group": group, "mode": mode, "method": LABELS[mode], "seed": seed,
                             "step": p["step"], "elapsed_seconds": p["elapsed_seconds"],
                             "validation_bpb": p["validation_bpb"], "target_bpb": frozen["quality_target_bpb"]})
        if mode == "adaptive":
            for p in train_logs:
                sampling.append({"group": group, "seed": seed, "step": p["step"],
                                 "sample_fraction": p["sample_fraction"], "audit_layers": p["audit_layers"],
                                 "gradient_error": p["gradient_error"]})
    assert found == expected and len(rows) == 36
    assert all(len(h) == 1 for h in hashes.values()), "Paired arms do not match"
    aggregates = []
    for group in ("small", "medium"):
        for mode in LABELS:
            arms = [r for r in rows if r["group"] == group and r["mode"] == mode]
            if not arms:
                continue
            test_ci = interval([r["test_bpb"] for r in arms])
            reached = sum(r["target_reached"] for r in arms)
            aggregates.append({"group": group, "mode": mode, "method": LABELS[mode], "n": len(arms),
                               "test_bpb": test_ci["mean"], "test_ci_lower": test_ci["lower"], "test_ci_upper": test_ci["upper"],
                               "training_seconds": statistics.mean(r["training_seconds"] for r in arms),
                               "target_seconds": statistics.mean(r["target_seconds"] for r in arms) if reached == len(arms) else None,
                               "target_reached": reached, "target_status": f"{reached}/{len(arms)} reached",
                               "logged_sample_fraction": statistics.mean(r["logged_sample_fraction"] for r in arms)})
    curve_groups = defaultdict(list)
    for p in learning:
        curve_groups[p["group"], p["mode"], p["step"]].append(p)
    mean_curves = [{"group": g, "mode": m, "method": LABELS[m], "step": step,
                    "validation_bpb": statistics.mean(p["validation_bpb"] for p in points),
                    "elapsed_seconds": statistics.mean(p["elapsed_seconds"] for p in points),
                    "target_bpb": frozen["quality_target_bpb"], "n": len(points)}
                   for (g, m, step), points in curve_groups.items()]
    result = {"status": "complete", "runs": len(rows), "predicted_training_bytes": sum(r["tokens"] for r in rows),
              "frozen_source_sha256": frozen["source_sha256"], "integrity_checks_passed": True,
              "checkpoint_file_hashes_verified": not records_only,
              "primary": paired(rows, "small", frozen), "medium": paired(rows, "medium", frozen),
              "aggregates": aggregates, "seed_results": rows, "mean_learning_curves": mean_curves,
              "learning_curves": learning, "sampling_observations": sampling, "evidence": evidence,
              "limitations": ["One corpus and Apple M5 Pro; no CUDA or large-model evidence.",
                              "Time-to-quality uses scheduled probes of the first 65,536 validation bytes, not the full split.",
                              "Serial local runs share an uncontrolled desktop environment; timing variation remains.",
                              "Student-t intervals assume independent approximately normal seed differences; n=5 primary and n=3 secondary are small.",
                              "Logged sample fractions are unweighted means over layers at recorded steps, not full-run FLOP reductions.",
                              "Hadamard control is an inspired baseline, not an official LBP-WHT or INSTANT reproduction.",
                              "Final test is used only for analysis; no method was changed after the freeze."]}
    out = Path(output_dir) if output_dir else ROOT / ("artifacts/records-only-analysis" if records_only else "artifacts/analysis")
    out.mkdir(parents=True,exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, table in (("seed-results", rows), ("method-means", aggregates),
                        ("paired-primary", result["primary"]["pairs"]), ("learning-curves", learning)):
        with (out / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps({k: result[k] for k in ("runs", "predicted_training_bytes", "primary", "medium")}, indent=2))


if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, str(ROOT))
    parser=argparse.ArgumentParser()
    parser.add_argument("--records-only",action="store_true",help="Recompute statistics from public records without claiming local checkpoint-file verification")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    main(args.records_only,args.output_dir)
