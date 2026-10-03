"""Strict analysis of V2's frozen paired experiment; never tune training here."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
T95 = {5: 2.7764451051977987, 7: 2.4469118511449692}


def interval(values):
    mean = statistics.mean(values)
    half = T95[len(values)] * statistics.stdev(values) / math.sqrt(len(values))
    return dict(n=len(values), mean=mean, lower=mean - half, upper=mean + half)


def crossing(curve, target):
    return next((point for point in curve if point["validation_bpb"] <= target), None)


def analyze(records_only=False):
    from cotangent.runtime import source_hash
    from research.v2.pilot import source_hash as v2_source_hash
    frozen_path = ROOT / "artifacts/v2/frozen-study.json"
    frozen = json.loads(frozen_path.read_text())
    protocol_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    assert source_hash(ROOT) == frozen["v1_source_sha256"]
    assert v2_source_hash() == frozen["source_sha256"]
    manifest_path = ROOT / "data/manifest.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == frozen["data_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    rows, curves, evidence = [], [], []
    paired_hashes = defaultdict(set)
    planned = {(job["size"], job["arm"], job["seed"]) for job in frozen["jobs"]}
    observed = {tuple(p.relative_to(ROOT / "artifacts/v2/final").parts[:1]) +
                (p.parent.name.rsplit("-", 1)[0], int(p.parent.name.rsplit("-", 1)[1]))
                for p in (ROOT / "artifacts/v2/final").glob("*/*/summary.json")}
    assert planned == observed, f"Incomplete or extra runs: missing={planned-observed}, extra={observed-planned}"
    for job in frozen["jobs"]:
        size, arm, seed = job["size"], job["arm"], job["seed"]
        path = ROOT / "artifacts/v2/final" / size / f"{arm}-{seed}" / "summary.json"
        result = json.loads(path.read_text())
        assert result["status"] == "completed", path
        assert (result["size"], result["arm"], result["seed"]) == (size, arm, seed)
        assert result["config"] == frozen["configs"][size]
        for key in ("source_sha256", "v1_source_sha256", "data_manifest_sha256"):
            assert result[key] == frozen[key], (path, key)
        assert result["protocol_sha256"] == protocol_hash
        assert result["final_test"]["tokens"] == manifest["splits"]["test"]["bytes"] - 1
        assert result["final_validation"]["tokens"] == manifest["splits"]["validation"]["bytes"] - 1
        assert result["tokens_budget"] == result["config"]["steps"] * 8 * 256
        assert len(result["step_seconds"]) == result["config"]["steps"]
        assert math.isclose(sum(result["step_seconds"]), result["training_seconds"], rel_tol=1e-12)
        assert result["peak_swap_delta_bytes"] <= 512 * 2**20
        raw = [json.loads(line) for line in (path.parent / "curve.jsonl").read_text().splitlines()]
        probes = [p for p in raw if "validation_bpb" in p]
        assert probes == result["validation_curve"]
        assert [p["step"] for p in probes] == list(range(100, result["config"]["steps"] + 1, 100))
        assert all(p["validation_tokens"] == 65536 for p in probes)
        checkpoint = ROOT / result["checkpoint"]
        if not records_only:
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == result["checkpoint_sha256"]
        paired_hashes[size, seed].add((result["initialization_sha256"], result["schedule_sha256"],
                                      result["tokens_budget"], result["parameter_count"]))
        hit = crossing(probes, frozen["quality_target_bpb"])
        rows.append(dict(size=size, arm=arm, seed=seed, parameters=result["parameter_count"],
            steps=result["config"]["steps"], tokens=result["tokens_budget"],
            training_seconds=result["training_seconds"], target_seconds=hit["elapsed_seconds"] if hit else None,
            target_step=hit["step"] if hit else None, test_bpb=result["final_test"]["bpb"],
            validation_bpb=result["final_validation"]["bpb"],
            evaluation_completed_seconds=result["evaluation_completed_seconds"],
            peak_driver_mib=result["peak_mps_driver_bytes"] / 2**20))
        curves.extend(dict(size=size, arm=arm, method="Native fused AdamW" if arm == "native-fused" else "Cotangent exact packing", seed=seed, **p) for p in probes)
        evidence.append(dict(size=size, arm=arm, seed=seed, summary=str(path.relative_to(ROOT)),
                             summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             checkpoint=result["checkpoint"], checkpoint_sha256=result["checkpoint_sha256"]))
    assert len(rows) == 24 and all(len(values) == 1 for values in paired_hashes.values())
    comparisons = {}
    for size in ("small", "medium"):
        pairs = []
        for seed in sorted({r["seed"] for r in rows if r["size"] == size}):
            arms = {r["arm"]: r for r in rows if r["size"] == size and r["seed"] == seed}
            native, packed = arms["native-fused"], arms["packed-fused"]
            reached = native["target_seconds"] is not None and packed["target_seconds"] is not None
            pairs.append(dict(size=size, seed=seed, native_target_seconds=native["target_seconds"],
                packed_target_seconds=packed["target_seconds"], native_target_step=native["target_step"],
                packed_target_step=packed["target_step"],
                elapsed_reduction_fraction=1 - packed["target_seconds"] / native["target_seconds"] if reached else None,
                training_reduction_fraction=1 - packed["training_seconds"] / native["training_seconds"],
                complete_evaluation_reduction_fraction=1 - packed["evaluation_completed_seconds"] / native["evaluation_completed_seconds"],
                native_test_bpb=native["test_bpb"], packed_test_bpb=packed["test_bpb"],
                test_difference_bpb=packed["test_bpb"] - native["test_bpb"], both_reached=reached))
        quality = interval([p["test_difference_bpb"] for p in pairs])
        all_reached = all(p["both_reached"] for p in pairs)
        timing = interval([p["elapsed_reduction_fraction"] for p in pairs]) if all_reached else None
        time_pass = bool(timing and timing["mean"] >= frozen["meaningful_elapsed_reduction_fraction"] and timing["lower"] > 0)
        quality_pass = quality["upper"] <= frozen["quality_noninferiority_margin_bpb"]
        comparisons[size] = dict(pairs=pairs, elapsed_reduction_fraction=timing,
            training_reduction_fraction=interval([p["training_reduction_fraction"] for p in pairs]),
            complete_evaluation_reduction_fraction=interval([p["complete_evaluation_reduction_fraction"] for p in pairs]),
            quality_difference_bpb=quality, all_reached=all_reached, time_gate_pass=time_pass,
            quality_gate_pass=quality_pass, success=time_pass and quality_pass and all_reached)
    aggregates = []
    for size in ("small", "medium"):
        for arm in frozen["arms"]:
            group = [r for r in rows if r["size"] == size and r["arm"] == arm]
            aggregates.append(dict(size=size, arm=arm, method="Native fused AdamW" if arm == "native-fused" else "Cotangent exact packing", seeds=len(group),
                test_bpb=statistics.mean(r["test_bpb"] for r in group),
                training_seconds=statistics.mean(r["training_seconds"] for r in group),
                target_seconds=statistics.mean(r["target_seconds"] for r in group) if all(r["target_seconds"] is not None for r in group) else None))
    means = []
    for size in ("small", "medium"):
        for arm in frozen["arms"]:
            for step in range(100, frozen["configs"][size]["steps"] + 1, 100):
                points = [p for p in curves if (p["size"], p["arm"], p["step"]) == (size, arm, step)]
                means.append(dict(size=size, arm=arm, method=points[0]["method"], step=step,
                    validation_bpb=statistics.mean(p["validation_bpb"] for p in points),
                    elapsed_seconds=statistics.mean(p["elapsed_seconds"] for p in points)))
    result = dict(version=2, runs=len(rows), predicted_training_bytes=sum(r["tokens"] for r in rows),
        source_sha256=frozen["source_sha256"], protocol_sha256=protocol_hash,
        integrity_checks_passed=True, checkpoint_file_hashes_verified=not records_only,
        small=comparisons["small"], medium=comparisons["medium"],
        primary_success=comparisons["small"]["success"], replication_success=comparisons["medium"]["success"],
        rows=rows, aggregates=aggregates, mean_learning_curves=means, learning_curves=curves,
        evidence=evidence, heldout_scope=frozen["heldout_scope"], limitations=frozen["limitations"])
    out = ROOT / ("artifacts/v2/records-only-analysis" if records_only else "artifacts/v2/analysis")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, table in (("seed-results", rows), ("method-means", aggregates),
                        ("paired-results", comparisons["small"]["pairs"] + comparisons["medium"]["pairs"]),
                        ("learning-curves", curves)):
        with (out / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    print(json.dumps({key: result[key] for key in ("runs", "predicted_training_bytes", "primary_success", "replication_success", "small", "medium")}))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--records-only", action="store_true")
    analyze(p.parse_args().records_only)
