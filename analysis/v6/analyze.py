"""Audit and summarize the complete frozen extension, including adverse pairs."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
import mpmath as mp
from research.v6.runtime import ROOT, check_prior, digest

def interval(values, confidence=.99):
    values = list(values)
    assert len(values) > 1 and all(math.isfinite(v) for v in values)
    n, mean = len(values), statistics.mean(values)
    df = n - 1
    tail = 1 - confidence
    root = mp.findroot(lambda t: mp.betainc(df / 2, .5, 0, df / (df + t * t), regularized=True) - tail, (2, 4))
    critical = float(root)
    half = critical * statistics.stdev(values) / math.sqrt(n)
    return dict(n=n, mean=mean, lower=mean-half, upper=mean+half, confidence=confidence, critical=critical)

def analyze(records_only=False):
    path = ROOT / "artifacts/v6/frozen-study.json"
    frozen = json.loads(path.read_text())
    assert check_prior() == frozen["fingerprints"]
    assert digest(ROOT / "artifacts/v6/pilots/index.json") == frozen["pilot_index_sha256"]
    assert digest(ROOT / "artifacts/v6/verification-final.json") == frozen["verification_sha256"]
    assert digest(ROOT / "artifacts/v6/verification-compiled.json") == frozen["compiled_verification_sha256"]
    protocol_hash = digest(path)
    planned = {f"{j['comparison']}--{j['model']}--{j['arm']}--{j['seed']}" for j in frozen["jobs"]}
    observed = {p.parent.name for p in (ROOT / "artifacts/v6/final").glob("*/summary.json")}
    assert planned == observed, (planned-observed, observed-planned)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    rows, evidence, curves = [], [], []
    grouped = defaultdict(dict)
    for job in frozen["jobs"]:
        name = f"{job['comparison']}--{job['model']}--{job['arm']}--{job['seed']}"
        summary_path = ROOT / "artifacts/v6/final" / name / "summary.json"
        s = json.loads(summary_path.read_text())
        assert s["status"] == "completed", name
        assert s["fingerprints"] == frozen["fingerprints"] and s["protocol_sha256"] == protocol_hash
        assert s["official"] and s["config"] == frozen["configs"][job["model"]]
        assert all(s[k] == job[k] for k in ("model", "arm", "seed", "steps"))
        assert s["torch"] == "2.14.1" and s["torch_revision"] == "5c4886908584029761b579af026dcfb627c84070"
        metadata = json.loads((summary_path.parent / "metadata.json").read_text())
        assert all(s[k] == v for k, v in metadata.items() if k != "status")
        assert len(s["step_seconds"]) == job["steps"] and all(math.isfinite(t) and t > 0 for t in s["step_seconds"])
        assert math.isclose(sum(s["step_seconds"]), s["training_seconds"], rel_tol=1e-12)
        assert s["peak_rss_bytes"] <= 5*2**30 and s["peak_mps_driver_bytes"] <= 8*2**30 and s["peak_swap_delta_bytes"] <= 512*2**20
        for split in ("validation", "test"):
            assert s["final_" + split]["tokens"] == manifest["splits"][split]["bytes"] - 1
            assert math.isfinite(s["final_" + split]["bpb"])
        raw = [json.loads(line) for line in (summary_path.parent / "curve.jsonl").read_text().splitlines()]
        assert [r["step"] for r in raw if "loss_bpb" in r] == list(range(1, job["steps"], 10))
        assert [r for r in raw if "diagnostic_bpb" in r] == s["probes"]
        assert [r["step"] for r in s["probes"]] == list(range(200, job["steps"] + 1, 200))
        if not records_only:
            assert digest(ROOT / s["checkpoint"]) == s["checkpoint_sha256"]
        row = dict(**job, parameters=s["parameters"], parameter_tensors=s["parameter_tensors"], length_buckets=s["length_buckets"],
                   initialization_sha256=s["initialization_sha256"], schedule_sha256=s["schedule_sha256"],
                   training_seconds=s["training_seconds"], setup_seconds=s["setup_seconds"], total_seconds=s["total_seconds"],
                   test_bpb=s["final_test"]["bpb"], validation_bpb=s["final_validation"]["bpb"], final_model_sha256=s["final_model_sha256"],
                   peak_driver_mib=s["peak_mps_driver_bytes"]/2**20, peak_rss_mib=s["peak_rss_bytes"]/2**20)
        rows.append(row)
        grouped[job["comparison"], job["model"], job["seed"]][job["arm"]] = row
        curves.extend(dict(**job, **point) for point in s["probes"])
        evidence.append(dict(**job, summary=str(summary_path.relative_to(ROOT)), summary_sha256=digest(summary_path),
                             checkpoint=s["checkpoint"], checkpoint_sha256=s["checkpoint_sha256"]))
    assert len(rows) == 128
    pairs, breadth = [], []
    for name in frozen["configs"]:
        model_pairs = []
        for seed in range(3001, 3013):
            g = grouped["breadth", name, seed]
            native, candidate = g[frozen["baselines"][name]], g[frozen["candidates"][name]]
            assert all(native[k] == candidate[k] for k in ("initialization_sha256", "schedule_sha256", "steps", "parameters"))
            pair = dict(model=name, seed=seed, baseline=native["arm"], candidate=candidate["arm"],
                        training_reduction=1-candidate["training_seconds"]/native["training_seconds"],
                        total_reduction=1-candidate["total_seconds"]/native["total_seconds"],
                        test_difference=candidate["test_bpb"]-native["test_bpb"], validation_difference=candidate["validation_bpb"]-native["validation_bpb"],
                        native_seconds=native["training_seconds"], candidate_seconds=candidate["training_seconds"],
                        native_test_bpb=native["test_bpb"], candidate_test_bpb=candidate["test_bpb"],
                        identical_final_hash=native["final_model_sha256"] == candidate["final_model_sha256"])
            pairs.append(pair); model_pairs.append(pair)
        natives = [r for r in rows if r["comparison"] == "breadth" and r["model"] == name and r["arm"] == frozen["baselines"][name]]
        candidates = [r for r in rows if r["comparison"] == "breadth" and r["model"] == name and r["arm"] == frozen["candidates"][name]]
        breadth.append(dict(model=name, parameters=natives[0]["parameters"], parameter_tensors=natives[0]["parameter_tensors"], buckets=natives[0]["length_buckets"],
                            baseline=frozen["baselines"][name], candidate=frozen["candidates"][name],
                            training_reduction=interval(p["training_reduction"] for p in model_pairs),
                            total_reduction=interval(p["total_reduction"] for p in model_pairs),
                            test_difference=interval(p["test_difference"] for p in model_pairs),
                            validation_difference=interval(p["validation_difference"] for p in model_pairs),
                            native_training_seconds=statistics.mean(r["training_seconds"] for r in natives),
                            candidate_training_seconds=statistics.mean(r["training_seconds"] for r in candidates),
                            native_test_bpb=statistics.mean(r["test_bpb"] for r in natives), candidate_test_bpb=statistics.mean(r["test_bpb"] for r in candidates),
                            native_setup_seconds=statistics.mean(r["setup_seconds"] for r in natives), candidate_setup_seconds=statistics.mean(r["setup_seconds"] for r in candidates),
                            native_peak_driver_mib=max(r["peak_driver_mib"] for r in natives), candidate_peak_driver_mib=max(r["peak_driver_mib"] for r in candidates),
                            slower_pairs=sum(p["training_reduction"] < 0 for p in model_pairs), worse_test_pairs=sum(p["test_difference"] > 0 for p in model_pairs)))
    factorial_pairs = []
    for seed in range(2001, 2009):
        g = grouped["factorial", "transformer-small", seed]
        assert len(g) == 4
        assert len({(r["initialization_sha256"], r["schedule_sha256"], r["steps"], r["parameters"]) for r in g.values()}) == 1
        a, b, c, d = [g[k]["training_seconds"] for k in ("scalar-native", "bucket-native", "scalar-packed", "bucket-packed")]
        # Factor effects normalized to the native-scalar cell in each block.
        factorial_pairs.append(dict(seed=seed, norm_only=1-b/a, packing_only=1-c/a, combined=1-d/a,
                                    norm_main_effect=(a+c-b-d)/(2*a), packing_main_effect=(a+b-c-d)/(2*a),
                                    interaction=(a-b-c+d)/a))
    factorial = {key: interval(p[key] for p in factorial_pairs) for key in factorial_pairs[0] if key != "seed"}
    result = dict(version=6, runs=len(rows), fingerprints=frozen["fingerprints"], protocol_sha256=protocol_hash,
                  integrity_checks_passed=True, checkpoint_file_hashes_verified=not records_only,
                  descriptive=True, confidence=.99, breadth=breadth, pairs=pairs, factorial=factorial,
                  factorial_pairs=factorial_pairs, rows=rows, curves=curves, evidence=evidence,
                  inference_scope="Per-configuration descriptive paired intervals; no simultaneous coverage or new powered quality gate. No universal backend or novel-algorithm claim.")
    directory = ROOT / ("artifacts/v6/records-only-analysis" if records_only else "artifacts/v6/analysis")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, records in (("runs", rows), ("pairs", pairs), ("factorial-pairs", factorial_pairs)):
        with (directory / (name + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    print(json.dumps(dict(runs=len(rows), breadth=breadth, factorial=factorial)))
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--records-only", action="store_true")
    analyze(p.parse_args().records_only)
