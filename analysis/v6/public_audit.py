"""Independent standard-library audit of public evidence; no GPU or weights."""
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_fingerprints():
    result = {}
    h = hashlib.sha256()
    paths = []
    for directory in ("cotangent", "scripts", "tests", "configs"):
        paths.extend(p for p in (ROOT/directory).rglob("*") if p.is_file() and p.suffix in (".py", ".json"))
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode()); h.update(p.read_bytes())
    result["v1"] = h.hexdigest()
    for version in range(2, 7):
        h = hashlib.sha256()
        for p in sorted((ROOT/f"research/v{version}").glob("*.py")):
            h.update(p.name.encode()); h.update(p.read_bytes())
        result[f"v{version}"] = h.hexdigest()
    result["data"] = digest(ROOT/"data/manifest.json")
    return result

def verify_interval(values, result, critical):
    n = len(values)
    mean = statistics.mean(values)
    half = critical*statistics.stdev(values)/math.sqrt(n)
    assert result["n"] == n
    for key, expected in (("mean", mean), ("lower", mean-half), ("upper", mean+half)):
        assert math.isclose(result[key], expected, rel_tol=1e-10, abs_tol=1e-12), (key, result[key], expected)

def main():
    fp = source_fingerprints()
    v5 = json.loads((ROOT/"artifacts/v5/frozen-study.json").read_text())
    for version in range(1, 5):
        assert fp[f"v{version}"] == v5[f"v{version}_source_sha256"]
    assert fp["v5"] == v5["source_sha256"] and fp["data"] == v5["data_manifest_sha256"]
    result5 = json.loads((ROOT/"artifacts/v5/analysis/results.json").read_text())
    for entry in result5["evidence"]:
        assert digest(ROOT/entry["summary"]) == entry["summary_sha256"]
    for key in ("elapsed_reduction_fraction", "training_reduction_fraction", "quality_difference_bpb"):
        field = "test_difference_bpb" if key == "quality_difference_bpb" else key
        verify_interval([p[field] for p in result5["small"]["pairs"]], result5["small"][key], v5["two_sided_student_t_critical"])
    protocol_path = ROOT/"artifacts/v6/frozen-study.json"
    protocol = json.loads(protocol_path.read_text())
    assert fp == protocol["fingerprints"]
    for name, key in (("pilots/index.json", "pilot_index_sha256"), ("verification-final.json", "verification_sha256"), ("verification-compiled.json", "compiled_verification_sha256")):
        assert digest(ROOT/"artifacts/v6"/name) == protocol[key]
    result = json.loads((ROOT/"artifacts/v6/analysis/results.json").read_text())
    assert result["protocol_sha256"] == digest(protocol_path)
    assert result["runs"] == len(protocol["jobs"]) == 128
    groups = {}
    for entry in result["evidence"]:
        path = ROOT/entry["summary"]
        assert digest(path) == entry["summary_sha256"]
        record = json.loads(path.read_text())
        assert record["status"] == "completed" and record["fingerprints"] == fp
        assert record["protocol_sha256"] == result["protocol_sha256"]
        assert len(record["step_seconds"]) == entry["steps"]
        assert math.isclose(sum(record["step_seconds"]), record["training_seconds"], rel_tol=1e-12)
        groups[entry["comparison"], entry["model"], entry["seed"], entry["arm"]] = record
    # Independent Student-t critical values at two-sided 99%, df=11 and df=7.
    critical = {12: 3.105806515539281, 8: 3.4994832973505026}
    for row in result["breadth"]:
        values, totals, quality = [], [], []
        for seed in range(3001, 3013):
            base = groups["breadth", row["model"], seed, row["baseline"]]
            candidate = groups["breadth", row["model"], seed, row["candidate"]]
            assert base["initialization_sha256"] == candidate["initialization_sha256"]
            assert base["schedule_sha256"] == candidate["schedule_sha256"]
            values.append(1-candidate["training_seconds"]/base["training_seconds"])
            totals.append(1-candidate["total_seconds"]/base["total_seconds"])
            quality.append(candidate["final_test"]["bpb"]-base["final_test"]["bpb"])
        verify_interval(values, row["training_reduction"], critical[12])
        verify_interval(totals, row["total_reduction"], critical[12])
        verify_interval(quality, row["test_difference"], critical[12])
    for key, stats in result["factorial"].items():
        verify_interval([r[key] for r in result["factorial_pairs"]], stats, critical[8])
    assert digest(ROOT/"paper/manuscript.tex") == json.loads((ROOT/"artifacts/v6/publication.json").read_text())["tex_sha256"]
    print(json.dumps(dict(status="passed", v5_runs=200, v6_runs=128, immutable_sources=True,
                          scope="Public source/record hashes and independent paired-statistic calculations. Does not verify local weight files or execute Metal training.")))

if __name__ == "__main__":
    main()
