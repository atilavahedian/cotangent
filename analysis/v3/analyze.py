"""Audit all 41 V3 runs and evaluate only the predeclared primary decision."""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
T95 = {3:4.302652729911275, 5:2.7764451051977987, 15:2.1447866879169273}
LABELS = {"native-fused":"Native fused AdamW", "incidence-fused":"Deterministic embedding control",
          "incidence-packed-flatnorm":"Cotangent deterministic + packed"}


def interval(values):
    mean = statistics.mean(values)
    half = T95[len(values)]*statistics.stdev(values)/math.sqrt(len(values))
    return dict(n=len(values),mean=mean,lower=mean-half,upper=mean+half)


def paired(rows, size, seeds, control, frozen, confirmatory=False):
    pairs = []
    for seed in seeds:
        arms = {r["arm"]:r for r in rows if r["size"]==size and r["seed"]==seed}
        native, candidate = arms[control], arms[frozen["candidate"]]
        reached = native["target_seconds"] is not None and candidate["target_seconds"] is not None
        pairs.append(dict(size=size,seed=seed,control=control,native_target_seconds=native["target_seconds"],
            packed_target_seconds=candidate["target_seconds"],native_target_step=native["target_step"],
            packed_target_step=candidate["target_step"],
            elapsed_reduction_fraction=1-candidate["target_seconds"]/native["target_seconds"] if reached else None,
            training_reduction_fraction=1-candidate["training_seconds"]/native["training_seconds"],
            complete_evaluation_reduction_fraction=1-candidate["evaluation_completed_seconds"]/native["evaluation_completed_seconds"],
            native_test_bpb=native["test_bpb"],packed_test_bpb=candidate["test_bpb"],
            test_difference_bpb=candidate["test_bpb"]-native["test_bpb"],both_reached=reached))
    quality = interval([p["test_difference_bpb"] for p in pairs])
    all_reached = all(p["both_reached"] for p in pairs)
    timing = interval([p["elapsed_reduction_fraction"] for p in pairs]) if all_reached else None
    time_pass = bool(timing and timing["mean"]>=frozen["meaningful_elapsed_reduction_fraction"] and timing["lower"]>0)
    quality_pass = quality["upper"]<=frozen["quality_noninferiority_margin_bpb"]
    return dict(pairs=pairs,all_reached=all_reached,confirmatory=confirmatory,
        elapsed_reduction_fraction=timing,quality_difference_bpb=quality,
        training_reduction_fraction=interval([p["training_reduction_fraction"] for p in pairs]),
        complete_evaluation_reduction_fraction=interval([p["complete_evaluation_reduction_fraction"] for p in pairs]),
        time_gate_pass=time_pass,quality_gate_pass=quality_pass,
        success=all_reached and time_pass and quality_pass)


def analyze(records_only=False):
    from cotangent.runtime import source_hash
    from research.v2.pilot import source_hash as v2_source_hash
    from research.v3.pilot import source_hash as v3_source_hash
    protocol = ROOT/"artifacts/v3/frozen-study.json"
    frozen = json.loads(protocol.read_text())
    protocol_hash = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert source_hash(ROOT)==frozen["v1_source_sha256"]
    assert v2_source_hash()==frozen["v2_source_sha256"]
    assert v3_source_hash()==frozen["source_sha256"]
    manifest = ROOT/"data/manifest.json"
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()==frozen["data_manifest_sha256"]
    data = json.loads(manifest.read_text())
    planned = {(j["size"],j["arm"],j["seed"]) for j in frozen["jobs"]}
    summaries = list((ROOT/"artifacts/v3/final").glob("*/*/summary.json"))
    observed = {(p.parent.parent.name,p.parent.name.rsplit("-",1)[0],int(p.parent.name.rsplit("-",1)[1])) for p in summaries}
    assert planned==observed, f"Missing/extra runs: {planned-observed}, {observed-planned}"
    rows, curves, evidence = [], [], []
    hashes = defaultdict(set)
    for job in frozen["jobs"]:
        size, arm, seed = job["size"],job["arm"],job["seed"]
        path = ROOT/"artifacts/v3/final"/size/f"{arm}-{seed}"/"summary.json"
        s = json.loads(path.read_text())
        assert s["status"]=="completed" and (s["size"],s["arm"],s["seed"])==(size,arm,seed)
        assert s["config"]==frozen["configs"][size]
        for key in ("source_sha256","v1_source_sha256","v2_source_sha256","data_manifest_sha256"):
            assert s[key]==frozen[key]
        assert s["protocol_sha256"]==protocol_hash
        assert s["deterministic_required"]==(arm!="native-fused") and not s["deterministic_fill_uninitialized_memory"]
        for split in ("validation","test"):
            assert s[f"final_{split}"]["tokens"]==data["splits"][split]["bytes"]-1
        assert s["tokens_budget"]==s["config"]["steps"]*2048
        assert len(s["step_seconds"])==s["config"]["steps"]
        assert math.isclose(sum(s["step_seconds"]),s["training_seconds"],rel_tol=1e-12)
        assert s["peak_swap_delta_bytes"]<=512*2**20
        raw = [json.loads(line) for line in (path.parent/"curve.jsonl").read_text().splitlines()]
        probes = [p for p in raw if "validation_bpb" in p]
        assert probes==s["validation_curve"]
        assert [p["step"] for p in probes]==list(range(100,s["config"]["steps"]+1,100))
        assert all(p["validation_tokens"]==65536 for p in probes)
        if not records_only:
            assert hashlib.sha256((ROOT/s["checkpoint"]).read_bytes()).hexdigest()==s["checkpoint_sha256"]
        hashes[size,seed].add((s["initialization_sha256"],s["schedule_sha256"],s["tokens_budget"],s["parameter_count"]))
        hit = next((p for p in probes if p["validation_bpb"]<=frozen["quality_target_bpb"]),None)
        rows.append(dict(size=size,arm=arm,method=LABELS[arm],seed=seed,parameters=s["parameter_count"],
            steps=s["config"]["steps"],tokens=s["tokens_budget"],training_seconds=s["training_seconds"],
            target_seconds=hit["elapsed_seconds"] if hit else None,target_step=hit["step"] if hit else None,
            test_bpb=s["final_test"]["bpb"],validation_bpb=s["final_validation"]["bpb"],
            evaluation_completed_seconds=s["evaluation_completed_seconds"],peak_driver_mib=s["peak_mps_driver_bytes"]/2**20))
        curves.extend(dict(size=size,arm=arm,method=LABELS[arm],seed=seed,**p) for p in probes)
        evidence.append(dict(**job,summary=str(path.relative_to(ROOT)),summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                             checkpoint=s["checkpoint"],checkpoint_sha256=s["checkpoint_sha256"]))
    assert len(rows)==41 and all(len(value)==1 for value in hashes.values())
    primary = paired(rows,"small",frozen["primary_seeds"],frozen["primary_control"],frozen,True)
    medium = paired(rows,"medium",frozen["medium_seeds"],frozen["primary_control"],frozen)
    ablation = paired(rows,"small",frozen["ablation_seeds"],"incidence-fused",frozen)
    aggregates, means = [], []
    for size in ("small","medium"):
        for arm in LABELS:
            group = [r for r in rows if r["size"]==size and r["arm"]==arm]
            if not group:
                continue
            aggregates.append(dict(size=size,arm=arm,method=LABELS[arm],seeds=len(group),
                test_bpb=statistics.mean(r["test_bpb"] for r in group),
                training_seconds=statistics.mean(r["training_seconds"] for r in group),
                target_seconds=statistics.mean(r["target_seconds"] for r in group) if all(r["target_seconds"] is not None for r in group) else None))
            for step in range(100,frozen["configs"][size]["steps"]+1,100):
                points = [p for p in curves if (p["size"],p["arm"],p["step"])==(size,arm,step)]
                means.append(dict(size=size,arm=arm,method=LABELS[arm],seeds=len(points),step=step,
                    validation_bpb=statistics.mean(p["validation_bpb"] for p in points),
                    elapsed_seconds=statistics.mean(p["elapsed_seconds"] for p in points)))
    result = dict(version=3,runs=len(rows),predicted_training_bytes=sum(r["tokens"] for r in rows),
        source_sha256=frozen["source_sha256"],protocol_sha256=protocol_hash,integrity_checks_passed=True,
        checkpoint_file_hashes_verified=not records_only,primary_success=primary["success"],
        small=primary,medium=medium,ablation=ablation,rows=rows,aggregates=aggregates,
        mean_learning_curves=means,learning_curves=curves,evidence=evidence,
        heldout_scope=frozen["heldout_scope"],limitations=frozen["limitations"])
    out = ROOT/("artifacts/v3/records-only-analysis" if records_only else "artifacts/v3/analysis")
    out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    for name,table in (("seed-results",rows),("method-means",aggregates),("learning-curves",curves),
                       ("paired-results",primary["pairs"]+medium["pairs"]),("paired-ablation",ablation["pairs"])):
        with (out/f"{name}.csv").open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(table[0]))
            writer.writeheader(); writer.writerows(table)
    print(json.dumps(dict(runs=41,predicted_training_bytes=result["predicted_training_bytes"],
        primary_success=result["primary_success"],
        comparisons={key:{k:result[key][k] for k in ("elapsed_reduction_fraction","training_reduction_fraction","quality_difference_bpb","success","confirmatory")} for key in ("small","medium","ablation")})))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--records-only",action="store_true")
    analyze(p.parse_args().records_only)
