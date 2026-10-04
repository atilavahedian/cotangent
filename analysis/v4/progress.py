"""Read-only campaign inventory; never evaluate a partial primary verdict."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
frozen=json.loads((ROOT/"artifacts/v4/frozen-study.json").read_text())
rows={}
for job in frozen["jobs"]:
    path=ROOT/"artifacts/v4/final"/job["size"]/f"{job['arm']}-{job['seed']}"/"summary.json"
    if path.exists():
        s=json.loads(path.read_text())
        if s["status"]=="completed":rows[job["size"],job["seed"],job["arm"]]=s
counts={"primary":0,"descriptive":0,"equivalence":0}
equivalence=[]
seen=set()
for job in frozen["jobs"]:
    pair=job["size"],job["seed"],job["comparison"]
    if pair in seen:continue
    seen.add(pair)
    arms=("incidence-fused","incidence-bucket") if job["comparison"]=="equivalence" else ("native-fused","native-bucket")
    members=[rows.get((job["size"],job["seed"],arm)) for arm in arms]
    if all(members):
        counts[job["comparison"]]+=1
        if job["comparison"]=="equivalence":
            a,b=members
            equivalence.append(dict(size=job["size"],seed=job["seed"],
                final_weights_identical=a["final_model_sha256"]==b["final_model_sha256"],
                test_scores_identical=a["final_test"]==b["final_test"]))
print(json.dumps(dict(completed=len(rows),planned=70,completed_pairs=counts,
    equivalence=equivalence,primary_verdict="pending all planned runs")))
