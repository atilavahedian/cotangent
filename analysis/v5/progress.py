"""Inventory only. No interim significance tests or partial win verdict."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
frozen=json.loads((ROOT/'artifacts/v5/frozen-study.json').read_text())
completed=set();failed=[];started=0
for job in frozen['jobs']:
    path=ROOT/'artifacts/v5/final'/job['size']/f"{job['arm']}-{job['seed']}"
    started+=path.exists()
    if (path/'summary.json').exists():
        result=json.loads((path/'summary.json').read_text())
        if result['status']=='completed':completed.add((job['seed'],job['arm']))
        elif result['status']=='failed':failed.append(job)
pairs=sum(all((seed,arm) in completed for arm in ('native-fused','native-bucket')) for seed in frozen['primary_seeds'])
print(json.dumps(dict(completed=len(completed),planned=200,completed_pairs=pairs,started=started,failed=failed,primary_verdict='pending all 200 runs')))
