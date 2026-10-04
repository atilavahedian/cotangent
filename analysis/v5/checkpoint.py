"""Stage immutable completed observations only; never stage an active partial run."""
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
paths=[]
for path in sorted((ROOT/'artifacts/v5/final').glob('*/*/summary.json')):
    if json.loads(path.read_text())['status']=='completed':
        paths.append(str(path.parent.relative_to(ROOT)))
        log=path.parent.with_suffix('.log')
        if log.exists():paths.append(str(log.relative_to(ROOT)))
assert paths
subprocess.run(['git','add','--',*paths],cwd=ROOT,check=True)
print(json.dumps(dict(staged_completed_runs=len(paths)//2)))
