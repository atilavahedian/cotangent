"""Serial execution of the predeclared jobs; logs and failures remain inspectable."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
frozen = json.loads((ROOT / "artifacts/v5/frozen-study.json").read_text())
for index, job in enumerate(frozen["jobs"]):
    output = ROOT / "artifacts/v5/final" / job["size"] / f"{job['arm']}-{job['seed']}"
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite or silently skip {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Job {index + 1}/{len(frozen['jobs'])}: {job}", flush=True)
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run([sys.executable, "-m", "research.v5.study", "--arm", job["arm"],
                                 "--size", job["size"], "--seed", str(job["seed"])],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Job failed ({result.returncode}); inspect {output.with_suffix('.log')}")
    summary = json.loads((output / "summary.json").read_text())
    print(json.dumps({k: summary[k] for k in ("arm", "size", "seed", "training_seconds", "final_test")}), flush=True)
