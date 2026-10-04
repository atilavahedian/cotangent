"""Run the frozen extension serially and retain every planned outcome."""
import json
import os
import subprocess
import sys
from research.v6.runtime import ROOT, check_prior

def main():
    frozen = json.loads((ROOT / "artifacts/v6/frozen-study.json").read_text())
    assert check_prior() == frozen["fingerprints"]
    for i, job in enumerate(frozen["jobs"]):
        output = ROOT / "artifacts/v6/final" / f"{job['comparison']}--{job['model']}--{job['arm']}--{job['seed']}"
        if (output / "summary.json").exists():
            existing = json.loads((output / "summary.json").read_text())
            assert existing["fingerprints"] == frozen["fingerprints"]
            # Completed or failed observations are immutable; never rerun them.
            continue
        if output.exists():
            raise RuntimeError(f"Incomplete observation requires inspection: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        log = output.parent / (output.name + ".log")
        command = [sys.executable, "-m", "research.v6.run", "--official", "--model", job["model"], "--arm", job["arm"], "--seed", str(job["seed"]), "--steps", str(job["steps"]), "--output", str(output)]
        with log.open("w") as f:
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(ROOT.parents[1] / "work/inductor-v6"), TORCHINDUCTOR_COMPILE_THREADS="1")
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(json.dumps(dict(completed_index=i + 1, planned=len(frozen["jobs"]), returncode=proc.returncode, **job)), flush=True)
        if proc.returncode:
            raise RuntimeError(f"Frozen run failed; inspect retained log {log}")

if __name__ == "__main__":
    main()
