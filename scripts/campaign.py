"""Execute the frozen study serially, checking source and dataset integrity."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from cotangent.runtime import source_hash

ROOT = Path(__file__).resolve().parents[1]


def check_integrity(frozen):
    actual = source_hash(ROOT)
    if actual != frozen["source_sha256"]:
        raise RuntimeError(f"Source freeze violated: {actual} != {frozen['source_sha256']}")
    if hashlib.sha256((ROOT / "data/manifest.json").read_bytes()).hexdigest() != frozen["data_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed")


def main():
    frozen = json.loads((ROOT / "artifacts/frozen-study.json").read_text())
    check_integrity(frozen)
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for split, info in manifest["splits"].items():
        if hashlib.sha256((ROOT / "data" / f"{split}.bin").read_bytes()).hexdigest() != info["sha256"]:
            raise RuntimeError(f"Dataset integrity violation: {split}")
    start = time.perf_counter()
    outcomes = []
    for job in frozen["jobs"]:
        check_integrity(frozen)
        group, mode, seed = job["group"], job["mode"], job["seed"]
        out = ROOT / "artifacts/final" / group / f"{mode}-{seed}"
        if out.exists():
            summary = json.loads((out / "summary.json").read_text())
            if summary["status"] != "completed" or summary["source_sha256"] != frozen["source_sha256"]:
                raise RuntimeError(f"Existing run is incomplete or incompatible: {out}")
            print(f"Preserving completed frozen run: {group}/{mode}/{seed}", flush=True)
            outcomes.append({**job, "status": "already_completed"})
            continue
        print(f"START {group} {mode} seed={seed}", flush=True)
        transcript = ROOT / "artifacts/final" / group / f"{mode}-{seed}.log"
        transcript.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "cotangent.train", "--config", str(ROOT / job["config"]),
                   "--mode", mode, "--seed", str(seed), "--phase", "final", "--output", str(out)]
        with transcript.open("w", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                log.write(line)
                try:
                    item = json.loads(line)
                    if item.get("step", 0) % 500 == 0:
                        print(line.strip(), flush=True)
                except json.JSONDecodeError:
                    print(line.strip(), flush=True)
            code = process.wait()
        check_integrity(frozen)
        outcomes.append({**job, "exit_code": code, "status": "completed" if code == 0 else "failed"})
        (ROOT / "artifacts/campaign-progress.json").write_text(json.dumps({"outcomes": outcomes,
                                                                         "launcher_elapsed_seconds": time.perf_counter() - start}, indent=2) + "\n")
        if code:
            raise RuntimeError(f"Frozen run failed and was preserved: {group}/{mode}/{seed}")
        print(f"DONE {group} {mode} seed={seed}", flush=True)
    (ROOT / "artifacts/campaign-complete.json").write_text(json.dumps({"jobs": outcomes,
                                                                     "launcher_elapsed_seconds": time.perf_counter() - start,
                                                                     "source_sha256": frozen["source_sha256"]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
