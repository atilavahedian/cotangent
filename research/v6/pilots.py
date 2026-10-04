"""Training-only baseline screening; compile failures and timeouts are retained."""
import json
import os
import subprocess
import sys
from research.v6.runtime import ROOT, check_prior
from research.v6.models import CONFIGS

def main():
    check_prior()
    root = ROOT / "artifacts/v6/pilots"
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for model in CONFIGS:
        # Compilation is tested explicitly, without pretending unsupported paths ran.
        for arm in ("scalar-native", "bucket-native", "scalar-packed", "bucket-packed", "foreach-native", "auto-native", "loop-native", "compile-native", "aot-native"):
            for seed in (1801, 1802):
                output = root / f"{model}--{arm}--{seed}"
                log = root / f"{output.name}.log"
                if output.exists():
                    raise FileExistsError(output)
                cmd = [sys.executable, "-m", "research.v6.run", "--model", model, "--arm", arm, "--seed", str(seed), "--steps", "120", "--output", str(output)]
                env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(ROOT.parents[1] / "work/inductor-v6"), TORCHINDUCTOR_COMPILE_THREADS="1")
                timeout = 180 if arm == "compile-native" else 120
                with log.open("w") as f:
                    try:
                        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
                        status = "completed" if proc.returncode == 0 else "failed"
                    except subprocess.TimeoutExpired:
                        status = "timeout"
                record = dict(model=model, arm=arm, seed=seed, status=status, log=str(log.relative_to(ROOT)))
                if (output / "summary.json").exists():
                    summary = json.loads((output / "summary.json").read_text())
                    record["summary"] = str((output / "summary.json").relative_to(ROOT))
                    if status == "completed":
                        record.update(warmed_training_seconds=summary["warmed_training_seconds"], training_seconds=summary["training_seconds"], diagnostic_bpb=summary["diagnostic_bpb"], setup_seconds=summary["setup_seconds"])
                    else:
                        record["error"] = summary.get("error")
                records.append(record)
                (root / "index.json").write_text(json.dumps(records, indent=2) + "\n")
                print(json.dumps(record), flush=True)
                # A failed first compile capability check is sufficient; retrying
                # the same unsupported graph with another seed adds no evidence.
                if status != "completed":
                    break
    print("Baseline screening complete", flush=True)

if __name__ == "__main__":
    main()
