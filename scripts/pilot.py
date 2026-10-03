"""Sequential, training-only diagnostic sweep; preserve unsuccessful runs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    jobs = [("native", "fp32"), ("exact_custom", "fp32"), ("uniform", "fp32"),
            ("importance", "fp32"), ("adaptive", "fp32"), ("wht", "fp32"),
            ("native", "bf16"), ("exact_custom", "bf16"), ("adaptive", "bf16")]
    failures = []
    for mode, precision in jobs:
        cfg = json.loads((ROOT / "configs/pilot.json").read_text())
        cfg["precision"] = precision
        path = ROOT / "configs" / f"pilot-{precision}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        out = ROOT / "artifacts/pilot" / f"{mode}-{precision}-v2"
        if out.exists():
            print(f"Preserving existing pilot: {out.name}", flush=True)
            continue
        print(f"Starting pilot {mode} {precision}", flush=True)
        result = subprocess.run([sys.executable, "-m", "cotangent.train", "--config", str(path),
                                 "--mode", mode, "--seed", "7", "--phase", "pilot", "--output", str(out)], cwd=ROOT)
        if result.returncode:
            failures.append({"mode": mode, "precision": precision, "exit_code": result.returncode})
            if not (out / "summary.json").exists():
                out.mkdir(exist_ok=True, parents=True)
                (out / "summary.json").write_text(json.dumps({"status": "startup_failed", "mode": mode,
                                                            "precision": precision, "phase": "pilot",
                                                            "exit_code": result.returncode}, indent=2) + "\n")
    (ROOT / "artifacts/pilot/launcher.json").write_text(json.dumps({"jobs": jobs, "failures": failures}, indent=2) + "\n")


if __name__ == "__main__":
    main()
