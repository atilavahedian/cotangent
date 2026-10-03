"""Second diagnostic phase after removing controller synchronization overhead."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    jobs = [("small", mode) for mode in ["native", "exact_custom", "uniform", "importance", "adaptive", "wht"]]
    jobs += [("medium", mode) for mode in ["native", "adaptive"]]
    outcomes = []
    for size, mode in jobs:
        cfg = json.loads((ROOT / "configs/pilot.json").read_text())
        cfg["precision"] = "bf16"
        cfg["policy"]["warmup"] = 8
        if size == "medium":
            cfg["model"].update(width=512, layers=6, heads=8)
        path = ROOT / "configs" / f"pilot-v3-{size}.json"
        path.write_text(json.dumps(cfg, indent=2) + "\n")
        out = ROOT / "artifacts/pilot" / f"{mode}-bf16-v3-{size}"
        if out.exists():
            continue
        print(f"Starting refined pilot {size} {mode}", flush=True)
        result = subprocess.run([sys.executable, "-m", "cotangent.train", "--config", str(path),
                                 "--mode", mode, "--seed", "7", "--phase", "pilot", "--output", str(out)], cwd=ROOT)
        outcomes.append({"size": size, "mode": mode, "exit_code": result.returncode})
        if result.returncode and not (out / "summary.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps({"status": "startup_failed", "phase": "pilot", "mode": mode}, indent=2))
    (ROOT / "artifacts/pilot/refined-launcher.json").write_text(json.dumps(outcomes, indent=2) + "\n")


if __name__ == "__main__":
    main()
