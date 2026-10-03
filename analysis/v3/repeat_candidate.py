"""Post-study diagnostic: repeat the identical candidate larger-model training.

This is not an additional primary pair and must never enter the frozen gate.
The unmodified frozen runner executes in an isolated mirror, avoiding result
overwrites. Data links point only to this Cotangent project's pinned local data.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT.parents[1] / "work/v3-candidate-repeat"
assert not WORK.exists(), "Refusing to overwrite a diagnostic"
WORK.mkdir(parents=True)
for folder in ("cotangent", "scripts", "tests", "configs", "research/v2", "research/v3"):
    shutil.copytree(ROOT / folder, WORK / folder, ignore=shutil.ignore_patterns("__pycache__"))
(WORK / "artifacts/v3").mkdir(parents=True)
shutil.copy2(ROOT / "artifacts/v3/frozen-study.json", WORK / "artifacts/v3/frozen-study.json")
(WORK / "data").mkdir()
shutil.copy2(ROOT / "data/manifest.json", WORK / "data/manifest.json")
for name in ("train.bin", "validation.bin", "test.bin"):
    (WORK / "data" / name).symlink_to(ROOT / "data" / name)
out = ROOT / "artifacts/v3/diagnostics"
out.mkdir(parents=True, exist_ok=True)
with (out / "candidate-repeat-409.log").open("w") as log:
    subprocess.run([sys.executable, "-m", "research.v3.study", "--arm", "incidence-packed-flatnorm",
                    "--size", "medium", "--seed", "409"], cwd=WORK,
                   stdout=log, stderr=subprocess.STDOUT, check=True)
repeat_dir = WORK / "artifacts/v3/final/medium/incidence-packed-flatnorm-409"
original = json.loads((ROOT / "artifacts/v3/final/medium/incidence-packed-flatnorm-409/summary.json").read_text())
repeat = json.loads((repeat_dir / "summary.json").read_text())
for key in ("source_sha256", "v1_source_sha256", "v2_source_sha256", "protocol_sha256", "data_manifest_sha256",
            "initialization_sha256", "schedule_sha256", "config", "tokens_budget"):
    assert original[key] == repeat[key], key
checkpoint = ROOT / "artifacts/checkpoints/v3/diagnostics/candidate-repeat-409.pt"
checkpoint.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(WORK / repeat["checkpoint"], checkpoint)
repeat["checkpoint"] = str(checkpoint.relative_to(ROOT))
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == repeat["checkpoint_sha256"]
shutil.copytree(repeat_dir, out / "candidate-repeat-409")
(out / "candidate-repeat-409/summary.json").write_text(json.dumps(repeat, indent=2) + "\n")
comparison = dict(scope="Single post-study candidate-repeat diagnostic, excluded from all frozen gates; same seed/source/data/schedule/config, a new MPS execution.",
                  original_test_bpb=original["final_test"]["bpb"], repeat_test_bpb=repeat["final_test"]["bpb"],
                  repeat_minus_original_bpb=repeat["final_test"]["bpb"]-original["final_test"]["bpb"],
                  identical_final_model_hash=original["final_model_sha256"] == repeat["final_model_sha256"],
                  matched_inputs_verified=True)
(out / "candidate-repeat-comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps(comparison))
