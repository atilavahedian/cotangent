"""Restore every local V3 snapshot; check hashes and finite exact inference."""
import hashlib
import json
from pathlib import Path

import torch

from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer
from cotangent.runtime import model_hash

ROOT = Path(__file__).resolve().parents[2]
torch.set_num_threads(4)
frozen = json.loads((ROOT / "artifacts/v3/frozen-study.json").read_text())
checks = []
for job in frozen["jobs"]:
    summary_path = ROOT / "artifacts/v3/final" / job["size"] / f"{job['arm']}-{job['seed']}" / "summary.json"
    summary = json.loads(summary_path.read_text())
    checkpoint = ROOT / summary["checkpoint"]
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == summary["checkpoint_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Transformer(ModelConfig(**saved["model_config"]), Policy(mode="native"))
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    actual_hash = model_hash(model)
    assert actual_hash == summary["final_model_sha256"]
    prompt = torch.tensor(list(b"An exact derivative follows the chain rule."), dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(prompt)
    assert logits.shape == (1, len(prompt[0]), 256) and torch.isfinite(logits).all().item()
    checks.append(dict(**job, model_sha256=actual_hash, finite_exact_cpu_inference=True))
result = dict(status="passed", snapshots=len(checks), checks=checks,
              scope="CPU restoration and finite exact forward; no official held-out data evaluated again.")
(ROOT / "artifacts/v3/verification-snapshots.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(status="passed", snapshots=len(checks))))
