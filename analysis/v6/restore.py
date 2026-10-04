"""Restore every extension snapshot and verify hashes without repeating test data."""
import json
import torch
from cotangent.runtime import model_hash
from research.v6.models import make_model
from research.v6.runtime import ROOT, digest

def main():
    torch.set_num_threads(4)
    rows = json.loads((ROOT / "artifacts/v6/analysis/results.json").read_text())["evidence"]
    checks = []
    for row in rows:
        path = ROOT / row["checkpoint"]
        assert digest(path) == row["checkpoint_sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = make_model(checkpoint["model_name"], checkpoint["seed"], "cpu")
        model.load_state_dict(checkpoint["model"])
        assert model_hash(model) == checkpoint["final_model_sha256"]
        with torch.no_grad():
            output, _ = model(torch.zeros((1, 8), dtype=torch.long))
        assert torch.isfinite(output).all()
        checks.append(dict(model=row["model"], arm=row["arm"], seed=row["seed"], hash_verified=True, cpu_inference_finite=True))
    result = dict(status="passed", snapshots=len(checks), checks=checks, scope="Weight-file and reconstructed parameter hashes plus finite CPU inference; official test data not re-evaluated.")
    (ROOT / "artifacts/v6/verification-snapshots.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status="passed", snapshots=len(checks))))

if __name__ == "__main__":
    main()
