"""Hardware derivative equivalence and BF16 finite-gradient checks."""
import json
from pathlib import Path

import torch

from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(4)
    cfg = ModelConfig(width=64, layers=2, heads=4, sequence=32)
    output = []
    for precision in ["fp32", "bf16"]:
        torch.manual_seed(10)
        native = Transformer(cfg, Policy(mode="native")).to("mps")
        torch.manual_seed(10)
        custom = Transformer(cfg, Policy(mode="exact_custom", collect=False)).to("mps")
        tokens = torch.arange(64, device="mps").reshape(2, 32)
        targets = (tokens + 1) % 256
        for model in [native, custom]:
            with torch.autocast("mps", dtype=torch.bfloat16, enabled=precision == "bf16"):
                model(tokens, targets)[1].backward()
        ng = torch.cat([p.grad.flatten().float() for p in native.parameters()])
        cg = torch.cat([p.grad.flatten().float() for p in custom.parameters()])
        relative = float(((ng - cg).norm() / ng.norm()).item())
        cosine = float(torch.nn.functional.cosine_similarity(ng, cg, dim=0).item())
        threshold = 1e-5 if precision == "fp32" else 0.015
        assert relative < threshold, (precision, relative)
        assert torch.isfinite(cg).all().item()
        output.append({"precision": precision, "relative_gradient_error": relative,
                       "cosine_similarity": cosine, "threshold": threshold, "passed": True})
    path = ROOT / "artifacts/verification-mps.json"
    path.write_text(json.dumps({"backend": "mps", "checks": output}, indent=2) + "\n")
    print(path.read_text())


if __name__ == "__main__":
    main()
