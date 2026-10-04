"""Predeclare the extension after training-only screening and numerical checks."""
import json
import random
from datetime import datetime, timezone
from research.v6.runtime import ROOT, check_prior, digest
from research.v6.models import CONFIGS

def main():
    target = ROOT / "artifacts/v6/frozen-study.json"
    if target.exists():
        raise FileExistsError(target)
    verification = json.loads((ROOT / "artifacts/v6/verification.json").read_text())
    assert verification["primary_dtype_boundary_pass"] and verification["model_parity_pass"] and verification["frozen_v4_update_match"]
    pilots = json.loads((ROOT / "artifacts/v6/pilots/index.json").read_text())
    baselines, screening = {}, {}
    for model in CONFIGS:
        candidates = []
        for arm in ("scalar-native", "foreach-native", "auto-native", "loop-native", "compile-native", "aot-native"):
            rows = [r for r in pilots if r["model"] == model and r["arm"] == arm]
            valid = len(rows) == 2 and all(r["status"] == "completed" for r in rows)
            mean = sum(r["warmed_training_seconds"] for r in rows) / 2 if valid else None
            screening[model + ":" + arm] = dict(eligible=valid, mean_warmed_training_seconds=mean)
            if valid:
                candidates.append((mean, arm))
        assert candidates, model
        baselines[model] = min(candidates)[1]
    blocks = []
    for i, seed in enumerate(range(2001, 2009)):
        arms = ["scalar-native", "bucket-native", "scalar-packed", "bucket-packed"]
        arms = arms[i % 4:] + arms[:i % 4]
        blocks.append([dict(model="transformer-small", arm=arm, seed=seed, steps=800, comparison="factorial") for arm in arms])
    for model in CONFIGS:
        for i, seed in enumerate(range(3001, 3013)):
            arms = [baselines[model], "bucket-packed"]
            if i % 2:
                arms.reverse()
            blocks.append([dict(model=model, arm=arm, seed=seed, steps=1200, comparison="breadth") for arm in arms])
    random.Random(2026100306).shuffle(blocks)
    protocol = dict(version=6, frozen_at=datetime.now(timezone.utc).isoformat(), fingerprints=check_prior(), configs=CONFIGS,
                    jobs=[j for b in blocks for j in b], baselines=baselines, screening=screening,
                    pilot_index_sha256=digest(ROOT / "artifacts/v6/pilots/index.json"), verification_sha256=digest(ROOT / "artifacts/v6/verification.json"),
                    design="Eight four-arm balanced Latin-order factorial blocks; twelve paired seed blocks per architecture against the fastest eligible training-only native baseline, balanced six AB/six BA. All blocks shuffled once before official runs.",
                    primary_metric="Paired fixed-budget training time, including transfer, all forward/backward/gradient handling/AdamW and equal host/resource checks, with every step synchronized. Setup and complete evaluation separately reported.",
                    analysis="Descriptive extension, not a replacement for V5. Report all paired observations, 99% two-sided Student-t intervals per configuration and factorial contrasts. No new universal quality/speed success gate; no optional sample extension. V5 remains the sole powered time-to-target/quality decision.",
                    quality="Full reused official WikiText-2 validation/test splits evaluated once after each run, without tuning on those outcomes. Last 5% of training is used only for pilots/progress diagnostics. Broader quality results are descriptive; no claim of fresh-corpus generalization.",
                    compilation="Fullgraph static-shape torch.compile forward/backward with native fused AdamW and clipping; compile failures/timeouts retained. Successful compiler paths qualify through two training-only pilot seeds. Setup includes compilation. No eager fallback silently relabelled compiled.",
                    hardware="One available Apple M5 Pro; CPU restriction checks are not an independent Apple-device replication. CUDA and distributed systems unavailable locally.",
                    publication="All weights local; source, raw records, protocols, statistics, figures and paper public.")
    target.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps(dict(jobs=len(protocol["jobs"]), baselines=baselines, protocol_sha256=digest(target))))

if __name__ == "__main__":
    main()
