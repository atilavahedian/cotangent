"""Restore each released model and check its identity and finite exact inference.

Uses an authored prompt, never repeats test-set evaluation or selects models.
"""
from pathlib import Path
import hashlib
import json
import sys
import torch
from infer import load

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cotangent.runtime import model_hash


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    r=json.loads((ROOT/"artifacts/analysis/results.json").read_text())
    checks=[]
    prompt=torch.tensor([list(b"The history of science ")],dtype=torch.long)
    for row in r["evidence"]:
        summary=json.loads((ROOT/row["summary"]).read_text())
        model,saved=load(ROOT/row["checkpoint"],"cpu")
        actual=model_hash(model)
        assert actual==summary["final_model_sha256"]
        assert saved["source_sha256"]==r["frozen_source_sha256"]
        logits,_=model(prompt)
        assert logits.shape==(1,prompt.shape[1],256)
        assert torch.isfinite(logits).all()
        checks.append({"checkpoint":row["checkpoint"],"strict_state_restore":True,
                       "model_sha256":actual,"model_hash_matches_summary":True,
                       "finite_exact_inference":True,"backend":"cpu-fp32"})
        del model, saved, logits
    out={"status":"passed","snapshots":len(checks),
         "scope":"Strict checkpoint restoration, trained weight identity and finite exact inference on an authored prompt. No held-out reevaluation or capability claim.",
         "checks":checks}
    (ROOT/"artifacts/snapshot-verification.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":"passed","snapshots":len(checks)}))


if __name__=="__main__":
    main()
