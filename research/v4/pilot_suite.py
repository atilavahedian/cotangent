"""Serial training-only pilots, preserving every completed measurement."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for size,seeds,steps in (("small",[101,109],500),("medium",[101],300)):
 for seed in seeds:
  for arm in ("native-fused","native-bucket","incidence-fused","incidence-bucket"):
   output=ROOT/"artifacts/v4/pilots"/f"{size}-{arm}-{seed}"
   earlier=ROOT/"artifacts/v4/pilots"/f"{arm}-{seed}"
   if size=="small" and earlier.exists(): output=earlier
   if (output/"summary.json").exists():
    assert json.loads((output/"summary.json").read_text())["status"]=="completed"
    continue
   subprocess.run([sys.executable,"-m","research.v4.pilot","--arm",arm,"--seed",str(seed),"--size",size,"--steps",str(steps),"--output",str(output)],cwd=ROOT,check=True)
print("All pre-freeze training-only pilots completed.")
