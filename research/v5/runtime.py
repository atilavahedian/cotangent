"""Source fingerprint for the independent replication harness."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def source_hash():
 h=hashlib.sha256()
 for p in sorted((ROOT/"research/v5").glob("*.py")):
  h.update(p.name.encode());h.update(p.read_bytes())
 return h.hexdigest()
