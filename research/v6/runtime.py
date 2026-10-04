"""Fingerprints for new sources and every previously frozen method."""
import hashlib
import json
from pathlib import Path
from cotangent.runtime import source_hash as v1
from research.v2.pilot import source_hash as v2
from research.v3.pilot import source_hash as v3
from research.v4.pilot import source_hash as v4
from research.v5.runtime import source_hash as v5
ROOT = Path(__file__).resolve().parents[2]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def fingerprints():
    h = hashlib.sha256()
    for p in sorted((ROOT / "research/v6").glob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes())
    return dict(v1=v1(ROOT), v2=v2(), v3=v3(), v4=v4(), v5=v5(), v6=h.hexdigest(), data=digest(ROOT / "data/manifest.json"))

def check_prior():
    current = fingerprints()
    old = json.loads((ROOT / "artifacts/v5/frozen-study.json").read_text())
    for key in ("v1", "v2", "v3", "v4"):
        assert current[key] == old[key + "_source_sha256"], key
    assert current["v5"] == old["source_sha256"]
    assert current["data"] == old["data_manifest_sha256"]
    return current
