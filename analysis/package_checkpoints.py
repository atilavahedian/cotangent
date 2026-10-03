"""Package every completed frozen snapshot for public research reproduction."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    result=json.loads((ROOT/"artifacts/analysis/results.json").read_text())
    assert result["runs"]==36 and result["integrity_checks_passed"]
    out=ROOT/"artifacts/release"
    out.mkdir(exist_ok=True)
    archives=[]
    for group in ("small","medium"):
        members=[r for r in result["evidence"] if r["group"]==group]
        dest=out/f"cotangent-{group}-checkpoints.tar.gz"
        with tarfile.open(dest,"w:gz",compresslevel=1) as archive:
            for row in members:
                path=ROOT/row["checkpoint"]
                assert hashlib.sha256(path.read_bytes()).hexdigest()==row["checkpoint_sha256"]
                archive.add(path,arcname=row["checkpoint"],recursive=False)
        archives.append({"file":dest.name,"bytes":dest.stat().st_size,
                         "sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
                         "members":[{"path":r["checkpoint"],"sha256":r["checkpoint_sha256"]} for r in members]})
    manifest={"study":"Cotangent protocol-v1","source_sha256":result["frozen_source_sha256"],
              "snapshot_scope":"Model weights and metadata only; optimizer states are not included.",
              "archives":archives}
    (out/"checkpoints-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"archives":[{k:v for k,v in a.items() if k!="members"} for a in archives]},indent=2))


if __name__=="__main__":
    main()
