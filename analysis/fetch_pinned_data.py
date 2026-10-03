"""Reproduce the frozen byte preparation from its existing manifest.

Unlike the original discovery downloader, this never resolves a moving revision
or rewrites the pinned manifest. Existing files are verified, never overwritten.
"""
from pathlib import Path
import argparse
import hashlib
import json
from urllib.request import urlopen
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"data")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/"data/manifest.json").read_text())
    for split,entry in manifest["splits"].items():
        dest=args.output_dir/f"{split}.bin"
        if dest.exists():
            assert digest(dest.read_bytes())==entry["sha256"], f"Existing data mismatch: {dest}"
            print(json.dumps({"split":split,"status":"verified-existing","sha256":entry["sha256"]}),flush=True)
            continue
        text=[]
        for index,source in enumerate(entry["sources"]):
            with urlopen(source["url"],timeout=180) as response:
                payload=response.read()
            assert digest(payload)==source["sha256"], f"Pinned parquet mismatch: {split}/{index}"
            import pyarrow as pa
            rows=pq.read_table(pa.BufferReader(payload),columns=["text"])["text"].to_pylist()
            assert len(rows)==source["rows"]
            text.extend(rows)
        payload="\n".join(text).encode("utf-8")
        assert len(text)==entry["rows"] and len(payload)==entry["bytes"]
        assert digest(payload)==entry["sha256"], f"Preparation mismatch: {split}"
        with dest.open("xb") as f:
            f.write(payload)
        print(json.dumps({"split":split,"status":"reproduced","sha256":entry["sha256"],"bytes":len(payload)}),flush=True)


if __name__=="__main__":
    main()
