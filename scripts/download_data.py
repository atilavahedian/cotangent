"""Download and pin the public WikiText-2 raw dataset; never use another project."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATASET = "Salesforce/wikitext"
CONFIG = "wikitext-2-raw-v1"


def fetch_json(url: str):
    with urlopen(url, timeout=90) as response:
        return json.load(response)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DATA.mkdir(exist_ok=True)
    if (DATA / "manifest.json").exists():
        raise SystemExit("A pinned manifest already exists. Refusing to replace it.")
    split_url = "https://datasets-server.huggingface.co/splits?" + urlencode({"dataset": DATASET})
    parquet_url = "https://datasets-server.huggingface.co/parquet?" + urlencode({"dataset": DATASET})
    split_meta = fetch_json(split_url)
    parquet_meta = fetch_json(parquet_url)
    revision = fetch_json(f"https://huggingface.co/api/datasets/{DATASET}")["sha"]
    tree = fetch_json(f"https://huggingface.co/api/datasets/{DATASET}/tree/{revision}/{CONFIG}?recursive=true")
    entries = [item for item in tree if item["type"] == "file" and item["path"].endswith(".parquet")]
    known = {item["split"] for item in split_meta["splits"] if item["config"] == CONFIG}
    assert known == {"train", "validation", "test"}
    manifest = {"dataset": DATASET, "config": CONFIG, "revision": revision,
                "license_note": "Dataset card states CC BY-SA 4.0; repository metadata lists CC BY-SA 3.0. Source data is not redistributed.",
                "source": f"https://huggingface.co/datasets/{DATASET}",
                "viewer_requests": [split_url, parquet_url],
                "processing": "Read text column in original row order; join rows with newline; UTF-8 bytes, alphabet 0..255.",
                "splits": {}}
    for split in ["train", "validation", "test"]:
        files = sorted([item for item in entries if Path(item["path"]).name.startswith(split + "-")], key=lambda item: item["path"])
        assert files, f"No pinned parquet files found for {split}"
        text = []
        records = []
        for item in files:
            filename = Path(item["path"]).name
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{item['path']}"
            dest = DATA / f"{split}-{filename}"
            with urlopen(url, timeout=180) as response:
                dest.write_bytes(response.read())
            values = pq.read_table(dest, columns=["text"])["text"].to_pylist()
            text.extend(values)
            records.append({"url": url, "sha256": sha256(dest), "rows": len(values), "bytes": dest.stat().st_size})
        payload = "\n".join(text).encode("utf-8")
        dest = DATA / f"{split}.bin"
        dest.write_bytes(payload)
        manifest["splits"][split] = {"rows": len(text), "bytes": len(payload), "sha256": sha256(dest), "sources": records}
        print(json.dumps({"split": split, "rows": len(text), "bytes": len(payload), "sha256": sha256(dest)}), flush=True)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
