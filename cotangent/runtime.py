"""Timing, resource guards, hashes, and deterministic data schedules."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import psutil
import torch


def synchronize(device):
    if str(device) == "mps":
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def model_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def source_hash(root):
    h = hashlib.sha256()
    files = []
    for folder in ["cotangent", "scripts", "tests", "configs"]:
        files.extend(p for p in (Path(root) / folder).rglob("*") if p.is_file() and p.suffix in {".py", ".json"})
    for p in sorted(files):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


class Guard:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.swap_baseline = psutil.swap_memory().used
        self.peak_rss = 0
        self.peak_driver = 0
        self.peak_allocated = 0
        self.peak_swap_delta = 0

    def check(self):
        rss = self.process.memory_info().rss
        swap_delta = max(0, psutil.swap_memory().used - self.swap_baseline)
        driver = torch.mps.driver_allocated_memory() if str(self.device) == "mps" else 0
        allocated = torch.mps.current_allocated_memory() if str(self.device) == "mps" else 0
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_driver = max(self.peak_driver, driver)
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_swap_delta = max(self.peak_swap_delta, swap_delta)
        if rss > 5 * 2**30 or driver > 8 * 2**30 or swap_delta > 512 * 2**20:
            raise RuntimeError(f"RESOURCE_GUARD: RSS={rss}, MPS driver={driver}, swap delta={swap_delta}")
        return {"rss": rss, "mps_driver": driver, "mps_allocated": allocated, "swap_delta": swap_delta}

    def summary(self):
        return {"peak_rss_bytes": self.peak_rss, "peak_mps_driver_bytes": self.peak_driver,
                "peak_mps_allocated_bytes": self.peak_allocated, "peak_swap_delta_bytes": self.peak_swap_delta}


def load_bytes(path):
    return np.memmap(path, dtype=np.uint8, mode="r")


def batch(data, offsets, sequence, device):
    values = np.stack([data[int(o):int(o) + sequence + 1] for o in offsets]).astype(np.int64)
    tensor = torch.from_numpy(values).to(device)
    return tensor[:, :-1], tensor[:, 1:]


def make_schedule(length, steps, batch_size, sequence, seed):
    rng = np.random.default_rng(seed + 100000)
    return rng.integers(0, length - sequence - 1, size=(steps, batch_size), dtype=np.int64)


def schedule_hash(schedule):
    return hashlib.sha256(schedule.tobytes()).hexdigest()
