"""Non-secret hardware and dependency inventory, without reading other projects."""
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import psutil
import torch


def environment():
    def sysctl(name):
        try:
            return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    try:
        swap = psutil.swap_memory().used
    except OSError:
        swap = None
    return {"platform": platform.platform(), "machine": platform.machine(),
            "cpu": sysctl("machdep.cpu.brand_string"), "model": sysctl("hw.model"),
            "ram_bytes": psutil.virtual_memory().total,
            "available_ram_bytes": psutil.virtual_memory().available,
            "swap_used_bytes": swap, "python": sys.version.split()[0],
            "torch": torch.__version__, "numpy": numpy.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads()}


if __name__ == "__main__":
    result = environment()
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
