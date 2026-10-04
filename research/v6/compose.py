"""Test composition with the compiler before the official protocol is frozen."""
import gc
import json
import os
import subprocess
import sys
import torch
from cotangent.runtime import batch, load_bytes, make_schedule, model_hash
from research.v3.embedding import install
from research.v6.models import make_model, CONFIGS
from research.v6.methods import Handler, forward_callable
from research.v6.runtime import ROOT, check_prior

def main():
    torch.set_num_threads(4)
    check_prior()
    root = ROOT / "artifacts/v6/pilots"
    index_path = root / "index.json"
    rows = json.loads(index_path.read_text())
    checks = []
    for name in CONFIGS:
        viable = [r for r in rows if r["model"] == name and r["arm"] == "compile-native"]
        if not (len(viable) == 2 and all(r["status"] == "completed" for r in viable)):
            continue
        for seed in (1801, 1802):
            arm = "compile-bucket"
            output = root / f"{name}--{arm}--{seed}"
            if output.exists():
                raise FileExistsError(output)
            log = root / (output.name + ".log")
            command = [sys.executable, "-m", "research.v6.run", "--model", name, "--arm", arm, "--seed", str(seed), "--steps", "120", "--output", str(output)]
            env = dict(os.environ, TORCHINDUCTOR_CACHE_DIR=str(ROOT.parents[1] / "work/inductor-v6"), TORCHINDUCTOR_COMPILE_THREADS="1")
            with log.open("w") as f:
                proc = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=180)
            s = json.loads((output / "summary.json").read_text())
            assert proc.returncode == 0
            record = dict(model=name, arm=arm, seed=seed, status="completed", log=str(log.relative_to(ROOT)),
                          summary=str((output / "summary.json").relative_to(ROOT)), warmed_training_seconds=s["warmed_training_seconds"],
                          training_seconds=s["training_seconds"], diagnostic_bpb=s["diagnostic_bpb"], setup_seconds=s["setup_seconds"])
            rows.append(record)
            index_path.write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        # Identical compiled programs and deterministic derivatives in both arms
        # isolate layout/norm/update behavior from compiler arithmetic changes.
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = False
        native, candidate = make_model(name, 1851), make_model(name, 1851)
        install(native); install(candidate)
        first, second = Handler(native, "compile-native"), Handler(candidate, "compile-bucket")
        forwards = forward_callable(native, "compile-native"), forward_callable(candidate, "compile-bucket")
        data = load_bytes(ROOT / "data/train.bin")
        schedule = make_schedule(int(len(data) * .95), 4, 8, 256, 1851)
        for step in range(4):
            first.zero_grad(); second.zero_grad()
            x, y = batch(data, schedule[step], 256, "mps")
            with torch.autocast("mps", dtype=torch.bfloat16):
                _, loss = forwards[0](x, y); _, other_loss = forwards[1](x, y)
            loss.backward(); other_loss.backward()
            gradient_equal = all(torch.equal(a.grad, b.grad) for a, b in zip(native.parameters(), candidate.parameters()))
            norm, other_norm = first.step(), second.step()
            checks.append(dict(model=name, step=step + 1, loss_equal=bool(torch.equal(loss, other_loss)),
                               gradient_equal=gradient_equal, norm_equal=bool(torch.equal(norm, other_norm)),
                               parameters_equal=model_hash(native) == model_hash(candidate)))
        del native, candidate, first, second, forwards, loss, other_loss, norm, other_norm, x, y
        gc.collect(); torch.mps.empty_cache(); torch._dynamo.reset()
        torch.use_deterministic_algorithms(False)
    path = ROOT / "artifacts/v6/verification-compiled.json"
    if path.exists():
        raise FileExistsError(path)
    result = dict(fingerprints=check_prior(), checks=checks,
                  all_passed=all(all(r[k] for k in ("loss_equal", "gradient_equal", "norm_equal", "parameters_equal")) for r in checks),
                  scope="Four BF16 deterministic-adjoint updates per executable compiled architecture, against the same compiler execution path; no claim of equality with eager compiler arithmetic.")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(compiled_checks=len(checks), all_passed=result["all_passed"])), flush=True)

if __name__ == "__main__":
    main()
