import json
from pathlib import Path
import torch
from torch.nn import functional as F
from research.v3.segmented import reduce

torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)
checks = []
for n, vocab, width in ((11, 5, 7), (2048, 256, 256), (2048, 256, 512)):
    torch.manual_seed(97)
    indices = torch.randint(vocab, (n,))
    # Guarantee repeated indices and entirely unvisited rows in the edge case.
    if vocab == 5:
        indices = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0])
    gradient = torch.randn(n, width)
    expected = torch.zeros(vocab, width, dtype=torch.float64)
    for i, index in enumerate(indices):
        expected[index] += gradient[i].double()
    mi, mg = indices.to("mps"), gradient.to("mps")
    a, b = reduce(mi, mg, vocab), reduce(mi, mg, vocab)
    torch.mps.synchronize()
    assert torch.equal(a.cpu(), b.cpu())
    error = float((a.cpu().double() - expected).abs().max())
    relative = error / float(expected.abs().max())
    assert relative < 3e-6, relative
    assert a[vocab-1].abs().max().item() == 0. if vocab == 5 else True
    checks.append(dict(rows=n, vocabulary=vocab, width=width, deterministic_repeated_outputs=True,
                       maximum_absolute_error_against_cpu_double=error, scaled_error=relative))
root = Path(__file__).resolve().parents[2]
(root / "artifacts/v3").mkdir(parents=True, exist_ok=True)
result = dict(status="passed", checks=checks, scope="Actual authored Metal kernel versus independently accumulated CPU double reference, with repeats and unvisited rows.")
(root / "artifacts/v3/verification-segmented.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(result))
