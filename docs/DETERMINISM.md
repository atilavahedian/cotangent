# Exact adjoints, execution variability and the V3 experiment

V2's flat gradient norm/AdamW saved training time but failed both primary gates.
The immutable 24-run result remains in `artifacts/v2/analysis/results.json`.

An unchanged native baseline repeat used the same source, initialization, batch
schedule and optimizer settings as medium seed 257. Its full-test BPB changed
from **2.692846 to 2.828499**, a **0.135652-BPB difference**. This is a single
post-study diagnostic, excluded from every frozen comparison. It demonstrates
execution sensitivity in that run; it does not identify all causes of all gaps.

The installed Torch source revision is
`5c4886908584029761b579af026dcfb627c84070`. Its
[MPS embedding backward](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/Embedding.mm)
dispatches a row-accumulation kernel. The [embedding Metal shader at that same revision](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/kernels/Embedding.metal)
uses floating-point atomic addition. Current main adds a nondeterminism alert,
but the installed binary **accepts** our small strict-determinism probe. The
actual runtime behavior is recorded; the flag alone is not a reproducibility
certificate.

## Incidence matrices give the exact embedding adjoint

Let `z_i` be a byte index, `W` a `V × d` embedding table and `H` the `N × V`
incidence matrix: `H[i,v] = 1` if `z_i = v`, otherwise zero. Forward lookup is
`E = HW`. For incoming cotangents `D = dL/dE`,

```
dL/dW = H^T D
```

This sums every repeated-index contribution. The project constructs the zeros
and ones by an integer comparison and computes the adjoint in **FP32**, even
under BF16 transformer autocast. No rows or cotangents are discarded or rounded
to BF16. Forward lookup remains native and identical.

The positional lookup is a unique prefix `arange(T)`. Its broadcast-add backward
has already summed the batch dimension. Its weight adjoint is that incoming
gradient with unused trailing rows zero-padded, avoiding a general scatter.

This deterministically structured adjoint is followed by V2's mathematically
equivalent contiguous gradient/norm/AdamW update. Other transformer derivatives
remain native. Floating-point reduction trees can differ from the native atomic
implementation; the final quality gate is not replaced by the algebraic proof.

## Preserve experiments that did not help

Initial deterministic prototypes incurred redundant NaN fills from
`torch.utils.deterministic.fill_uninitialized_memory`. Our operations initialize
their complete outputs before reading them; disabling those fills is explicitly
permitted in [PyTorch's reproducibility guidance](https://docs.pytorch.org/docs/main/notes/randomness.html#filling-uninitialized-memory).
The slower measurements are retained. Seventeen CPU derivative/analysis tests
passed; full-model FP32/BF16 MPS checks compare native gradients and repeat the
candidate's full gradients bitwise.

An independently authored Metal segmented-reduction kernel was also tested.
Stable row sorting gives each vocabulary/feature coordinate one writing thread
and a fixed summation order. GPU-sorted and CPU-batch-plan variants passed direct
GPU checks, including repeated indices and unused vocabulary rows. They did not
show a convincing speed advantage over the simpler incidence implementation in
training-only pilots; the final candidate therefore uses native FP32 GEMM.
The Metal prototypes remain inspectable rather than being presented as wins.

Preserving native per-parameter clipping norms made the deterministic native and
packed pilot models **bitwise identical** across two seeds, but saved too little
time. The selected global-norm version must demonstrate its own trained-quality
noninferiority. None of those pilot losses use the official validation/test data.

## The independent freeze

V3 retains V2's **10% minimum elapsed reduction** and **0.01-BPB maximum upper
quality bound**, rather than relaxing either failed requirement. It predeclares
15 new primary paired seeds, five deterministic-control ablations, and three
larger-model pairs: **41 runs**. The larger comparison and ablations are
descriptive; they cannot substitute for a failed primary comparison. All planned
runs are preserved, and there is no optional stopping.

The official WikiText-2 corpus is openly reused. The supported inference concerns
timing and a quality regression on this corpus, not new unseen-domain capability.
Dense incidence computation is specific to a bounded vocabulary here (`V=256`);
its `O(N V d)` arithmetic and extra buffers do not establish a large-vocabulary
or activation-memory improvement. The established linear adjoint, GEMM, packing
and fused optimizer are credited as prior art; the research contribution is their
tested application and the numerical diagnosis, not a claimed new chain rule.
