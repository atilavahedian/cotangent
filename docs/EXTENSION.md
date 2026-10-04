# Extension study

The extension asks three questions: which part of gradient handling accounts for
the measured saving, whether it transfers to other model layouts, and whether
the comparison survives stronger executable baselines on the available machine.

The factorial study crosses scalar versus bucketed parameter norms with native
versus packed fused AdamW. It preserves the two-level global norm in all four
treatments. A separate breadth study compares the combined treatment with the
fastest eligible baseline selected on training-only pilots for each architecture.

The models are the existing 3.35M and 19.28M transformers, a new grouped-query
attention decoder with RMSNorm and gated feed-forward layers, and a causal
dilated convolutional language model. All predict bytes from the same pinned
WikiText-2 corpus. The extension evaluates full validation and test splits only
after each planned run. Previously exposed official splits remain quality
regression data, not a new generalization test.

Compilation is exercised directly on the installed MPS backend, with graph
fallback disabled. Failures and timeouts are observations, not speedup results.
The private multi-tensor norm/multiply operations are also screened because the
public clipping API rejects foreach=True on MPS. Baseline selection uses warmed
training time on two diagnostic seeds; compilation/setup cost is reported
separately and charged to end-to-end time.

Eight factorial seed blocks use a balanced Latin order. Each of four architecture
comparisons uses twelve paired seeds, six in each execution order. Blocks are
shuffled before official runs. Sample sizes are fixed. Extension intervals are
descriptive 99% paired Student-t intervals per comparison; the independent V5
study retains its original powered decision. A gain on one architecture cannot
rescue an adverse result on another.

Numerical checks cover the full model and optimizer states for both FP32 and
BF16 forward precision, with matched deterministic embedding derivatives to
isolate update arithmetic. Norm sweeps also cover odd lengths, short lengths,
threadgroup boundaries, bucket sizes, FP32, FP16 and BF16 tensors. Each dtype is
reported separately. Official training retains native embedding derivatives.

The resource guard caps sampled process RSS at 5 GiB, MPS driver memory at
8 GiB, and swap growth at 512 MiB. Training runs are serial. One Apple M5 Pro is
available; multiple-device and CUDA replication remain outside the measured
scope. Existing studies and their source fingerprints remain unchanged.
