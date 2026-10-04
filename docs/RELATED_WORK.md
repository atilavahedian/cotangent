# Related work and source review

## Training systems

AdamW separates decoupled weight decay from the adaptive gradient update
([Loshchilov and Hutter, 2019](https://arxiv.org/abs/1711.05101)). Its moment
updates act coordinatewise, which permits a consistent reindexing of parameters,
gradients and optimizer states under shared hyperparameters and step counts.

[PyTorch FSDP flat parameters](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_flat_param.py)
provide an existing example of parameter flattening and shared views in a
distributed training system. The present experiments concern a single-device,
unsharded optimizer layout. They do not benchmark distributed FSDP.

[DeepSpeed FusedAdam](https://deepspeed.readthedocs.io/en/latest/_modules/deepspeed/ops/adam/fused_adam.html)
combines elementwise Adam operations and multi-tensor launch batching.
[NVIDIA Apex's multi-tensor L2 norm](https://github.com/NVIDIA/apex/blob/master/csrc/multi_tensor_l2norm_kernel.cu)
also provides direct prior art for evaluating norms across multiple tensors.
Those CUDA implementations are source comparisons here, not measured MPS
baselines. The executable alternatives are recorded in the V6 pilot index:
native fused, automatic and loop AdamW; native multi-tensor norm/multiply
operations; fullgraph TorchInductor; and AOT eager graph capture.

[Ansel et al., 2024](https://docs.pytorch.org/assets/pytorch2-2.pdf) describe
TorchDynamo and TorchInductor. V6 exercises the installed Metal compiler path,
not the CUDA performance conclusions of that paper. The model's forward and
backward are compiled with static shapes and fullgraph capture. Gradient
handling and native fused AdamW remain outside that compiled model call in both
comparison arms. Compilation and gradient handling are therefore complementary
components of the measured execution path.

## Floating-point reductions

[Ahrens, Demmel and Nguyen, 2020](https://doi.org/10.1145/3389360) address
reproducible summation with accumulators whose results are independent of input
summation order. Cotangent instead preserves the pinned backend's existing
reduction behavior for the tested layouts. It adds no order-independent
accumulator and supplies no cross-device reproducibility guarantee.

The inspected PyTorch source is pinned to revision
5c4886908584029761b579af026dcfb627c84070. In
[norm_kernel_mps](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/mps/operations/ReduceOps.mm),
the inner L2 fast path requires more than one output element and exactly one
nontrivial reduced dimension at the innermost axis. The B x (K/2) x 2 view
avoids that branch when both reduced axes are nontrivial. The generic path uses
reduction_size=K and min(MAX_THREADGROUP_SIZE,K) threads per group. The retained
V4 probe demonstrates that ordinary single-axis row batching can change norms.

The source review supports the dispatcher diagnosis. The numerical controls
test its consequence for the gradient layouts used in the experiments.

## Models and corpus

The GQA decoder uses grouped query/key-value heads
([Ainslie et al., 2023](https://arxiv.org/abs/2305.13245)), a gated feed-forward
layer ([Shazeer, 2020](https://arxiv.org/abs/2002.05202)), and RMSNorm
([Zhang and Sennrich, 2019](https://arxiv.org/abs/1910.07467)). It is trained
from scratch; it is not an uptrained checkpoint from the GQA paper.
The convolutional model uses residual causal dilated convolutions, a design
associated with [WaveNet](https://arxiv.org/abs/1609.03499), adapted here to byte
language prediction with depthwise convolution and pointwise feed-forward layers.

The corpus is WikiText-2, introduced with
[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843).
All models operate on a pinned byte serialization, rather than reproducing the
original paper's word-level perplexity benchmark. Official splits are reused
for disclosed numerical and quality regression checks.

Sources and metadata were checked on October 3, 2026 (client date). The paper
uses normal citations to these sources; measured claims point to the frozen
protocols and public records in this repository.
