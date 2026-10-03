# Exact packing: a second Cotangent experiment

The adaptive sampled backward in V1 failed its frozen comparison. That evidence
remains intact. V2 instead tests an exact systems optimization against a stronger
existing baseline: PyTorch's native **fused AdamW** on MPS.

## What changes

The model computes its full native forward and backward. Independent parameter
leaves share slices of one contiguous master parameter vector. After backward,
the completed gradients are concatenated once, clipped together, and passed to
one fused AdamW optimizer tensor. No gradient is sampled, projected, skipped or
replaced. Packing happens before any forward pass. The original state dictionary
still contains the same named model weights.

This reduces the number of optimizer tensors and norm reductions. Whether it
reduces time is a measured backend question; this project does not infer a kernel
speedup from arithmetic counts. The extra concatenation is fully charged.

## Why the update is mathematically equivalent

Write `theta = concat(theta_1, ..., theta_L)` and `g = concat(g_1, ..., g_L)`.
Native global clipping uses `c = min(1, 1/(||g||_2 + 1e-6))`. Because
`||g||_2^2 = sum_l ||g_l||_2^2`, the clipped vector is exactly the concatenation
of the clipped parameter gradients in real arithmetic.

AdamW's first and second moment recurrences are coordinatewise:

```
m_t = beta1 * m_(t-1) + (1 - beta1) * (c*g_t)
v_t = beta2 * v_(t-1) + (1 - beta2) * (c*g_t)^2
theta_t = (1 - lr_t*decay) * theta_(t-1)
          - lr_t * m_hat_t / (sqrt(v_hat_t) + epsilon)
```

Concatenation commutes with every coordinatewise operation and preserves the
global norm. Therefore the optimizer iterates coincide in real arithmetic when
learning rates, betas, epsilon, decay, step counts and active coordinates match.
Finite precision does not promise identical bits: parallel norm reductions and
fused kernels can reorder floating-point work. We verify numerical updates and
full trained quality rather than assuming bitwise identity.

All our model parameters have the same optimizer hyperparameters and update at
every step. The implementation rejects sparse or missing gradients and mixed
dtypes/devices. It does not support distinct parameter groups, frozen coordinates,
moving the model after packing, or distributed gradient synchronization.

## Evidence and prior art

CPU double tests compare eight successive updates, gradients, norms and both
AdamW moments with and without clipping at 2e-14 tolerance. An actual transformer
check covers FP32 and BF16 on MPS. An initial bitwise-equality assertion failed at
4.66e-10 absolute gradient difference; that failure is preserved, and the backend
check reports numerical tolerances explicitly.

Training-only pilots compare automatic, foreach and fused native AdamW, plus
packed automatic/fused variants. Final evaluation uses the independently frozen
24-run protocol in `artifacts/v2/frozen-study.json`. V1's official held-out corpus
is reused openly as a quality regression check. V2's new seeds and balanced pair
order test whether the timing advantage replicates.

Parameter flattening and gradient buffers are established systems techniques:
for example [PyTorch FSDP's flat parameter implementation](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_flat_param.py).
[PyTorch documents fused AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html)
as an existing optimization. This project claims an implementation and measured
application on this setup, not invention of those techniques or a replacement
for backpropagation. It leaves full activation storage and all derivative matrix
multiplications intact and adds one dense gradient vector.
