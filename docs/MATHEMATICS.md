# What Cotangent approximates

Reverse-mode differentiation pulls output cotangents back through transposed
Jacobians. The name reflects that operation: the object being approximated here
is the parameter cotangent of a linear map, while its input cotangent stays exact.

## A sum of rank-one contributions

For a linear map `Y = X W^T`, let `D = dL/dY`. With batch and token dimensions
flattened, `X` has shape `N x d_in`, `D` has shape `N x d_out`, and:

```
G = dL/dW = D^T X = sum_i d_i x_i^T
dL/dX = D W
```

Write `Z_i = d_i x_i^T`. For independently sampled indices `i_j ~ p`, use:

```
G_hat = (1/k) sum_(j=1..k) Z_(i_j) / p_(i_j)
```

This is a standard importance-sampled matrix-product estimator. Its rank is at
most `k`. Sampling and weighting happen before the matrix product, rather than
after an expensive full gradient has already been calculated. Duplicate draws
are retained and their cost is measured.

With replacement, the estimator is also known as a Hansen–Hurwitz estimator.
The frozen implementation's docstring uses the broader inverse-probability name
"Horvitz–Thompson"; that name conventionally refers to without-replacement
inclusion weighting. The equations above specify the actual implemented estimator.

The estimator follows the classical sampled matrix-product construction in
[Drineas, Kannan and Mahoney (2006)](https://doi.org/10.1137/S0097539704442684).
Neither importance sampling nor low-rank backward computation is claimed as a
new invention. The project investigates this controller and implementation.

Conditioned on the current model and minibatch, with positive probabilities:

```
E[G_hat] = G
E[||G_hat - G||_F^2] = (sum_i ||Z_i||_F^2 / p_i - ||G||_F^2) / k
||Z_i||_F = ||d_i||_2 ||x_i||_2
```

Minimizing the second moment subject to `sum_i p_i = 1` gives
`p_i proportional to ||d_i||_2 ||x_i||_2`, by a Lagrange multiplier or
Cauchy–Schwarz. The implementation mixes this distribution with 5% uniform mass
to keep probabilities positive and limit extreme sampling weights.

The arithmetic gain is bounded: activation gradients still require `D W`.
Ignoring projection/sampling overhead, approximating only the weight gradient
changes a dense layer's forward-plus-backward cost from roughly `6 N d_in d_out`
to `(4 N + 2 k) d_in d_out`. The ideal per-layer limit is 1.5x as `k -> 0`;
attention, nonlinearities, the output head, AdamW and overhead reduce that limit.
This method does **not** reduce saved activation storage. Peak-memory observations
must not be represented as a designed activation-compression benefit.

## Why activation gradients stay exact

Projecting the activation gradient changes the signal arriving at earlier layers.
Nonlinearities can then amplify or suppress that error. Cotangent's sampled
candidate computes `D W` exactly, so (before finite-precision effects) every
layer's sampling estimator operates on the same forward activations and incoming
gradient that exact backpropagation would produce at that current model state.

The WHT-style comparator does compress both products. It uses an orthonormal
sequence-space basis `P`:

```
X_c = P^T X
D_c = P^T D
G_hat = D_c^T X_c
dX_hat = P (D_c W)
```

This saves stored linear activations and is exact at full rank, but at reduced
rank it is biased. It is an independently written mechanism control inspired by
LBP-WHT. It is not the official LBP-WHT implementation or a reproduction of
INSTANT's calibrated, separately chosen projections.

## Adaptive budget

Periodic audit steps compute the exact weight gradient and record its norm, the
importance second moment, and the error of an independent sampled estimate. A
single batched device-to-host read selects the following interval's sample count:

```
k_est = ceil((second_moment / last_audited_norm_squared - 1) / tolerance^2)
```

The sample count is clipped to the available rows. If it reaches 75% of the
available rows, ordinary steps use native exact backpropagation, avoiding
probability construction and gathering overhead. Otherwise the probabilities
reflect the current minibatch but the sample count remains fixed until the next
audit. No per-layer host synchronization is needed on ordinary steps.

**The stale second moment and gradient norm make this a heuristic, not a certified per-step error
bound.** Adaptation uses no held-out observations. Audit steps, sampling costs,
CPU synchronization and dense fallbacks all count in timing. A large required
sample count is evidence that this approximation may be unsuitable; it is not
permission to weaken the accuracy criterion after observing outcomes.

## A limited convergence statement

For an `L`-smooth objective, an unbiased gradient estimator with conditional
variance at most `sigma^2`, and the plain SGD update `theta' = theta - eta g_hat`,
the smoothness inequality yields:

```
E[f(theta')] <= f(theta)
              - eta (1 - L eta/2) ||grad f(theta)||^2
              + L eta^2 sigma^2 / 2
```

Under bounded variance, an objective bounded below, and suitable step sizes,
standard summation bounds give convergence toward stationary points in
expectation. This is not a theorem of global optimality, and does not establish a
useful bound for a real transformer. The actual experiments use AdamW and gradient
clipping; clipping is nonlinear and need not preserve unbiasedness. Therefore the
SGD statement is explanatory theory, not a convergence guarantee for the measured
training procedure. The controller's stale-norm approximation must be evaluated.

## Verification

Tests check exact custom derivatives against autograd, full-rank WHT equivalence,
analytic unbiasedness, Monte Carlo variance, zero-signal probability handling,
exact activation-gradient propagation, and whole-transformer gradient equivalence.
The final research result still depends on actual training and held-out evaluation.
