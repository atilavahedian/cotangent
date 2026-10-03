# Cotangent

**A standalone research study of cheaper backpropagation.**

Cotangent investigates whether structured gradient approximation and adaptive
computation can reduce the time required to train a transformer without degrading
held-out language-model quality. It is built from a clean directory, with newly
written model, training, and research code. It does not reuse any of the author's
previous projects, weights, data preparations, or experiment results.

**Status: research in progress. No performance advantage has been demonstrated.**

The deliverables are a reproducible implementation, measured experiments,
mathematical notes, raw evidence, and an HTML explanation of what worked and what
did not. Approximation overhead, unsuccessful trials, and missing comparisons
remain part of the record.

## Research questions

1. Can a lower-rank backward computation reach comparable held-out quality in less
   end-to-end training time than a strong exact-backpropagation baseline?
2. Does allocating approximation accuracy adaptively outperform a fixed budget?
3. Do memory or arithmetic savings actually translate into a useful training gain
   on the measured hardware?

## Related work

- [Low-rank backpropagation via Walsh–Hadamard projections, NeurIPS 2023](https://arxiv.org/abs/2309.15275)
- [INSTANT: gradient and activation compression, ICLR 2026](https://github.com/hieu-trannn/INSTANT)
- [APOLLO: memory-efficient optimizer states](https://arxiv.org/abs/2412.05270)
- [Moonwalk: inverse-forward differentiation, AISTATS 2026](https://proceedings.mlr.press/v300/krylov26a.html)

These methods address different bottlenecks. This project will name exactly which
comparisons were implemented and measured; a local improvement is not evidence
of superiority to all existing training methods.

## License

New project code is MIT licensed. External datasets and any explicitly attributed
third-party material retain their own licenses. Training data is downloaded by the
user and is not included in this repository.
