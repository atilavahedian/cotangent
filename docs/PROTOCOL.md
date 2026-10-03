# Cotangent research protocol

## Independence

This is a new standalone project. Do not import source, trained checkpoints,
datasets, configurations, results, or research narratives from the author's other
projects. Public literature and general-purpose installed dependencies may inform
new implementations, with attribution.

## Outcomes and comparison

The primary outcome is **end-to-end time to a common held-out quality target**.
Lower arithmetic counts or faster isolated kernels are secondary outcomes.
Comparison runs share model initialization, model size, training data, byte
tokenization, batch-index schedules, optimizer, precision, and evaluation data.
All selector, projection, audit, synchronization, and optimizer overhead is timed.
Evaluation is reported separately and also included in a campaign elapsed-time
measurement. Approximate methods do not get extra data or undisclosed tuning.

The study includes native exact backpropagation, an exact custom-backward control,
a fixed approximation baseline, and an adaptive candidate. Additional controls
may be added before the final experiment is frozen. Checkpointing and mixed
precision are considered during diagnostic preflight. Unsupported backend
features and unfavorable trials are retained, not described as successes.

## Experimental phases

1. **Diagnostic preflight:** validate derivatives and estimator properties, profile
   representative tensor shapes, choose a resource-safe configuration. Use a
   training-only diagnostic holdout for method development. This phase cannot
   establish a final performance advantage.
2. **Frozen comparison:** publish the selected configuration, seed list, data
   hashes, schedules, exact definition of the quality target and tolerances,
   source-tree hash, and stopping rules before collecting final outcomes.
3. **Final evaluation:** run the predetermined arms and paired seeds sequentially,
   with interleaved arm order across seeds. Use the official validation split for
   learning curves and the official test split once per final checkpoint. Do not
   revise the method after seeing test results.
4. **Analysis and release:** retain all results, report paired differences and
   uncertainty, explain failures, publish the source and HTML report.

Pilot-driven changes are documented before the freeze. Any subsequent algorithm
change requires a new explicitly marked exploratory experiment rather than
silently replacing a failed frozen run.

## Success gates

Both quality and real training efficiency are required. A final claim of a useful
training improvement requires the frozen quality tolerance to pass and a positive
time-to-quality gain across the paired seeds. Report individual seeds, not just
the most favorable average. A run that never reaches the common target is
right-censored, not assigned an invented completion time. A memory improvement
alone is reported as a memory improvement. A one-hardware small-model result does
not establish large-model, CUDA, or universal superiority.

## Resource rules

Local execution only. No paid services or remote compute. One training job at a
time; do not modify or kill unrelated processes. Preflight actual memory use.
Use fixed training tensor shapes. Abort and preserve partial evidence if process
RSS exceeds 5 GiB, MPS driver allocation exceeds 8 GiB, or swap use grows by more
than 512 MiB over the run's baseline. These are hard limits, not targets. Record
hardware, dependency versions, timing synchronization and resource observations.

## Evidence

Store raw JSONL learning curves, summary JSON, initialization/batch/data/source
hashes, mathematical verification results, profiler measurements, failures,
checkpoints, and analysis code. Every figure and conclusion must derive from
these records. Distinguish proposed, implemented, verified, trained, evaluated,
and demonstrated claims.
