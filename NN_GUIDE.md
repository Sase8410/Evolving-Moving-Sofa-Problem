# Neural guidance in V3.2

Use **Search + NN** in the GUI. It collects real geometry evaluations, fits an
ensemble of three small MLPs, and ranks candidate motions. No GPU or separate
training command is required. **Search V3** remains the default baseline.

Guidance starts only when held-out prediction error beats a constant predictor
and pairwise ranking accuracy among stronger held-out candidates reaches 60%.
Some short runs correctly never pass this gate. The holdout is also used for
model selection, so it is not an independent test set.

Ordinary proposals remain available during guidance. If their recent mean
accepted improvement is at least as good as guided proposals, the next 64
proposals use ordinary evolution. This is an operational heuristic rather than
a causal performance estimate.

## Reusing learning

Leave **Reuse saved NN learning** checked to reuse a compatible dataset and
numeric checkpoint. Compatibility includes source hashes, sample count and
reconstruction tolerance. V3.1 checkpoints are preserved in their old folders;
they are not mixed with this release's changed code. Uncheck reuse for a new
folder. Never delete another process's active.lock while it is running.

Passing finalists are saved separately in `validated_training.json`. On later
runs, a small ridge model can learn the gap between sampled and validated area.
It needs at least 32 unique validated examples and a useful holdout result.
Until then, this correction is disabled. No predicted score establishes that
a sofa fits the hallway.

## Comparing fairly

**Compare NN / V3** starts fresh learning for each paired seed. **Final study**
adds a common runtime cap, tolerance sweep and independent audit. Neither
cold-start experiment measures the benefit of a previously warmed-up loss
model. See RELEASE_RESULTS.md for actual measurements and README.md for paths,
commands and limitations.

Inspect `neural` in result JSON for fits, guided evaluations, ranking accuracy,
fallback counts and validation-loss-model readiness. `training.jsonl` contains
fit diagnostics; `evaluations.jsonl` contains real sampled-area training rows.
The final `validated_area` is produced by geometry reconstruction and checking.
