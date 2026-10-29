# Moving Sofa V3.2 — release measurements

## Result and release decision

The best rebuilt example has validated area **2.215251416**. It passed bounded numerical verification and an independent shifted-grid audit of **6,009 poses**, with zero colliding samples, zero measured outside area, continuous segments and valid entrance/exit occupancy.

This is a floating-point numerical result, not a formal certificate or a global-optimality claim. The example was selected from the experiments and then reconstructed at tighter tolerances. Its larger area is **not** an NN speedup or an additional optimization run.

**V3 remains the default.** Neural guidance improved mean area at the fixed evaluation budget, but it did not establish an advantage under the common runtime cap. The learned reconstruction-loss correction remains experimental; these cold-start benchmarks do not measure a warmed-up correction model.

## Paired experiments

Each study used seeds 100–129 and fresh NN learning per seed. Method order alternated. All 120 runs passed their bounded numerical checks. Controls started at 5, the sampled objective used 49 points, tolerance was 0.004, local/adaptive refinement was enabled, and one finalist received full reconstruction/validation after diverse pool promotion.

| Study / metric | V3 | V3 + NN |
|---|---:|---:|
| 600 evaluations: Passes | 30 | 30 |
| 600 evaluations: Mean validated area | 2.138711 | 2.159871 |
| 600 evaluations: Best validated area | 2.195402 | 2.193589 |
| 600 evaluations: Mean total seconds | 7.362 | 10.609 |
| 600 evaluations: Mean objective evaluations | 600.0 | 600.0 |
| 12-second total cap: Passes | 30 | 30 |
| 12-second total cap: Mean validated area | 2.186622 | 2.179331 |
| 12-second total cap: Best validated area | 2.198699 | 2.196462 |
| 12-second total cap: Mean total seconds | 8.158 | 8.171 |
| 12-second total cap: Mean objective evaluations | 1188.3 | 886.3 |

**Fixed evaluation budget:** NN won 21 of 30 pairs. Mean paired NN−V3 area difference: +0.021160; paired bootstrap 95% interval [+0.000647, +0.039981].

**Common runtime cap:** NN won 14 of 30 pairs. Mean paired NN−V3 area difference: -0.007291; paired bootstrap 95% interval [-0.017538, +0.001923].

The evaluation-budget mean-area increase was about 0.99%, with higher training/runtime cost. This is evidence for that short-budget setting, not a universal performance guarantee. The runtime-cap interval includes zero; the lower NN mean does not establish a statistically clear loss either.

The runtime study allocated 55% of the 12-second cap to search and reserved the rest for selection and validation. Mean total times were about 8.2 seconds, so these are **common budget caps**, not runs forced to last exactly 12 seconds. No finalization timeouts or numerical-work cap overruns were observed. Final output serialization is outside the reported numerical-work clock.

The evaluation study resumed after interruption with identical source hashes and package versions. It spans two sessions, so its runtime averages are environment/session-specific. The timed study ran in one continuation session. The budgets and tolerance differ from your earlier 1,500-evaluation V3.1 benchmark; do not interpret the absolute area difference as a version regression or gain.

## Fixed-motion reconstruction convergence

Only reconstruction tolerance changed below. The motion was held fixed.

| Displacement target | Validated area | Gain from previous |
|---:|---:|---:|
| 0.0040 | 2.198699101 | — |
| 0.0020 | 2.208463664 | +0.009764562 |
| 0.0010 | 2.212517115 | +0.004053451 |
| 0.0005 | 2.215251416 | +0.002734301 |

The last increment remains about 0.00273 area units. This sweep demonstrates reduced conservative reconstruction loss, not complete numerical convergence or a rigorous error estimate. Further refinement is optional research.

## Verification and provenance

- Regression suite: 34 tests; captured output is in `verification/test_output.txt`.
- Desktop interface: rendered and visually inspected; old and new result-loading paths exercised.
- Windows launchers are provided; native Windows desktop interaction was not executed in the Linux test environment.
- Numerical source hashes, settings and dependency versions are included in both study manifests.
- After measurements, only the package `__version__` declaration was corrected from 3.1.0-neural-experimental to 3.2.0. All other measured source hashes match the release. See `verification/release_version_metadata.json` and `release_source_check.json`. Use a new output folder for a new benchmark; the completed original manifests remain unchanged.

## Reproduce this study

```bash
python run_v3.py final-study --budget 600 --seed 100 --seeds 30 --target .004 --finalists 1 --seconds 12 --output results/reproduced_study
```

For a longer experiment, increase the budget and runtime cap in a fresh folder. The GUI uses 1,500 evaluations and three finalists by default; this release study deliberately used the smaller budget listed above.

Full records: `results/v32_release_study/final_summary.json`, per-method result JSON, `runs.csv`, comparison figures, `convergence.json`, `convergence_best.json`, and `audit.json`. The demo animates the exact geometry/motion in `convergence_best.json`.
