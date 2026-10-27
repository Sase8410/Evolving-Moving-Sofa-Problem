# Moving Sofa V3.1: neural-guided search

V3 optimizes independent continuous translation and rotation splines, reconstructs
freeform geometry from inverse hallway intersections, and checks the resulting
rigid motion with conservative displacement bounds. This update extends your V3 directly; V2 is not required. The optional V1
comparison files remain unchanged in `legacy_v1/`.

## Start the GUI

Install Python 3.10 or newer, extract the entire ZIP, and install the dependencies:

```bash
python -m pip install -r requirements.txt
python run_v3.py
```

You can also open `run_v3.py` in VS Code and click **Run Python File**. No terminal
input is required after launch. Keep `sofa_v3/` beside the launcher. Click **Train + search NN** to collect
simulation data, train the networks, and guide the search automatically. No
separate training dataset or GPU is needed. **Search V3 baseline** runs ordinary
V3 with learning disabled. **Compare NN / V3** runs a fresh paired experiment
with an editable seed count (30 by default). The budget and seed are editable.
The window also provides Hammersley check, Stop, Play/pause, Save result and
Load result. Search runs in a background thread; Stop interrupts training,
reconstruction and verification. See [NN_GUIDE.md](NN_GUIDE.md) for the full workflow.

NN searches automatically save their result, candidate journal, training
diagnostics and numeric model checkpoint inside `results/nn_learning/`.
Leave **Reuse saved NN learning** checked to reuse compatible data and weights
next time; uncheck it to start fresh. This reuses learned rankings, not the
previous optimizer population. Baseline searches still use **Save result**.
Your existing V3 JSON results load without conversion. `legacy_v1/` is needed
only for the optional old V1 benchmark, not NN search or NN/V3 comparison.

If your Python installation lacks a desktop GUI backend, install Tk support
for that Python installation. The original GUIs remain available in `legacy_v1/`.

## What changed

| Module | Responsibility |
|---|---|
| `motion.py` | Natural cubic splines for x, y and angle; exact polynomial derivative extrema in floating point; knot insertion |
| `geometry.py` | Inverse hallway intersections, retained concavities/holes, adaptive interval resolution and conservative reconstruction |
| `search.py` | Differential evolution, bounded Powell refinement, contact-guided knot insertion, NN-guided candidate evaluation |
| `neural.py` | Three trained MLPs, held-out error gate, exploration, persistent training data and safe numeric model checkpoints |
| `verify.py` | Forward geometry checks, independent dense sample grid and recursive clearance bounds |
| `baseline.py` | Reproduce the attached Step 8 candidate search, then apply the same final validator as V3 |
| `experiments.py` | Paired seeds, equal objective budgets, resumption, ablations, CSV/JSON summaries and progress plots |
| `reference.py` | Analytic Hammersley motion and shape approximation, for reference testing only |
| `gui.py` | Interactive search, playback and exports |

The attached Step 8 implementation already used inverse hallway intersections.
The redesign expands the motion representation and optimizer, and strengthens
validation; it does not claim inverse geometry itself is new.

## Search and geometry

The default discovery seed is a simple rotation at a fixed center. Neither
Hammersley's motion nor Gerver's shape is injected into discovery runs. Each
turn starts at angle 0 and ends at pi/2. Interior angles vary independently and
can reverse; symmetry and monotonic rotation are not imposed. Straight entrance
and exit segments complete the journey between separate hallway arms.

Search starts with five spline controls. Differential evolution mixes broad
random starts and candidates near the seed. Powell refinement improves the
best result. One additional knot is placed in an interval selected by wall-cut
probes and curvature, then refined with the remaining budget. This is a
contact-guided heuristic, not a solver for Gerver's analytic contact equations.
The optional surrogate ranks proposals; predictions never substitute for real
geometry evaluations or final verification.

Hallway intersections retain every component until clipping finishes, then
select the largest connected polygon. No convex hull replaces a sofa, and holes
are not filled. Boundary resolution grows as more wall constraints are added.
Coarse **search area** is only an objective value: it can exceed the true optimum
because intermediate constraints are missing. Only **validated area** should be
used in performance statements.

## What validation means

For a point at radius at most R from the local origin, its speed is bounded by

`sqrt(max|x'|² + max|y'|²) + R max|angle'|`.

For each interval of duration h, its displacement from the midpoint is at most
`h * speed_bound / 2`. Reconstruction intersects the inverse hallway inset by
that displacement plus a numerical margin. The inset uses straight walls and a
conservative square exclusion at the concave corner, avoiding inscribed circular
buffer approximations. Refinement continues until the displacement bound is
below the requested target. A smaller target usually retains more area and costs
more time. Exact straight translations within a common convex arm need only
endpoint containment; this avoids unnecessary interval subdivision.

The final checker separately transforms the sofa forward and measures its
distance to the original hallway boundary. It recursively proves each interval
clear under the same motion-speed inequality, also checking a separate regular
pose grid. The forward checker does not use the reconstruction inset. The two
paths share the motion-bound routine, which is covered by derivative tests.
Endpoint occupancy, connectedness and polygon validity are checked. A recursion
limit produces `inconclusive`, never a pass. Failed results have
`validated_area: null`.

`bounded_numerical_pass` is a conservative **floating-point numerical check**,
not an exact-arithmetic or formally verified certificate. Shapely/GEOS and SciPy
use floating-point arithmetic. In addition, search has finite controls, bounded
translations, a local shape window x in [-3,3], y in [-2,1], and hallway arms of
length 8. These are explicit computational restrictions; there is no global
convergence guarantee.

## Reproduce experiments

```bash
# One discovery run, with no reference shape supplied
python run_v3.py search --budget 4000 --seed 42 --target .001

# Reproduce an ablated search directly
python run_v3.py search --budget 1500 --fixed-angles
python run_v3.py search --budget 1500 --no-adaptive

# Known reference: checks representational and validation accuracy, not discovery
python run_v3.py reference --target .0005

# Paired V1/V3 comparison; each seed gets the same search-evaluation budget
python run_v3.py benchmark --seeds 30 --budget 1500 --output results/full_comparison

# Also disable local refinement, adaptive controls, or angle optimization
python run_v3.py benchmark --seeds 30 --budget 1500 --ablations --output results/ablations

# NN search with persistent learning (scikit-learn is included in requirements)
python run_v3.py search --budget 1500 --surrogate

# A genuinely fresh neural-versus-V3 comparison; includes training runtime
python run_v3.py compare-nn --seeds 30 --budget 300 --finalists 1 --target .003 --output results/my_nn_comparison

# New learning dataset/model, independent of previously saved learning
python run_v3.py search --budget 1500 --surrogate --fresh-nn

python -m unittest discover -s tests -v

# Separate shifted-grid audit of the saved high-area result
python audit_export.py results/discovery_4000_seed42.json
```

The benchmark counts the initial objective evaluation and every candidate
evaluation. Neural predictions do not count because they perform no geometry
evaluation. Finalist rebuilding/verification and the adaptive contact probe are
separate overhead; search and total runtimes are both reported. Equal evaluation
counts do **not** mean equal computational cost: the representations and number
of hallway intersections differ. Use runtime as a second comparison axis.

The V1 comparator is specifically the attached
`sofa_freeform_optimizer_gui.py` (Step 8), not an unavailable later version or a
historical scalar score. Its random sequence, acceptance rule and candidate
areas are tested against the original. Both methods use common conservative
final reconstruction and verification, so V1's final metric differs from its
original sampled-check output. The attachment files themselves are unmodified.

Each experiment directory has a settings/environment/source-hash manifest,
per-seed JSON including full progress curves and motions, `runs.csv`,
`summary.json` and `comparison.png`. A rerun resumes completed records only if
the settings and source hashes match. Choose a new directory after changing
the code. Alternating method order reduces systematic execution-order bias.
Summaries report feasibility, best/mean/median area, sample standard deviation,
runtimes and a paired bootstrap interval. Failed runs are excluded from the
conditional area mean and included as zero in a separately labeled mean.

## Mathematical target and attribution

The goal is independent numerical reproduction toward Gerver, not a claim to
exceed the standard moving-sofa optimum. Baek's paper was submitted to arXiv on
29 November 2024 and states the proof of Gerver's optimality. An arXiv DOI is
not itself evidence of journal publication.

- Jineon Baek, *Optimality of Gerver's Sofa*: https://arxiv.org/abs/2411.19826
- Kuangdai Leng et al., *Deep Learning Evidence for Global Optimality of Gerver's
  Sofa*: https://arxiv.org/abs/2407.11106
- Dan Romik, constructions and background: https://danromik.com/pages/sofas.html

The neural-network paper motivates independent time-dependent translation and
rotation. V3 uses our own SciPy spline, polygon clipping and optimizer
implementation; it is not a reproduction of that paper's neural training or
differentiable waterfall algorithm. The Hammersley reference has analytic area
`pi/2 + 2/pi = 2.207416099...`; Gerver's reference is approximately
`2.2195316688`. Neither reference is evidence that discovery reached that area.
The historical 1.53945 baseline and the claimed V2 0.92% improvement were not
supplied as result artifacts and are not treated as verified measurements here.

See `NN_RESULTS.md` for the measurements completed for V3.1.
`RESULTS.md` and the original result directories preserve the previous V3
measurements; those results did not use this neural implementation.
