# AI Moving Sofa

A desktop Python application for finding a route for a rigid polygon and for
discovering new sofa shapes together with their movements through a unit-width
L-shaped hallway. All normal interaction happens in one GUI.

## Start

Keep all four application `.py` files in the **same folder**. Python 3.11 or
newer is recommended; development checks used Python 3.12.

Install dependencies once in the Python environment selected in VS Code:

```bash
python -m pip install -r requirements.txt
```

Open `sofa_optimizer_gui.py` and click **Run Python File**. Alternatively:

```bash
python sofa_optimizer_gui.py
```

There are no terminal input prompts. Use a desktop Matplotlib backend; an inline
notebook backend does not provide the same interactive window. Standard Windows
Python includes Tk for the Load/Save dialogs.

## What you can do

### Find a movement for your shape

1. Choose a preset and click **Apply preset**, or click **Draw polygon**.
2. When drawing, click boundary vertices in order in the shape editor. Right-click
   or click **Finish polygon** to close it. Backspace removes the last point;
   Escape cancels. Self-intersecting outlines are rejected.
3. Drag an existing boundary vertex to edit it. **Notch / extrude** opens controls
   for boundary position, width, and depth. **Apply edit** commits the preview;
   repeated edits can build more complex shapes. **Undo shape** restores an edit.
4. Select **Find path**, choose a budget/seed, and click **Run**.
5. **Play / pause** animates the result; **Motion** scrubs through it.

The polygon remains rigid and unchanged during path search. The planner searches
translations and rotations, including backward movement and reversed rotation.
It tries multiple entrance/exit orientations and uses an imported/current route
as a warm start when that route passes the current checker. It never shrinks a
fixed input to make it pass.

The planner uses a custom multi-root bidirectional RRT-Connect-style search,
followed by collision-checked shortcuts. It compares found routes first by a
conservative lower bound on clearance, then by translation plus radius-weighted
rotation. This is a finite heuristic search, **not a globally optimal planner**.
Clearance lower bounds can be conservative. A failure means no route was found
within the selected search, not that every possible route has been ruled out.

### Discover a sofa and its movement

1. Select **Discover sofa** and click **Run**. This starts from generic and random
   movements, independently of the displayed shape.
2. Watch the candidate boundary and estimated area change.
3. Wait for final reconstruction and motion checks. **Seconds** limits the search;
   reconstruction/checking happens afterward and can take additional time.
4. **Continue** starts another discovery run using the current movement as a warm
   start. The previous movement is included among the finalists. Imported routes
   must have the supported 0-to-90-degree entrance/exit structure; other routes
   remain usable by Find path.
5. **Refine area** reconstructs an already discovered result using a finer motion
   spacing and smaller inward margin, then rechecks it. It retains the previous
   result if refinement fails or produces less area. Each click reduces the
   spacing by a factor of four, down to 0.000125 hallway units.

Discovery optimizes reference-point positions and rotation control points.
Cubic splines propose smooth movements; the actual exported/executed movement
is a dense piecewise-linear route. Interior angles may reverse. The default
search progressively uses 5, 9, then 17 controls, combines population exploration
with local Powell optimization from multiple starts, and refines the sampling.
The shape is the largest connected polygonal component of the intersection of
transformed hallways, with separate full-entrance and full-exit constraints.
All components are retained until the final intersection is complete.

There are **no Gerver coordinates, Gerver motion tables, pretrained models,
target-shape rewards, or artificial area caps in the discovery engine**. Gerver's
approximate area is displayed only as a reference in the GUI. The optimizer is
an evolutionary/numerical search, not a trained neural network.

The search remains bounded: finite control counts, bounded reference positions
and intermediate angles, and fixed endpoint rotations of 0 and 90 degrees.
It does not search the entire mathematical space of possible motions. Different
seeds and larger budgets can help, but exact reproduction of Gerver is not
guaranteed. A time-limited run is not bitwise reproducible across machines even
with the same random seed.

## How collision checks work

- A pose transforms the entire polygon, including holes; vertex-only checks are
  not used. Outside area alone is not treated as proof of containment.
- A full passage requires the entire shape to begin in the entrance arm and end
  in the exit arm. A partial turn is not a success.
- For an interval, point displacement is bounded by translation plus the maximum
  vertex radius times rotation in radians. Endpoint clearance can therefore
  establish containment between frames. Otherwise the checker subdivides the
  interval and tests its midpoint.
- Pure horizontal/vertical translations can be checked from their endpoint
  shapes because every corresponding cross section of this L hallway is
  connected. Other pure translations can use swept polygon geometry.
- Near-contact intervals that cannot be resolved within the configured depth
  and resolution limits are explicitly reported; they are not labeled passed.
- For **discovery only**, fine reconstruction creates a new, slightly smaller
  polygon using an inward offset based on the maximum between-sample motion.
  The JSON separately records raw reconstruction area, inward margin, final
  area, collisions, and unresolved intervals. Fixed-shape planning never applies
  this offset to the input shape.
- Discovery removes redundant near-collinear boundary vertices at a tolerance
  of 1e-6 hallway units, reserves an additional 1e-5 margin, and checks the
  cleaned polygon again before reporting it.

**Motion bounds passed** means these numerical checks passed. Shapely/GEOS and
all arithmetic here are floating point, so it is not an exact-arithmetic proof.
The hallway width is always 1. Finite arms are lengthened when necessary to hold
an input shape; this does not change the corridor width or rescale the shape.

## Save, load, and inspect

**Save JSON** stores polygon geometry, holes, the route, settings, and available
verification information. It can also save an unverified custom shape.

**Load JSON** supports the earlier project exports containing `vertices` and
`route_waypoints`, full `geometry` objects, standalone GeoJSON Polygons, and a
plain list of polygon vertices. Previous PASS labels are never trusted after
loading: click **Check route** to verify with the current checker. For a saved
discovery, that check also re-enables **Refine area**.

`example_discovery.json` is an actual result produced by the generic search
during development, with its motion and audit data. It is optional and is never
automatically used to initialize a new discovery run. Load it to inspect it,
or leave it unused when testing independent discovery.

## Files

| File | Responsibility |
| --- | --- |
| `sofa_optimizer_gui.py` | The single desktop entry point, drawing/editing, background jobs, playback, and file dialogs |
| `sofa_geometry.py` | The single source for geometry, collision checks, reconstruction, and JSON I/O |
| `sofa_path_planner.py` | Fixed-shape motion search and route refinement |
| `sofa_discovery.py` | Automatic shape-and-motion discovery |
| `requirements.txt` | Runtime dependencies |
| `test_sofa.py` | Geometry, search, serialization, and GUI regression checks |
| `example_discovery.json` | Optional independently discovered result |
| `README.md` | Setup, usage, algorithms, and limitations |

## Validation

The included example came from a generic-start search with seed 42 and a
35-second search budget, followed by reconstruction at motion step 0.0005.

| Measured result | Value |
| --- | ---: |
| Final checked polygon area | 2.2170300746 |
| Raw fine reconstruction area, before inward offset | 2.2193567014 |
| Inward margin | 0.0002599706 |
| Poses examined by the adaptive checker | 14,991 |
| Colliding poses | 0 |
| Unresolved motion intervals | 0 |

This is a near-Gerver numerical result, not an exact reproduction. The fine
refinement took about 124 seconds in the development environment, in addition
to the original search/checks. Higher accuracy increases computation time.

Run `test_sofa.py` from your editor, or:

```bash
python -m unittest -v test_sofa
```

All 16 regression tests passed. They cover tunneling between valid poses, rotational collisions,
exact wall-contact translations, full entrance/exit completion, invalid shapes,
disconnected edits, holes, legacy JSON, backward angles, independent dense
motion checks, unchanged fixed-shape geometry, search failure, cancellation,
and GUI state handling. The GUI was rendered and exercised headlessly in the
development environment; Windows-native dialog behavior still depends on the
local Python/Tk installation.

## Research context

Reproducing a Gerver-like sofa numerically is the research target. Under the
standard unit-width right-angle problem, Baek's optimality result identifies
Gerver's area as the maximum. A numerical score above it should prompt an audit
of discretization and collision checks, not a claim of a new record.

- [Baek, Optimality of Gerver's Sofa](https://arxiv.org/abs/2411.19826)
- [Leng et al., Deep Learning Evidence for Global Optimality of Gerver's Sofa](https://arxiv.org/abs/2407.11106)
- [OMPL RRT-Connect documentation](https://ompl.kavrakilab.org/classompl_1_1geometric_1_1RRTConnect.html)
- [SciPy Powell optimization](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-powell.html)

The implementation here is a smaller geometry-based optimizer; it is not a
reimplementation of the neural-network paper.
