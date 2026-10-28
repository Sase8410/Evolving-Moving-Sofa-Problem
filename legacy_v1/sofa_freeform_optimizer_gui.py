"""Step 8: free-form sofas from intersections of transformed hallways.

Run this file from your editor. Requires numpy, shapely and matplotlib.
Standalone: no changes to Steps 2-7. Load a Step 7 JSON with Load route.
Uses a (1+1) evolutionary route search, followed by finer reconstruction
and independent denser checks. These are numerical sampled checks, not
a proof of continuous clearance or of global optimality.
"""

import json
import queue
import threading
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch, Polygon as PolygonPatch
from matplotlib.widgets import Button, Slider
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, box, mapping
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

HALLWAY = unary_union([box(-5, 0, 1, 1), box(0, 0, 1, 5)])
ENTRANCE = box(-5, 0, 0, 1)
EXIT = box(0, 1, 1, 5)
REFERENCE_ROUTE = np.array([
    [-2.5, .5, 0], [.23, .32, 0], [.1779, .5039, 15],
    [.1805, .6548, 30], [.2375, .7625, 45], [.3452, .8195, 60],
    [.4961, .8221, 75], [.68, .77, 90], [.5, 2.5, 90],
])
LIMITATION = (
    "Floating-point sampled geometry only; no continuous-motion certificate "
    "or global optimality claim. Finite hallway arms, fixed endpoint poses, "
    "monotone angles, and bounded waypoint positions restrict this search."
)


def validate_route(route):
    route = np.asarray(route, dtype=float)
    if (route.ndim != 2 or route.shape[1] != 3 or not 3 <= len(route) <= 65
            or not np.isfinite(route).all()):
        raise ValueError("Route needs 3-65 finite [x, y, degrees] waypoints.")
    if (not np.isclose(route[0, 2], 0) or not np.isclose(route[-1, 2], 90)
            or np.any(np.diff(route[:, 2]) < 0)
            or np.max(np.abs(route[:, :2])) > 6):
        raise ValueError("Use positions within +/-6 and angles increasing from 0 to 90.")
    if np.linalg.norm(np.diff(route[:, :2], axis=0), axis=1).sum() > 30:
        raise ValueError("Route is too long for this interactive search.")
    return route.copy()


def refine_waypoints(route):
    """Add degrees of freedom without changing the seed motion."""
    pieces = []
    for a, b in zip(route[:-1], route[1:]):
        n = max(1, int(np.ceil((b[2] - a[2]) / 7.5)))
        pieces.extend(np.linspace(a, b, n, endpoint=False))
    return validate_route(np.vstack([pieces, route[-1]]))


def sample_route(route, distance, angle):
    pieces = []
    for a, b in zip(route[:-1], route[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b[:2]-a[:2]) / distance)),
                int(np.ceil(abs(b[2]-a[2]) / angle)))
        pieces.append(np.linspace(a, b, n, endpoint=False))
    return np.vstack([*pieces, route[-1:]])


def local_hallway(region, pose):
    x, y, angle = pose
    return rotate(translate(region, xoff=-x, yoff=-y), -angle, origin=(0, 0))


def world_sofa(sofa, pose):
    x, y, angle = pose
    return translate(rotate(sofa, angle, origin=(0, 0)), xoff=x, yoff=y)


def polygons(geometry):
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygons(part)


def largest_polygon(geometry):
    return max(polygons(geometry), key=lambda p: p.area, default=Polygon())


def build_sofa(route, distance=.06, angle=3., checkpoint=lambda: None):
    """Largest polygon component of the sampled feasible intersection.

    Retain ALL components during clipping: a smaller intermediate component
    could become the largest survivor. Never use a convex hull or fill holes.
    Full entrance/exit restrictions prevent a partial-turn false solution.
    """
    feasible = local_hallway(ENTRANCE, route[0]).intersection(
        local_hallway(EXIT, route[-1]))
    for pose in sample_route(route, distance, angle):
        checkpoint()
        feasible = feasible.intersection(local_hallway(HALLWAY, pose))
        if feasible.is_empty:
            return Polygon()
    return largest_polygon(feasible)


def verify_sofa(sofa, route, checkpoint=lambda: None):
    poses = sample_route(route, .003, .15)
    maximum, collisions = 0., 0
    for pose in poses:
        checkpoint()
        moved = world_sofa(sofa, pose)
        outside = moved.difference(HALLWAY).area
        maximum = max(maximum, outside)
        collisions += int(not HALLWAY.covers(moved))
    endpoints_fit = bool(ENTRANCE.covers(world_sofa(sofa, route[0]))
                         and EXIT.covers(world_sofa(sofa, route[-1])))
    return {
        "area": float(sofa.area), "samples_checked": len(poses),
        "colliding_samples": collisions, "max_outside_area": float(maximum),
        "endpoints_in_separate_arms": endpoints_fit,
        "sampled_feasible": bool(not sofa.is_empty and sofa.is_valid
                                 and collisions == 0 and endpoints_fit),
        "max_distance": .003, "max_angle_degrees": .15,
    }


def clearance_margin(sofa, route):
    """Bound point travel to a nearest reconstruction sample, in float math.

    Rotation displaces a point by at most radius * angle (in radians).
    Half an interval suffices because either endpoint can be used. Eroding
    by this bound reserves room for translation and rotation between poses.
    This numerical safeguard is not an exact-arithmetic certificate.
    """
    if sofa.is_empty:
        return 0.
    radius = np.linalg.norm(np.asarray(sofa.exterior.coords), axis=1).max()
    steps = np.diff(sample_route(route, .006, .3), axis=0)
    motion = np.linalg.norm(steps[:, :2], axis=1) + radius*np.deg2rad(abs(steps[:, 2]))
    return float(.5 * motion.max() + 1e-7)


class Cancelled(Exception):
    pass


def optimize(seed_route, iterations, seed, events, stop):
    """One parent, one mutated offspring, elitist selection; fixed RNG seed."""
    def checkpoint():
        if stop.is_set():
            raise Cancelled()

    try:
        route = refine_waypoints(validate_route(seed_route))
        initial = route.copy()
        rng = np.random.default_rng(seed)
        best = build_sofa(route, checkpoint=checkpoint)
        archive = [(best.area, route.copy())]
        history = [float(best.area)]
        events.put(("progress", (0, route.copy(), best)))
        for i in range(1, iterations + 1):
            checkpoint()
            child = route.copy()
            # Mutate one waypoint and sometimes a neighbor for smooth changes.
            j = int(rng.integers(1, len(route) - 1))
            scale = (.07 * (1 - i / iterations) + .008)
            delta = rng.normal(0, scale, 2)
            child[j, :2] += delta
            if j + 1 < len(route) - 1 and rng.random() < .4:
                child[j + 1, :2] += .5 * delta
            child[1:-1, :2] = np.clip(child[1:-1, :2],
                initial[1:-1, :2] - .6, initial[1:-1, :2] + .6)
            candidate = build_sofa(child, checkpoint=checkpoint)
            if candidate.area > best.area:
                route, best = child, candidate
                archive.append((best.area, route.copy()))
                archive = sorted(archive, key=lambda item: -item[0])[:5]
            history.append(float(best.area))
            if i % 5 == 0 or i == iterations:
                events.put(("progress", (i, route.copy(), best)))

        events.put(("status", "Rebuilding finalists and checking finer motion..."))
        finalists = []
        attempts = []
        for coarse_area, candidate_route in archive:
            # This is a NEW, smaller geometry; coarse area is never reported
            # as the verified area of the reconstructed polygon.
            sofa = build_sofa(candidate_route, .006, .3, checkpoint)
            margin = clearance_margin(sofa, candidate_route)
            sofa = largest_polygon(sofa.buffer(-margin))
            report = verify_sofa(sofa, candidate_route, checkpoint)
            report["inward_margin"] = margin
            attempts.append({"search_area": float(coarse_area), **report})
            finalists.append((report["sampled_feasible"], sofa.area,
                              sofa, candidate_route, report, coarse_area))
        _, _, sofa, route, report, coarse_area = max(
            finalists, key=lambda item: (item[0], item[1]))
        result = {
            "step": 8, "method": "sampled hallway intersection; (1+1) evolution",
            "shape": "freeform_polygon", "geometry": mapping(sofa),
            "vertices": list(sofa.exterior.coords),
            "interior_rings": [list(r.coords) for r in sofa.interiors],
            "route_waypoints": route.tolist(), "seed_route": initial.tolist(),
            "search_area": float(coarse_area), "verification_report": report,
            "finalist_checks": attempts, "history": history,
            "settings": {"iterations": iterations, "seed": seed,
                         "xy_offset_bound": .6, "search_distance": .06,
                         "search_angle": 3., "rebuild_distance": .006,
                         "rebuild_angle": .3,
                         "margin_rule": "half maximum translation + radius*rotation; plus 1e-7"},
            "limitation": LIMITATION,
        }
        events.put(("done", (result, sofa, route)))
    except Cancelled:
        events.put(("status", "Stopped. Run Search to start a new run."))
    except Exception as error:
        events.put(("status", f"Search failed: {error}"))
    finally:
        events.put(("finished", None))


def polygon_path(sofa):
    """Compound path preserves holes in both local and animated views."""
    if sofa.is_empty:
        return PlotPath(np.zeros((1, 2)), [PlotPath.MOVETO])
    sofa = orient(sofa, sign=1)
    vertices, codes = [], []
    for ring in [sofa.exterior, *sofa.interiors]:
        coords = np.asarray(ring.coords)
        vertices.extend(coords)
        codes.extend([PlotPath.MOVETO] + [PlotPath.LINETO] * (len(coords)-2)
                     + [PlotPath.CLOSEPOLY])
    return PlotPath(vertices, codes)


class FreeformWindow:
    def __init__(self):
        self.seed_route = REFERENCE_ROUTE.copy()
        self.events, self.stop = queue.Queue(), threading.Event()
        self.worker = None
        self.result = None
        self.sofa = Polygon()
        self.poses = None
        self.index, self.playing = 0, False
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("Moving Sofa - Step 8 Free-form Search")
        self.ax = self.fig.add_axes([.06, .19, .46, .73])
        self.local = self.fig.add_axes([.59, .55, .36, .32])
        self.ax.add_patch(PolygonPatch(np.asarray(HALLWAY.exterior.coords),
            facecolor="#f1f5f9", edgecolor="#334155"))
        self.ax.set(xlim=(-5.3, 1.5), ylim=(-.6, 5.3), title="Rigid sofa and route")
        self.ax.set_aspect("equal")
        self.local.set_title("Free-form boundary")
        self.local.set_aspect("equal")
        self.line, = self.ax.plot([], [], "--", color="#2563eb", lw=1)
        self.patch = PathPatch(polygon_path(self.sofa), fc="#3b82f6", alpha=.8)
        self.outline = PathPatch(polygon_path(self.sofa), fc="#3b82f6", alpha=.8)
        self.ax.add_patch(self.patch)
        self.local.add_patch(self.outline)
        self.metrics = self.fig.text(.59, .50, "Load a saved route or use the default.", va="top")
        self.status = self.fig.text(.06, .12, "Ready. Search runs in the background.", fontsize=10)
        self.fig.text(.06, .055, "Sampled geometry only; continuous clearance and optimality are unproven.", fontsize=9)
        self.iterations = Slider(self.fig.add_axes([.68, .27, .25, .03]),
                                 "Iterations", 50, 1000, valinit=250, valstep=50)
        self.buttons = []
        for i, (label, callback) in enumerate([
            ("Load route", self.load), ("Search", self.start), ("Stop", self.cancel),
            ("Play / pause", self.play), ("Save", self.save),
        ]):
            button = Button(self.fig.add_axes([.59 + (i % 3)*.125,
                                              .19 - (i // 3)*.06, .115, .045]), label)
            button.on_clicked(callback)
            self.buttons.append(button)
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", self.close)

    def busy(self):
        return self.worker is not None

    def load(self, _=None):
        if self.busy():
            return
        root = None
        try:
            from tkinter import Tk, filedialog
            root = Tk()
            root.withdraw()
            name = filedialog.askopenfilename(title="Select a Step 7 or Step 8 saved result",
                                              filetypes=[("JSON result", "*.json")])
            if name:
                with open(name, encoding="utf-8") as file:
                    data = json.load(file)
                self.seed_route = validate_route(data["route_waypoints"])
                self.status.set_text(f"Loaded seed route: {Path(name).name}. Press Search.")
        except Exception as error:
            self.status.set_text(f"Could not load route: {error}")
        finally:
            if root is not None:
                root.destroy()
            self.fig.canvas.draw_idle()

    def start(self, _=None):
        if self.busy():
            return
        self.result, self.poses, self.playing = None, None, False
        self.stop.clear()
        self.status.set_text("Searching free-form sofas...")
        self.worker = threading.Thread(target=optimize, daemon=True, args=(
            self.seed_route.copy(), int(self.iterations.val), 42, self.events, self.stop))
        self.worker.start()

    def cancel(self, _=None):
        if self.busy():
            self.stop.set()
            self.status.set_text("Stopping...")

    def play(self, _=None):
        if self.poses is not None and not self.busy():
            self.playing = not self.playing

    def show(self, sofa, route):
        self.sofa = sofa
        self.line.set_data(route[:, 0], route[:, 1])
        self.outline.set_path(polygon_path(sofa))
        if not sofa.is_empty:
            a, b, c, d = sofa.bounds
            self.local.set(xlim=(a-.1, c+.1), ylim=(b-.1, d+.1))
        self.patch.set_path(polygon_path(world_sofa(sofa, route[0])))

    def tick(self):
        changed = False
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "progress":
                i, route, sofa = data
                self.show(sofa, route)
                self.metrics.set_text(f"Iteration: {i}\nSearch area: {sofa.area:.6f}\n"
                                      "Awaiting finer reconstruction and checks.")
            elif kind == "done":
                self.result, sofa, route = data
                self.show(sofa, route)
                self.poses = sample_route(route, .035, 1.5)
                self.index = 0
                r = self.result["verification_report"]
                passed = r["sampled_feasible"]
                self.patch.set_facecolor("#22c55e" if passed else "#ef4444")
                self.metrics.set_text(f"Reconstructed area: {r['area']:.6f}\n"
                    f"Boundary vertices: {max(0, len(sofa.exterior.coords)-1)}\n"
                    f"Check samples: {r['samples_checked']}\n"
                    f"Collisions: {r['colliding_samples']}\n"
                    f"Sampled check: {'PASS' if passed else 'FAIL'}")
                self.status.set_text("Finished. Play the motion or Save the result.")
            elif kind == "status":
                self.status.set_text(data)
            elif kind == "finished":
                self.worker = None
        if self.playing and self.poses is not None:
            self.patch.set_path(polygon_path(world_sofa(self.sofa, self.poses[self.index])))
            self.index = (self.index + 1) % len(self.poses)
            changed = True
        if changed:
            self.fig.canvas.draw_idle()

    def save(self, _=None):
        if self.result is None or self.busy():
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(__file__).resolve().with_name(f"freeform_sofa_{stamp}.json")
        try:
            with path.open("x", encoding="utf-8") as file:
                json.dump(self.result, file, indent=2, allow_nan=False)
                file.write("\n")
            self.status.set_text(f"Saved {path.name} beside this Python file.")
        except (OSError, ValueError) as error:
            self.status.set_text(f"Could not save: {error}")
        self.fig.canvas.draw_idle()

    def close(self, _=None):
        self.stop.set()
        self.timer.stop()


def main():
    window = FreeformWindow()
    plt.show()
    return window


if __name__ == "__main__":
    main()
