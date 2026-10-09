"""Step 5: run genetic sofa optimization entirely from a window.

Keep sofa_fitness.py, sofa_shape_simulation.py, and sofa_path_simulation.py
in this folder. Open this file in VS Code and click Run Python File.
The earlier sofa_genetic_algorithm.py is not required by this version.

Save result writes best_sofa_gui.json beside this script, replacing that file.
The route, shape family, and bounded dimensions limit this search. Sampling
does not prove continuous clearance or global optimality.
"""

import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox

from sofa_fitness import evaluate_sofa
from sofa_path_simulation import TURNING_ROUTE, sample_path, animate_path
from sofa_shape_simulation import create_candidate, check_path


# Width, height, and notch depth / height. Rectangles use only two genes.
BOUNDS = np.array([[0.20, 2.40], [0.15, 1.00], [0.02, 0.95]])
SEARCH_DETAIL = {"Standard": (0.04, 2.0), "Fine": (0.01, 0.5)}
OUTPUT_PATH = Path(__file__).resolve().with_name("best_sofa_gui.json")


def decode(genes):
    width, height = map(float, genes[:2])
    return {
        "width": width,
        "height": height,
        "notch_depth": height * float(genes[2]) if len(genes) == 3 else 0.0,
    }


def make_sofa(genes, settings):
    return create_candidate(settings["shape"], **decode(genes),
                            curve_points=settings["curve_points"])


def run_search(settings, messages, stop):
    """Background worker: only geometry and evolution, never window updates."""
    try:
        rng = np.random.default_rng(settings["seed"])
        count = settings["population"]
        dimensions = 2 if settings["shape"] == "rectangle" else 3
        lower, upper = BOUNDS[:dimensions].T
        poses = sample_path(TURNING_ROUTE, settings["max_distance"],
                            settings["max_angle"])
        population = rng.uniform(lower, upper, (count, dimensions))
        population[0] = np.array([1.5, 0.6, 0.25 / 0.6])[:dimensions]
        population[1] = np.array([0.4, 0.3, 0.25])[:dimensions]
        cache, history = {}, []
        best_genes, best_report = None, None

        for generation in range(settings["generations"] + 1):
            reports = []
            for genes in population:
                if stop.is_set():
                    break
                key = tuple(genes)
                if key not in cache:
                    cache[key] = evaluate_sofa(make_sofa(genes, settings), poses)
                report = cache[key]
                reports.append(report)
                if best_report is None or report["fitness"] > best_report["fitness"]:
                    best_genes, best_report = genes.copy(), report

            if not reports:
                break
            scores = np.array([report["fitness"] for report in reports])
            feasible = [r["area"] for r in reports if r["sampled_feasible"]]
            history.append({
                "generation": generation,
                "evaluated_candidates": len(reports),
                "complete_generation": len(reports) == count,
                "best_fitness": best_report["fitness"],
                "best_area": best_report["area"],
                "best_sampled_feasible": best_report["sampled_feasible"],
                "feasible_count": len(feasible),
            })
            messages.put(("generation", {
                "genes": best_genes.tolist(), "report": best_report,
                "history": history.copy(),
            }))
            if stop.is_set() or generation == settings["generations"]:
                break

            # Preserve two elites, then generate children from tournaments.
            order = np.argsort(-scores, kind="stable")
            population, scores = population[order], scores[order]

            def parent():
                contestants = rng.integers(0, count, 3)
                return population[contestants[np.argmax(scores[contestants])]]

            children = [population[0].copy(), population[1].copy()]
            while len(children) < count:
                a, b = parent(), parent()
                mix = rng.random(dimensions)
                child = mix * a + (1 - mix) * b
                mask = rng.random(dimensions) < settings["mutation_rate"]
                noise = rng.normal(0, settings["mutation_scale"], dimensions)
                child = np.clip(child + mask * noise * (upper - lower), lower, upper)
                children.append(child)
            population = np.array(children)

        if best_genes is None:
            messages.put(("done", None))
            return

        # A stopped search also checks its best evaluated candidate.
        messages.put(("checking", None))
        fine_distance = min(0.005, settings["max_distance"] / 8)
        fine_angle = min(0.25, settings["max_angle"] / 8)
        fine_poses = sample_path(TURNING_ROUTE, fine_distance, fine_angle)
        sofa = make_sofa(best_genes, settings)
        verification = evaluate_sofa(sofa, fine_poses)
        passed = best_report["sampled_feasible"] and verification["sampled_feasible"]
        messages.put(("done", {
            "shape": settings["shape"],
            "parameters": decode(best_genes),
            "genes": best_genes.tolist(),
            "gene_names": ["width", "height", "notch_ratio"][:dimensions],
            "gene_bounds": BOUNDS[:dimensions].tolist(),
            "vertices": np.asarray(sofa.exterior.coords).tolist(),
            "route_waypoints": TURNING_ROUTE.tolist(),
            "settings": settings.copy(),
            "verification_max_distance": fine_distance,
            "verification_max_angle": fine_angle,
            "search_report": best_report,
            "verification_report": verification,
            "passed_both_sampled_checks": bool(passed),
            "stopped_early": (history[-1]["generation"] < settings["generations"]
                              or not history[-1]["complete_generation"]),
            "history": history,
            "limitation": "Fixed route and shape family; sampled checks do not "
                          "prove continuous clearance or global optimality.",
        }))
    except Exception as error:
        messages.put(("error", f"{type(error).__name__}: {error}"))


class SofaEvolutionWindow:
    def __init__(self):
        self.busy = False
        self.closed = False
        self.result = None
        self.messages = Queue()
        self.stop_event = Event()
        self.worker = None
        self.settings = None

        self.fig = plt.figure(figsize=(12.8, 8.8), facecolor="#f8fafc")
        self.fig.canvas.manager.set_window_title("Moving Sofa - Genetic Search")
        self.fig.text(0.06, 0.945, "Evolving Moving Sofa", fontsize=20, weight="bold")
        self.fig.text(0.06, 0.91, "Watch the best design improve on the fixed turning route.",
                      fontsize=11, color="#475569")

        self.shape_ax = self.fig.add_axes([0.07, 0.515, 0.55, 0.335])
        self.shape_ax.set(xlim=(-1.35, 1.35), ylim=(-0.6, 0.65),
                          xlabel="Local x", ylabel="Local y", title="Best sofa")
        self.shape_ax.set_aspect("equal")
        self.shape_ax.grid(alpha=0.2)
        self.patch = PolygonPatch([[0, 0], [0, 0], [0, 0]],
                                  facecolor="#3b82f6", edgecolor="#1e3a8a", alpha=0.8)
        self.shape_ax.add_patch(self.patch)
        self.placeholder = self.shape_ax.text(0.5, 0.5, "Click Start search to begin",
                                              transform=self.shape_ax.transAxes,
                                              ha="center", color="#64748b")
        self.metrics = self.fig.text(0.065, 0.46, "No candidate evaluated yet.",
                                     va="top", fontsize=10)

        self.fitness_ax = self.fig.add_axes([0.075, 0.205, 0.54, 0.17])
        self.fitness_ax.set(xlim=(0, 40), ylim=(0, 1), xlabel="Generation",
                            ylabel="Best fitness", title="Search progress")
        self.fitness_ax.grid(alpha=0.2)
        self.line, = self.fitness_ax.plot([], [], color="#2563eb", lw=2)

        self.fig.text(0.73, 0.92, "SETTINGS FOR NEXT RUN", fontsize=11, weight="bold")
        self.shape = RadioButtons(self.fig.add_axes([0.73, 0.77, 0.23, 0.12]),
                                  ["rectangle", "notched", "curved"], active=2)
        self.population = self.add_slider("Population", 0.70, 8, 100, 40, 4, "%d")
        self.generations = self.add_slider("Generations", 0.62, 5, 100, 40, 5, "%d")
        self.rate = self.add_slider("Mutation probability", 0.54, 0, 1, 0.30, 0.05, "%.2f")
        self.scale = self.add_slider("Mutation strength", 0.46, 0.01, 0.25, 0.08, 0.01, "%.2f")
        self.seed = TextBox(self.fig.add_axes([0.80, 0.395, 0.14, 0.035]), "Seed  ",
                            initial="42")
        self.fig.text(0.73, 0.363, "Search detail", fontsize=10)
        self.detail = RadioButtons(self.fig.add_axes([0.73, 0.27, 0.23, 0.075]),
                                   list(SEARCH_DETAIL))

        self.start_button = Button(self.fig.add_axes([0.73, 0.195, 0.11, 0.045]),
                                   "Start search")
        self.stop_button = Button(self.fig.add_axes([0.85, 0.195, 0.11, 0.045]), "Stop")
        self.replay_button = Button(self.fig.add_axes([0.73, 0.125, 0.11, 0.045]),
                                    "Replay best")
        self.save_button = Button(self.fig.add_axes([0.85, 0.125, 0.11, 0.045]),
                                  "Save result")
        self.start_button.on_clicked(self.start)
        self.stop_button.on_clicked(self.stop)
        self.replay_button.on_clicked(self.replay)
        self.save_button.on_clicked(self.save)
        self.controls = [self.shape, self.population, self.generations,
                         self.rate, self.scale, self.seed, self.detail]
        self.fig.text(0.73, 0.085, "Fine uses more poses and runs slower.\n"
                      "Every result receives a finer final check.", fontsize=8, color="#475569")
        self.status = self.fig.text(0.06, 0.125, "Ready. Choose settings and click Start search.",
                                    va="top", fontsize=10, color="#334155")
        self.fig.text(0.06, 0.025, "Sampled checks only. The route stays fixed; continuous clearance is not proven.",
                      fontsize=9, color="#64748b")

        # Poll worker messages on the GUI thread; do not draw from the worker.
        self.timer = self.fig.canvas.new_timer(interval=60)
        self.timer.add_callback(self.poll)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", self.close)
        self.fig.sofa_window = self  # Retain widgets and timers in IDEs.
        self.update_controls()

    def add_slider(self, label, y, minimum, maximum, initial, step, fmt):
        self.fig.text(0.73, y + 0.035, label, fontsize=10)
        return Slider(self.fig.add_axes([0.73, y, 0.21, 0.023]), "",
                      minimum, maximum, valinit=initial, valstep=step, valfmt=fmt)

    def update_controls(self):
        for widget in self.controls:
            # RadioButtons.set_active(index) selects an option; use the shared
            # active property to enable/disable all widget types consistently.
            widget.active = not self.busy
        states = [(self.start_button, not self.busy), (self.stop_button, self.busy),
                  (self.replay_button, not self.busy and self.result is not None),
                  (self.save_button, not self.busy and self.result is not None)]
        for button, enabled in states:
            button.set_active(enabled)
            button.label.set_color("#0f172a" if enabled else "#94a3b8")
            button.ax.set_facecolor("#e2e8f0" if enabled else "#f1f5f9")

    def start(self, _=None):
        if self.busy or self.closed:
            return
        try:
            seed = int(self.seed.text)
            if seed < 0:
                raise ValueError
        except ValueError:
            self.status.set_text("Seed must be a nonnegative whole number.")
            self.fig.canvas.draw_idle()
            return
        distance, angle = SEARCH_DETAIL[self.detail.value_selected]
        self.settings = {
            "shape": self.shape.value_selected,
            "population": int(self.population.val),
            "generations": int(self.generations.val),
            "mutation_rate": float(self.rate.val),
            "mutation_scale": float(self.scale.val), "seed": seed,
            "max_distance": distance, "max_angle": angle, "curve_points": 60,
        }
        self.busy, self.result = True, None
        self.messages, self.stop_event = Queue(), Event()
        self.patch.set_visible(False)
        self.placeholder.set_visible(True)
        self.placeholder.set_text("Evaluating the initial population...")
        self.line.set_data([], [])
        self.fitness_ax.set_xlim(0, self.settings["generations"])
        self.fitness_ax.set_ylim(0, 1)
        self.shape_ax.set_title(f"Best sofa - {self.settings['shape']}")
        self.metrics.set_text("Waiting for generation 0...")
        self.status.set_text("Search running. Stop keeps the best candidate evaluated so far.")
        self.update_controls()
        self.worker = Thread(target=run_search,
                             args=(self.settings.copy(), self.messages, self.stop_event),
                             daemon=True)
        self.worker.start()
        self.fig.canvas.draw_idle()

    def stop(self, _=None):
        if self.busy:
            self.stop_event.set()
            self.status.set_text("Stopping after the current candidate, then checking the best result...")
            self.fig.canvas.draw_idle()

    def show_progress(self, data):
        sofa = make_sofa(data["genes"], self.settings)
        report, history = data["report"], data["history"]
        self.placeholder.set_visible(False)
        self.patch.set_visible(True)
        self.patch.set_xy(np.asarray(sofa.exterior.coords))
        self.patch.set_facecolor("#3b82f6")
        parameters = decode(data["genes"])
        self.metrics.set_text(
            f"Area: {report['area']:.6f}     Fitness: {report['fitness']:.6f}\n"
            f"Width: {parameters['width']:.4f}     Height: {parameters['height']:.4f}"
            f"     Notch depth: {parameters['notch_depth']:.4f}\n"
            f"Search collisions: {report['colliding_samples']} / {report['samples_checked']} poses"
        )
        self.line.set_data([row["generation"] for row in history],
                           [row["best_fitness"] for row in history])
        self.fitness_ax.relim()
        self.fitness_ax.autoscale(enable=True, axis="y")
        row = history[-1]
        self.status.set_text(
            f"Generation {row['generation']} / {self.settings['generations']}\n"
            f"Passing candidates: {row['feasible_count']} / {row['evaluated_candidates']} evaluated\n"
            "Final finer check is still pending."
        )

    def poll(self):
        if self.closed:
            return
        changed = False
        while True:
            try:
                kind, data = self.messages.get_nowait()
            except Empty:
                break
            changed = True
            if kind == "generation":
                self.show_progress(data)
            elif kind == "checking":
                self.status.set_text("Search ended. Rechecking the best sofa with finer sampling...")
            elif kind == "done":
                self.busy, self.result = False, data
                self.update_controls()
                if data is None:
                    self.placeholder.set_text("No candidate evaluated")
                    self.metrics.set_text("Stopped before the first evaluation.")
                    self.status.set_text("Stopped. Click Start search to try again.")
                else:
                    report = data["verification_report"]
                    passed = data["passed_both_sampled_checks"]
                    label = "Stopped early" if data["stopped_early"] else "Completed"
                    self.patch.set_facecolor("#22c55e" if passed else "#ef4444")
                    self.status.set_text(
                        f"{label}. {'Passed both sampled checks.' if passed else 'FAILED sampled checks.'}\n"
                        f"Finer check: {report['colliding_samples']} collisions / {report['samples_checked']} poses.\n"
                        "Replay best to inspect the route, or Save result to keep this run."
                    )
            elif kind == "error":
                self.busy, self.result = False, None
                self.update_controls()
                self.status.set_text(fill(f"Search failed: {data}", width=85))
        if changed:
            self.fig.canvas.draw_idle()

    def replay(self, _=None):
        if self.busy or self.result is None:
            return
        try:
            result = self.result
            sofa = make_sofa(result["genes"], result["settings"])
            poses = sample_path(result["route_waypoints"], result["verification_max_distance"],
                                result["verification_max_angle"])
            sofas, fits, outside = check_path(sofa, poses)
            self.playback = animate_path(poses, sofas, fits, outside)
        except Exception as error:
            self.status.set_text(fill(f"Could not open replay: {error}", width=85))
            self.fig.canvas.draw_idle()

    def save(self, _=None):
        if self.busy or self.result is None:
            return
        try:
            OUTPUT_PATH.write_text(json.dumps(self.result, indent=2, allow_nan=False) + "\n",
                                   encoding="utf-8")
            label = "passed" if self.result["passed_both_sampled_checks"] else "FAILED"
            self.status.set_text(f"Saved best_sofa_gui.json beside this script.\n"
                                 f"Sampled checks: {label}. Dimensions, vertices, settings, and history included.")
        except (OSError, ValueError) as error:
            self.status.set_text(fill(f"Could not save: {error}", width=85))
        self.fig.canvas.draw_idle()

    def close(self, _=None):
        self.closed = True
        self.stop_event.set()
        self.timer.stop()


if __name__ == "__main__":
    window = SofaEvolutionWindow()
    plt.show()
