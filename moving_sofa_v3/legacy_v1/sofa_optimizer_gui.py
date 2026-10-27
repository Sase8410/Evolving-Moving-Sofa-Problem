import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Button, RadioButtons, Slider

from sofa_fitness import evaluate_sofa
from sofa_genetic_algorithm import BOUNDS, decode, make_candidate
from sofa_path_simulation import HALLWAY, TURNING_ROUTE, sample_path
from sofa_shape_simulation import check_path, transform_sofa


LIMITATION = (
    "Fixed route and shape family; sampled checks do not prove "
    "continuous clearance or global optimality."
)


@dataclass(frozen=True)
class Settings:
    shape: str = "curved"
    population: int = 40
    generations: int = 40
    mutation_rate: float = 0.30
    mutation_scale: float = 0.08
    seed: int = 42
    points: int = 60
    max_distance: float = 0.04
    max_angle: float = 2.0


class SearchWorker(threading.Thread):

    def __init__(self, settings):
        super().__init__(daemon=True)
        self.settings = settings
        self.messages = Queue()
        self.running = threading.Event()
        self.cancelled = threading.Event()
        self.running.set()

    def cancel(self):
        self.cancelled.set()
        self.running.set()  

    def checkpoint(self):
        while not self.cancelled.is_set():
            if self.running.wait(timeout=0.05):
                if not self.cancelled.is_set():
                    return
        raise InterruptedError

    def send(self, kind, payload):
        self.checkpoint()
        self.messages.put((kind, payload))

    def run(self):
        try:
            self.search()
        except InterruptedError:
            pass
        except Exception as error:
            self.messages.put(("error", str(error)))

    def search(self):
        args = self.settings
        rng = np.random.default_rng(args.seed)
        count = 2 if args.shape == "rectangle" else 3
        lower, upper = BOUNDS[:count].T
        span = upper - lower
        poses = sample_path(TURNING_ROUTE, args.max_distance, args.max_angle)
        population = rng.uniform(lower, upper, (args.population, count))
        population[0] = np.array([1.5, 0.6, 0.25 / 0.6])[:count]
        population[1] = np.array([0.4, 0.3, 0.25])[:count]
        cache, history = {}, []

        for generation in range(args.generations + 1):
            reports = []
            for genes in population:
                self.checkpoint()
                key = tuple(genes)
                if key not in cache:
                    sofa = make_candidate(genes, args.shape, args.points)
                    cache[key] = evaluate_sofa(sofa, poses)
                reports.append(cache[key])

            fitness = np.array([r["fitness"] for r in reports])
            order = np.argsort(-fitness, kind="stable")
            population, fitness = population[order], fitness[order]
            reports = [reports[i] for i in order]
            best = reports[0]
            feasible = [r["area"] for r in reports if r["sampled_feasible"]]
            row = {
                "generation": generation,
                "best_fitness": best["fitness"],
                "best_area": best["area"],
                "best_sampled_feasible": best["sampled_feasible"],
                "best_feasible_area": max(feasible, default=None),
                "feasible_count": len(feasible),
            }
            history.append(row)
            self.send("generation", (population[0].copy(), best, row))
            if generation == args.generations:
                break

            def parent():
                indices = rng.integers(0, args.population, size=3)
                return population[indices[np.argmax(fitness[indices])]]

            children = [population[0].copy(), population[1].copy()]
            while len(children) < args.population:
                self.checkpoint()
                a, b = parent(), parent()
                mix = rng.random(count)
                child = mix * a + (1 - mix) * b
                mutate = rng.random(count) < args.mutation_rate
                changes = rng.normal(0, args.mutation_scale, count) * span
                children.append(np.clip(child + mutate * changes, lower, upper))
            population = np.array(children)

        self.send("checking", None)
        genes = population[0]
        sofa = make_candidate(genes, args.shape, args.points)
        fine_distance = min(0.005, args.max_distance / 8)
        fine_angle = min(0.25, args.max_angle / 8)
        fine_poses = sample_path(TURNING_ROUTE, fine_distance, fine_angle)
        self.checkpoint()
        verification = evaluate_sofa(sofa, fine_poses)
        self.checkpoint()
        sofas, fits, outside = check_path(sofa, fine_poses)
        names = ["width", "height", "notch_ratio"][:count]
        result = {
            "shape": args.shape,
            "parameters": decode(genes),
            "curve_points": args.points,
            "genes": dict(zip(names, map(float, genes))),
            "vertices": np.asarray(sofa.exterior.coords).tolist(),
            "route_waypoints": TURNING_ROUTE.tolist(),
            "settings": {
                **asdict(args),
                "gene_bounds": dict(zip(names, BOUNDS[:count].tolist())),
                "verification_max_distance": fine_distance,
                "verification_max_angle": fine_angle,
            },
            "search_report": best,
            "verification_report": verification,
            "passed_both_sampled_checks": bool(
                best["sampled_feasible"] and verification["sampled_feasible"]
            ),
            "history": history,
            "limitation": LIMITATION,
        }
        self.send("done", (result, fine_poses, sofas, fits, outside))


class OptimizerWindow:
    def __init__(self):
        self.job = None
        self.result = None
        self.history = []
        self.phase = "ready"
        self.playing = False
        self.index = 0

        self.fig = plt.figure(figsize=(13, 8), facecolor="#f8fafc")
        self.fig.sofa_optimizer = self 
        self.fig.canvas.manager.set_window_title("Moving Sofa - Optimizer")
        self.ax = self.fig.add_axes([0.055, 0.40, 0.565, 0.53])
        self.ax.set_facecolor("#e2e8f0")
        self.ax.add_patch(PolygonPatch(
            np.asarray(HALLWAY.exterior.coords), closed=True,
            facecolor="white", edgecolor="#334155", linewidth=2,
        ))
        self.ax.plot(TURNING_ROUTE[:, 0], TURNING_ROUTE[:, 1],
                     "--", color="#64748b", linewidth=1)
        self.patch = PolygonPatch(
            np.zeros((3, 2)), closed=True, edgecolor="#334155", alpha=0.85,
        )
        self.ax.add_patch(self.patch)
        self.ax.set(xlim=(-4, 1.5), ylim=(-0.6, 4), xlabel="x", ylabel="y",
                    title="Best candidate on the fixed turning route")
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.2)

        self.chart = self.fig.add_axes([0.085, 0.135, 0.515, 0.17])
        self.line, = self.chart.plot([], [], color="#2563eb", linewidth=2)
        self.chart.set(xlabel="Generation", ylabel="Best fitness")
        self.chart.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.chart.grid(alpha=0.2)

        self.fig.text(0.70, 0.95, "EVOLUTION SETTINGS", weight="bold")
        self.shape = RadioButtons(
            self.fig.add_axes([0.70, 0.80, 0.25, 0.12]),
            ["rectangle", "notched", "curved"], active=2,
        )
        self.sliders = {}
        specs = [
            ("population", "Population", 10, 100, 40, 10, "%d"),
            ("generations", "Generations", 5, 150, 40, 5, "%d"),
            ("mutation_rate", "Mutation probability", 0, 1, 0.30, 0.05, "%.2f"),
            ("mutation_scale", "Mutation strength", 0.01, 0.30, 0.08, 0.01, "%.2f"),
            ("seed", "Random seed", 0, 999, 42, 1, "%d"),
        ]
        for i, (name, label, low, high, value, step, fmt) in enumerate(specs):
            y = 0.735 - i * 0.058
            self.fig.text(0.70, y + 0.030, label, fontsize=9)
            self.sliders[name] = Slider(
                self.fig.add_axes([0.70, y, 0.23, 0.020]), "", low, high,
                valinit=value, valstep=step, valfmt=fmt,
            )

        self.metrics = self.fig.text(0.70, 0.448, "", va="top", fontsize=10)
        self.buttons = {}
        for label, position, callback in [
            ("Start", [0.70, 0.24, 0.075, 0.045], self.start),
            ("Pause", [0.785, 0.24, 0.075, 0.045], self.pause),
            ("Reset", [0.87, 0.24, 0.080, 0.045], self.reset),
            ("Replay", [0.70, 0.17, 0.12, 0.045], self.replay),
            ("Save", [0.84, 0.17, 0.11, 0.045], self.save),
        ]:
            button = Button(self.fig.add_axes(position), label)
            button.on_clicked(callback)
            self.buttons[label] = button

        self.fig.text(0.70, 0.10,
                      "Reset unlocks settings for a new run.\n"
                      "Pause finishes the current check.", fontsize=9)
        self.status = self.fig.text(0.055, 0.052, "", fontsize=9)
        self.fig.text(0.5, 0.017,
                      "Fixed route and shape family. Collisions between samples may be missed.",
                      ha="center", fontsize=9, color="#64748b")
        self.reset()
        self.timer = self.fig.canvas.new_timer(interval=25)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", self.close)

    def read_settings(self):
        values = {name: float(s.val) for name, s in self.sliders.items()}
        for name in ("population", "generations", "seed"):
            values[name] = int(values[name])
        return Settings(shape=self.shape.value_selected, **values)

    def controls(self):
        busy = self.phase in ("searching", "checking")
        for widget in [self.shape, *self.sliders.values()]:
            widget.active = self.job is None
        enabled = {
            "Start": self.job is None, "Pause": busy, "Reset": True,
            "Replay": self.result is not None, "Save": self.result is not None,
        }
        for name, button in self.buttons.items():
            button.active = enabled[name]
            button.label.set_alpha(1 if enabled[name] else 0.35)
        paused = busy and not self.job.running.is_set()
        self.buttons["Pause"].label.set_text("Resume" if paused else "Pause")
        self.buttons["Replay"].label.set_text("Stop replay" if self.playing else "Replay")
        self.fig.canvas.draw_idle()

    def start(self, _=None):
        if self.job is not None:
            return
        self.job = SearchWorker(self.read_settings())
        self.phase = "searching"
        self.status.set_text("Searching: evaluating the initial population...")
        self.controls()
        self.job.start()

    def pause(self, _=None):
        if self.phase not in ("searching", "checking"):
            return
        if self.job.running.is_set():
            self.job.running.clear()
            self.status.set_text("Paused after the current check. Click Resume to continue.")
        else:
            self.job.running.set()
            self.status.set_text("Resumed search / final checking.")
        self.controls()

    def reset(self, _=None):
        if self.job is not None:
            self.job.cancel()

        self.job, self.result = None, None
        self.phase, self.playing, self.index = "ready", False, 0
        self.history = []
        self.playback = None
        self.patch.set_visible(False)
        self.line.set_data([], [])
        self.chart.set(xlim=(-0.5, 1), ylim=(0, 1))
        self.metrics.set_text("Choose settings, then click Start.")
        self.status.set_text("Ready. Generation 0 is the initial population.")
        self.controls()

    def draw_sofa(self, sofa, fits):
        self.patch.set_visible(True)
        self.patch.set_xy(np.asarray(sofa.exterior.coords))
        self.patch.set_facecolor("#22c55e" if fits else "#ef4444")

    def update_generation(self, genes, report, row):
        self.history.append(row)
        settings = self.job.settings
        sofa = make_candidate(genes, settings.shape, settings.points)
        self.draw_sofa(transform_sofa(sofa, *TURNING_ROUTE[0]),
                       report["sampled_feasible"])
        self.line.set_data([r["generation"] for r in self.history],
                           [r["best_fitness"] for r in self.history])
        self.chart.set_autoscaley_on(True)
        self.chart.relim()
        self.chart.autoscale_view(scalex=False)
        self.chart.set_xlim(-0.5, max(1, settings.generations))
        self.metrics.set_text(
            f"Shape: {settings.shape}\n"
            f"Generation: {row['generation']} / {settings.generations}\n"
            f"Best area: {report['area']:.5f}\n"
            f"Best fitness: {report['fitness']:.5f}\n"
            f"Passing search samples: {row['feasible_count']} / {settings.population}\n"
            f"Best candidate collisions: {report['colliding_samples']}"
        )

    def tick(self):
        if self.job is not None:
            while True:
                try:
                    kind, data = self.job.messages.get_nowait()
                except Empty:
                    break
                if kind == "generation":
                    self.update_generation(*data)
                    if self.job.running.is_set():
                        self.status.set_text(
                            f"Searching: generation {data[2]['generation']} / "
                            f"{self.job.settings.generations}."
                        )
                elif kind == "checking":
                    self.phase = "checking"
                    if self.job.running.is_set():
                        self.status.set_text("Checking the winner at finer path intervals...")
                elif kind == "done":
                    self.result, *self.playback = data
                    self.phase = "done"
                    report = self.result["verification_report"]
                    self.metrics.set_text(self.metrics.get_text() +
                                          f"\nFiner check collisions: {report['colliding_samples']}")
                    passed = self.result["passed_both_sampled_checks"]
                    self.patch.set_facecolor("#22c55e" if passed else "#ef4444")
                    self.status.set_text(
                        "Finished: passed both sampled checks. Replay or save the result."
                        if passed else
                        "Finished: candidate failed a sampled check. Replay to inspect the collision."
                    )
                elif kind == "error":
                    self.phase = "error"
                    self.status.set_text(f"Search failed: {data}. Click Reset to try again.")
                self.controls()
        if self.playing:
            self.show_frame()

    def replay(self, _=None):
        if self.result is None:
            return
        self.playing = not self.playing
        if self.playing:
            self.index = 0
        else:
            self.status.set_text("Replay stopped. Click Replay to restart.")
        self.controls()

    def show_frame(self):
        poses, sofas, fits, outside = self.playback
        i = self.index
        self.draw_sofa(sofas[i], fits[i])
        self.status.set_text(
            f"Replay: sample {i + 1}/{len(poses)} | Angle {poses[i, 2]:.1f} deg | "
            f"Outside area {outside[i]:.6f}"
        )
        if not fits[i] or i == len(poses) - 1:
            self.playing = False
            ending = "Collision - stopped." if not fits[i] else "Replay finished."
            self.status.set_text(self.status.get_text() + " | " + ending)
        else:
            self.index += 1
        self.controls()

    def save(self, _=None):
        if self.result is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(__file__).resolve().with_name(f"best_sofa_{stamp}.json")
        try:
            with path.open("x", encoding="utf-8") as file:
                json.dump(self.result, file, indent=2, allow_nan=False)
                file.write("\n")
            self.playing = False
            self.status.set_text(f"Saved {path.name} beside this Python file.")
        except (OSError, ValueError) as error:
            self.status.set_text(f"Could not save: {error}")
        self.controls()

    def close(self, _=None):
        self.timer.stop()
        if self.job is not None:
            self.job.cancel()


def main():
    window = OptimizerWindow()
    plt.show()
    return window


if __name__ == "__main__":
    main()
