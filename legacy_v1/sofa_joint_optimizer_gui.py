import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sofa_fitness import evaluate_sofa
from sofa_genetic_algorithm import BOUNDS, decode, make_candidate
from sofa_optimizer_gui import OptimizerWindow, SearchWorker
from sofa_path_simulation import TURNING_ROUTE, sample_path
from sofa_shape_simulation import check_path


POSITION_RANGE = 0.30
ANGLE_RANGE = 6.0
FINALISTS = 5
LIMITATION = (
    "Bounded shape families and waypoint offsets; passing sampled checks "
    "does not prove continuous clearance or global optimality."
)


def shape_count(shape):
    return 2 if shape == "rectangle" else 3


def joint_bounds(shape):
    positions = np.tile([-POSITION_RANGE, POSITION_RANGE],
                        (2 * (len(TURNING_ROUTE) - 2), 1))
    angles = np.tile([-ANGLE_RANGE, ANGLE_RANGE],
                     (len(TURNING_ROUTE) - 4, 1))
    return np.vstack((BOUNDS[:shape_count(shape)], positions, angles))


def decode_route(genes, shape):

    count = shape_count(shape)
    route = TURNING_ROUTE.copy()
    position_count = 2 * (len(route) - 2)
    route[1:-1, :2] += genes[count:count + position_count].reshape(-1, 2)
    route[2:-2, 2] += genes[count + position_count:]
    return route


class JointSearchWorker(SearchWorker):
    def search(self):
        args = self.settings
        rng = np.random.default_rng(args.seed)
        count = shape_count(args.shape)
        bounds = joint_bounds(args.shape)
        lower, upper = bounds.T
        span = upper - lower
        population = rng.uniform(lower, upper, (args.population, len(bounds)))

        population[:args.population // 2, count:] = 0.0
        population[args.population // 2:, count:] *= 0.20
        population[0, :count] = np.array([1.5, 0.6, 0.25 / 0.6])[:count]
        population[1, :count] = np.array([0.4, 0.3, 0.25])[:count]
        population[:2, count:] = 0.0
        history, cache, archive = [], {}, {}

        def evaluate(genes, distance, angle):
            self.checkpoint()
            sofa = make_candidate(genes[:count], args.shape, args.points)
            poses = sample_path(decode_route(genes, args.shape), distance, angle)
            return evaluate_sofa(sofa, poses)

        for generation in range(args.generations + 1):
            reports = []
            current_cache = {}
            for genes in population:
                self.checkpoint()
                key = tuple(genes)
                report = cache.get(key)
                if report is None:
                    report = current_cache.get(key)
                if report is None:
                    report = evaluate(genes, args.max_distance, args.max_angle)
                current_cache[key] = report
                reports.append(report)
                if report["sampled_feasible"]:
                    archive[key] = (genes.copy(), report)
            cache = current_cache

            archive = dict(sorted(archive.items(),
                                  key=lambda item: -item[1][1]["area"])[:FINALISTS])
            order = np.argsort([-r["fitness"] for r in reports], kind="stable")
            population = population[order]
            reports = [reports[i] for i in order]
            fitness = np.array([r["fitness"] for r in reports])
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
                indices = rng.integers(0, args.population, 3)
                return population[indices[np.argmax(fitness[indices])]]

            children = [population[0].copy(), population[1].copy()]
            while len(children) < args.population:
                self.checkpoint()
                a, b = parent(), parent()
                mix = rng.random(len(bounds))
                child = mix * a + (1 - mix) * b
                mutation = rng.random(len(bounds)) < args.mutation_rate

                mode = rng.integers(0, 3)
                if mode == 0:
                    mutation[count:] = False
                elif mode == 1:
                    mutation[:count] = False
                delta = rng.normal(0, args.mutation_scale, len(bounds)) * span
                children.append(np.clip(child + mutation * delta, lower, upper))
            population = np.array(children)

        self.send("checking", None)
        distance = min(0.005, args.max_distance / 8)
        angle = min(0.25, args.max_angle / 8)
        finalists = list(archive.values()) or [(population[0].copy(), best)]
        attempts, selected = [], None
        for genes, report in finalists:
            verification = evaluate(genes, distance, angle)
            attempts.append({"area": report["area"],
                             "verification_report": verification})

            if selected is None or verification["sampled_feasible"]:
                selected = (genes, report, verification)
            if report["sampled_feasible"] and verification["sampled_feasible"]:
                break

        genes, report, verification = selected
        route = decode_route(genes, args.shape)
        sofa = make_candidate(genes[:count], args.shape, args.points)
        poses = sample_path(route, distance, angle)
        self.checkpoint()
        sofas, fits, outside = check_path(sofa, poses)
        result = {
            "step": 7,
            "shape": args.shape,
            "parameters": decode(genes[:count]),
            "curve_points": args.points,
            "joint_genes": genes.tolist(),
            "vertices": np.asarray(sofa.exterior.coords).tolist(),
            "route_waypoints": route.tolist(),
            "reference_route_waypoints": TURNING_ROUTE.tolist(),
            "settings": {
                **asdict(args), "gene_bounds": bounds.tolist(),
                "gene_layout": "shape genes, internal xy offsets, turning angle offsets",
                "verification_max_distance": distance,
                "verification_max_angle": angle,
                "finalist_limit": FINALISTS,
            },
            "search_report": report,
            "verification_report": verification,
            "passed_both_sampled_checks": bool(
                report["sampled_feasible"] and verification["sampled_feasible"]),
            "finalist_checks": attempts,
            "history": history,
            "limitation": LIMITATION,
        }
        self.send("done", (result, poses, sofas, fits, outside))


class JointOptimizerWindow(OptimizerWindow):
    def __init__(self):
        super().__init__()
        self.fig.canvas.manager.set_window_title("Moving Sofa - Shape and Route")
        self.ax.set_title("Best candidate and evolving route")
        self.ax.lines[0].set_alpha(0.35)
        self.ax.lines[0].set_label("Original route")
        self.route_line, = self.ax.plot(
            TURNING_ROUTE[:, 0], TURNING_ROUTE[:, 1],
            "o-", color="#2563eb", markersize=3, linewidth=1,
            label="Candidate route",
        )
        self.ax.legend(loc="upper left", fontsize=8)
        for label in self.fig.texts:
            if label.get_text().startswith("Fixed route and shape family."):
                label.set_text("Shape and route evolve within bounds. "
                               "Sampled clearance is not a continuous-motion proof.")

    def start(self, _=None):
        if self.job is not None:
            return
        self.job = JointSearchWorker(self.read_settings())
        self.phase = "searching"
        self.status.set_text("Searching sofa dimensions and route waypoints...")
        self.controls()
        self.job.start()

    def reset(self, _=None):
        super().reset()
        if hasattr(self, "route_line"):
            self.route_line.set_data(TURNING_ROUTE[:, 0], TURNING_ROUTE[:, 1])

    def update_generation(self, genes, report, row):
        shape = self.job.settings.shape
        super().update_generation(genes[:shape_count(shape)], report, row)
        route = decode_route(genes, shape)
        self.route_line.set_data(route[:, 0], route[:, 1])

    def tick(self):
        previous = self.result
        super().tick()
        if self.result is not None and self.result is not previous:
            route = np.asarray(self.result["route_waypoints"])
            self.route_line.set_data(route[:, 0], route[:, 1])
            self.draw_sofa(self.playback[1][0],
                           self.result["passed_both_sampled_checks"])
            report = self.result["verification_report"]
            self.metrics.set_text(
                f"Shape: {self.result['shape']}\n"
                f"Selected area: {report['area']:.5f}\n"
                f"Finer check collisions: {report['colliding_samples']}\n"
                f"Samples checked: {report['samples_checked']}\n"
                f"Finalists checked: {len(self.result['finalist_checks'])}\n"
                "Blue line: selected route"
            )
            self.fig.canvas.draw_idle()

    def save(self, _=None):
        if self.result is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = Path(__file__).resolve().with_name(f"joint_sofa_{stamp}.json")
        try:
            with path.open("x", encoding="utf-8") as file:
                json.dump(self.result, file, indent=2, allow_nan=False)
                file.write("\n")
            self.playing = False
            self.status.set_text(f"Saved {path.name} beside this Python file.")
        except (OSError, ValueError) as error:
            self.status.set_text(f"Could not save: {error}")
        self.controls()


def main():
    window = JointOptimizerWindow()
    plt.show()
    return window


if __name__ == "__main__":
    main()
