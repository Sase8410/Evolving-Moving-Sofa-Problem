import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sofa_fitness import evaluate_sofa
from sofa_path_simulation import TURNING_ROUTE, sample_path, animate_path
from sofa_shape_simulation import create_candidate, check_path

BOUNDS = np.array([[0.20, 2.40], [0.15, 1.00], [0.02, 0.95]])


def decode(genes):
    width, height = map(float, genes[:2])
    return {
        "width": width,
        "height": height,
        "notch_depth": height * float(genes[2]) if len(genes) == 3 else 0.0,
    }


def make_candidate(genes, shape, points):
    return create_candidate(shape, **decode(genes), curve_points=points)


def optimize(args, poses):
    rng = np.random.default_rng(args.seed)
    gene_count = 2 if args.shape == "rectangle" else 3
    lower, upper = BOUNDS[:gene_count].T
    span = upper - lower

    population = rng.uniform(lower, upper, (args.population, gene_count))

    population[0] = np.array([1.5, 0.6, 0.25 / 0.6])[:gene_count]
    population[1] = np.array([0.4, 0.3, 0.25])[:gene_count]

    history = []
    cache = {}

    def evaluate(genes):
        key = tuple(genes)
        if key not in cache:
            sofa = make_candidate(genes, args.shape, args.points)
            cache[key] = evaluate_sofa(sofa, poses)
        return cache[key]

    for generation in range(args.generations + 1):
        reports = [evaluate(genes) for genes in population]
        fitness = np.array([report["fitness"] for report in reports])
        order = np.argsort(-fitness, kind="stable")
        population = population[order]
        fitness = fitness[order]
        reports = [reports[i] for i in order]

        best = reports[0]
        feasible_areas = [r["area"] for r in reports if r["sampled_feasible"]]
        history.append({
            "generation": generation,
            "best_fitness": best["fitness"],
            "best_area": best["area"],
            "best_sampled_feasible": best["sampled_feasible"],
            "best_feasible_area": max(feasible_areas, default=None),
            "feasible_count": len(feasible_areas),
        })
        status = "passes samples" if best["sampled_feasible"] else "collides"
        print(
            f"Generation {generation:3d}/{args.generations} | "
            f"Fitness {best['fitness']:.6f} | Area {best['area']:.6f} | "
            f"Feasible {len(feasible_areas)}/{args.population} | {status}",
            flush=True,
        )

        if generation == args.generations:
            break

        def select_parent():
            contestants = rng.integers(0, args.population, size=3)
            winner = contestants[np.argmax(fitness[contestants])]
            return population[winner]

        next_population = [population[0].copy(), population[1].copy()]
        while len(next_population) < args.population:
            parent_a, parent_b = select_parent(), select_parent()
            mix = rng.random(gene_count)
            child = mix * parent_a + (1.0 - mix) * parent_b

            mutate = rng.random(gene_count) < args.mutation_rate
            changes = rng.normal(0.0, args.mutation_scale, gene_count) * span
            child = np.clip(child + mutate * changes, lower, upper)
            next_population.append(child)

        population = np.array(next_population)

    return population[0].copy(), reports[0], history


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["rectangle", "notched", "curved"],
                        default="curved")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--points", type=int, default=60,
                        help="Polygon points per curve; held fixed during search.")
    parser.add_argument("--mutation-rate", type=float, default=0.30,
                        help="Mutation probability per gene.")
    parser.add_argument("--mutation-scale", type=float, default=0.08,
                        help="Mutation standard deviation as a fraction of gene range.")
    parser.add_argument("--max-distance", type=float, default=0.04,
                        help="Maximum translation per search sample.")
    parser.add_argument("--max-angle", type=float, default=2.0,
                        help="Maximum rotation in degrees per search sample.")
    parser.add_argument("--output", type=Path, default=Path("best_sofa.json"))
    parser.add_argument("--plot", action="store_true",
                        help="Show fitness improvement after the search.")
    parser.add_argument("--animate", action="store_true",
                        help="Open the existing Play/Reset simulation afterward.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.population < 4 or args.generations < 0:
        parser.error("Use population >= 4 and generations >= 0.")
    if args.seed < 0 or args.points < 3:
        parser.error("Use seed >= 0 and points >= 3.")
    if not np.isfinite(args.mutation_rate) or not 0 <= args.mutation_rate <= 1:
        parser.error("Mutation rate must be between 0 and 1.")
    for name in ("mutation_scale", "max_distance", "max_angle"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"{name.replace('_', '-')} must be finite and positive.")

    poses = sample_path(TURNING_ROUTE, args.max_distance, args.max_angle)
    print(f"Fixed turning route | Shape: {args.shape} | Search samples: {len(poses)}")
    genes, search_report, history = optimize(args, poses)
    sofa = make_candidate(genes, args.shape, args.points)

    fine_distance = min(0.005, args.max_distance / 8)
    fine_angle = min(0.25, args.max_angle / 8)
    fine_poses = sample_path(TURNING_ROUTE, fine_distance, fine_angle)
    verification = evaluate_sofa(sofa, fine_poses)
    passed = search_report["sampled_feasible"] and verification["sampled_feasible"]

    gene_names = ["width", "height", "notch_ratio"][:len(genes)]
    result = {
        "shape": args.shape,
        "parameters": decode(genes),
        "curve_points": args.points,
        "genes": dict(zip(gene_names, map(float, genes))),
        "vertices": np.asarray(sofa.exterior.coords).tolist(),
        "route_waypoints": TURNING_ROUTE.tolist(),
        "settings": {
            "seed": args.seed,
            "population": args.population,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "mutation_scale": args.mutation_scale,
            "gene_bounds": dict(zip(gene_names, BOUNDS[:len(genes)].tolist())),
            "search_max_distance": args.max_distance,
            "search_max_angle": args.max_angle,
            "verification_max_distance": fine_distance,
            "verification_max_angle": fine_angle,
        },
        "search_report": search_report,
        "verification_report": verification,
        "passed_both_sampled_checks": bool(passed),
        "history": history,
        "limitation": "Fixed route and shape family; sampled checks are not a "
                      "proof of continuous clearance or global optimality.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")

    print(f"\nBest search candidate: {decode(genes)}")
    print(f"Area: {sofa.area:.6f} square units")
    print(f"Finer check: {verification['colliding_samples']} collisions / "
          f"{len(fine_poses)} samples")
    print(f"Passed both sampled checks: {passed}")
    if not passed:
        print("The saved candidate is not feasible at all checked poses. "
              "Rerun with smaller --max-distance and --max-angle values.")
    print("Sampling does not prove continuous collision-free motion.")
    print(f"Saved candidate and generation history: {args.output.resolve()}")

    if args.plot:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot([h["generation"] for h in history],
                [h["best_fitness"] for h in history], color="#2563eb")
        ax.set(xlabel="Generation", ylabel="Best search fitness (higher is better)",
               title=f"Moving Sofa - {args.shape} evolution on a fixed route")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()

    if args.animate:
        sofas, fits, outside_areas = check_path(sofa, fine_poses)
        animate_path(fine_poses, sofas, fits, outside_areas)


if __name__ == "__main__":
    main()
