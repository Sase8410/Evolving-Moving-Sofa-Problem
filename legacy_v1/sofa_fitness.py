import argparse
import numpy as np
from shapely.geometry import Polygon

from sofa_path_simulation import (
    TURNING_ROUTE, COLLISION_ROUTE, sample_path, animate_path,
)
from sofa_shape_simulation import create_candidate, check_path


def evaluate_sofa(sofa, poses, outside_weight=10.0):
    poses = np.asarray(poses, dtype=float)
    if (poses.ndim != 2 or poses.shape[1] != 3
            or len(poses) < 2 or not np.isfinite(poses).all()):
        raise ValueError("Use at least two finite [x, y, angle] poses.")
    if not np.isfinite(outside_weight) or outside_weight <= 0:
        raise ValueError("Outside weight must be finite and positive.")

    result = {
        "valid_shape": False,
        "sampled_feasible": False,
        "fitness": -np.inf,
        "area": None,
        "samples_checked": 0,
        "colliding_samples": None,
        "collision_fraction": None,
        "mean_outside_area": None,
        "max_outside_area": None,
        "first_collision_index": None,
        "reason": "Expected a valid, nonempty Polygon with positive finite area.",
    }

    if (not isinstance(sofa, Polygon) or sofa.is_empty
            or not sofa.is_valid or not np.isfinite(sofa.area)
            or sofa.area <= 0):
        return result

    _, fits, outside_areas = check_path(sofa, poses)
    collisions = np.flatnonzero(~fits)
    collision_count = len(collisions)
    collision_fraction = collision_count / len(poses)
    mean_outside = float(np.mean(outside_areas))
    max_outside = float(np.max(outside_areas))
    area = float(sofa.area)

    penalty = collision_fraction + outside_weight * (mean_outside + max_outside)
    sampled_feasible = collision_count == 0
    fitness = area if sampled_feasible else -penalty

    result.update({
        "valid_shape": True,
        "sampled_feasible": sampled_feasible,
        "fitness": float(fitness),
        "area": area,
        "samples_checked": len(poses),
        "colliding_samples": collision_count,
        "collision_fraction": collision_fraction,
        "mean_outside_area": mean_outside,
        "max_outside_area": max_outside,
        "first_collision_index": int(collisions[0]) if collision_count else None,
        "reason": ("No collisions at sampled poses." if sampled_feasible
                   else "Collisions detected at sampled poses."),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["rectangle", "notched", "curved"],
                        default="curved")
    parser.add_argument("--width", type=float, default=1.5)
    parser.add_argument("--height", type=float, default=0.6)
    parser.add_argument("--notch-depth", type=float, default=0.25)
    parser.add_argument("--points", type=int, default=60)
    parser.add_argument("--route", choices=["turn", "collision"], default="turn")
    parser.add_argument("--outside-weight", type=float, default=10.0)
    parser.add_argument("--max-distance", type=float, default=0.01)
    parser.add_argument("--max-angle", type=float, default=0.5)
    parser.add_argument("--animate", action="store_true",
                        help="Open the existing Play/Reset simulation window.")
    args = parser.parse_args()

    try:
        sofa = create_candidate(
            args.shape, args.width, args.height, args.notch_depth, args.points,
        )
        route = TURNING_ROUTE if args.route == "turn" else COLLISION_ROUTE
        poses = sample_path(route, args.max_distance, args.max_angle)
        result = evaluate_sofa(sofa, poses, args.outside_weight)
    except ValueError as error:
        parser.error(str(error))

    if not result["valid_shape"]:
        parser.error(result["reason"])

    print(f"Shape: {args.shape} | Route: {args.route}")
    print(f"Sofa area: {result['area']:.6f} square units")
    print(f"Fitness: {result['fitness']:.6f} (higher is better)")
    print(f"Sampled feasible: {result['sampled_feasible']}")
    print(f"Colliding samples: {result['colliding_samples']} / "
          f"{result['samples_checked']}")
    print(f"Collision fraction: {result['collision_fraction']:.2%}")
    print(f"Mean outside area: {result['mean_outside_area']:.6f}")
    print(f"Maximum outside area: {result['max_outside_area']:.6f}")
    first = result["first_collision_index"]
    if first is not None:
        print(f"First collision at sample {first + 1}: {poses[first]}")
    print("Sampling does not prove continuous collision-free motion.")

    if args.animate:
        sofas, fits, outside_areas = check_path(sofa, poses)
        animate_path(poses, sofas, fits, outside_areas)


if __name__ == "__main__":
    main()
