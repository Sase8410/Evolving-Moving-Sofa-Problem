import argparse
import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.validation import explain_validity

from sofa_path_simulation import (
    HALLWAY, TURNING_ROUTE, COLLISION_ROUTE, sample_path, animate_path,
)


def make_sofa(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 2
            or len(vertices) < 3 or not np.isfinite(vertices).all()):
        raise ValueError("Use at least three finite [x, y] vertices.")

    sofa = Polygon(vertices)
    if not sofa.is_valid:
        raise ValueError(f"Invalid sofa: {explain_validity(sofa)}")
    if sofa.is_empty or not np.isfinite(sofa.area) or sofa.area <= 0:
        raise ValueError("The sofa must have a finite, positive area.")
    return sofa


def create_candidate(shape, width=1.5, height=0.6,
                     notch_depth=0.25, curve_points=60):
    if not all(np.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Width and height must be finite and positive.")

    a, b = width / 2, height / 2

    if shape == "rectangle":
        return make_sofa([[-a, -b], [a, -b], [a, b], [-a, b]])

    if not np.isfinite(notch_depth) or not 0 < notch_depth < height:
        raise ValueError("Notch depth must be greater than 0 and less than height.")

    if shape == "notched":
        # Walk around the boundary, including the inward bottom notch.
        return make_sofa([
            [-a, -b], [-a / 3, -b],
            [-a / 3, -b + notch_depth], [a / 3, -b + notch_depth],
            [a / 3, -b], [a, -b], [a, b], [-a, b],
        ])

    if shape == "curved":
        if not isinstance(curve_points, (int, np.integer)) or curve_points < 3:
            raise ValueError("Use at least three points per curve.")
        # Two elliptical arches form a concave outline.
        # Straight polygon edges approximate the curves.
        angles = np.linspace(0, np.pi, curve_points)
        outer = np.column_stack((
            a * np.cos(angles), -b + height * np.sin(angles),
        ))
        inner = np.column_stack((
            a * np.cos(angles[::-1]),
            -b + notch_depth * np.sin(angles[::-1]),
        ))
        return make_sofa(np.vstack((outer, inner[1:-1])))

    raise ValueError(f"Unknown shape: {shape}")


def transform_sofa(sofa, x, y, angle_degrees):
    rotated = rotate(sofa, angle_degrees, origin=(0, 0))
    return translate(rotated, xoff=x, yoff=y)


def check_path(sofa, poses):
    sofas = [transform_sofa(sofa, *pose) for pose in poses]
    fits = np.array([HALLWAY.covers(s) for s in sofas], dtype=bool)
    outside_areas = np.array([s.difference(HALLWAY).area for s in sofas])
    return sofas, fits, outside_areas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", choices=["rectangle", "notched", "curved"],
                        default="curved")
    parser.add_argument("--width", type=float, default=1.5)
    parser.add_argument("--height", type=float, default=0.6)
    parser.add_argument("--notch-depth", type=float, default=0.25)
    parser.add_argument("--points", type=int, default=60)
    parser.add_argument("--route", choices=["turn", "collision"], default="turn")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    try:
        sofa = create_candidate(
            args.shape, args.width, args.height, args.notch_depth, args.points,
        )
        # For a custom shape, replace the call above with make_sofa(vertices).
    except ValueError as error:
        parser.error(str(error))

    route = TURNING_ROUTE if args.route == "turn" else COLLISION_ROUTE
    poses = sample_path(route)
    sofas, fits, outside_areas = check_path(sofa, poses)
    invalid = np.flatnonzero(~fits)

    print(f"Shape: {args.shape} | Vertices: {len(sofa.exterior.coords) - 1}")
    print(f"Sofa area: {sofa.area:.6f} square units")
    print(f"Route: {args.route} | Samples checked: {len(poses)}")
    print(f"Colliding samples: {len(invalid)}")
    if len(invalid):
        first = invalid[0]
        print(f"First collision at sample {first + 1}: {poses[first]}")
    else:
        print("No collisions found at sampled poses.")
    print("Sampling does not prove continuous collision-free motion.")

    if not args.check_only:
        animate_path(poses, sofas, fits, outside_areas)


if __name__ == "__main__":
    main()
