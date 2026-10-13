from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from shapely.affinity import affine_transform, translate
from shapely.geometry import Polygon, box, mapping, shape
from shapely.geometry.polygon import orient
from shapely.ops import unary_union
from shapely.validation import explain_validity


class Cancelled(Exception):
    """Cooperative cancellation from GUI or caller."""


def noop():
    pass


@dataclass(frozen=True)
class Hallway:
    length: float = 5.0

    def __post_init__(self):
        if not np.isfinite(self.length) or not 3 <= self.length <= 10000:
            raise ValueError("Hallway length must be finite and at least 3.")

    @property
    def region(self):
        return Polygon(
            [
                (-self.length, 0),
                (1, 0),
                (1, self.length),
                (0, self.length),
                (0, 1),
                (-self.length, 1),
            ]
        )

    @property
    def entrance(self):
        return box(-self.length, 0, 0, 1)

    @property
    def exit(self):
        return box(0, 1, 1, self.length)


def require_polygon(sofa):
    if not isinstance(sofa, Polygon) or sofa.is_empty:
        raise ValueError("Use one nonempty connected polygon.")
    coords = np.vstack([np.asarray(r.coords) for r in [sofa.exterior, *sofa.interiors]])
    if len(coords) > 100000 or not np.isfinite(coords).all():
        raise ValueError("Use at most 100,000 finite polygon vertices.")
    if np.max(np.abs(coords)) > 1000:
        raise ValueError("Coordinates must be within +/-1000 hallway units.")
    if not sofa.is_valid:
        raise ValueError(f"Invalid polygon: {explain_validity(sofa)}")
    if not np.isfinite(sofa.area) or sofa.area <= 1e-12:
        raise ValueError("The polygon must have positive finite area.")
    return orient(sofa, sign=1)


def make_sofa(vertices, holes=None):
    for ring in [vertices, *(holes or [])]:
        values = np.asarray(ring, dtype=float)
        if (
            values.ndim != 2
            or values.shape[1] != 2
            or len(values) < 3
            or not np.isfinite(values).all()
        ):
            raise ValueError("Polygon rings need at least three finite [x,y] vertices.")
    return require_polygon(Polygon(vertices, holes))


def centered(sofa):
    sofa = require_polygon(sofa)
    return translate(sofa, -sofa.centroid.x, -sofa.centroid.y)


def preset(name, width=1.5, height=0.6, notch=0.25):
    if not all(np.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError("Width and height must be positive.")
    a, b = width / 2, height / 2
    if name == "Rectangle":
        return box(-a, -b, a, b)
    if not 0 < notch < height:
        raise ValueError("Notch depth must lie between 0 and the height.")
    if name == "Notched":
        return make_sofa(
            [
                (-a, -b),
                (-a / 3, -b),
                (-a / 3, -b + notch),
                (a / 3, -b + notch),
                (a / 3, -b),
                (a, -b),
                (a, b),
                (-a, b),
            ]
        )
    if name == "Curved":
        t = np.linspace(0, np.pi, 80)
        outer = np.column_stack((a * np.cos(t), -b + height * np.sin(t)))
        inner = np.column_stack((a * np.cos(t[::-1]), -b + notch * np.sin(t[::-1])))
        return make_sofa(np.vstack((outer, inner[1:-1])))
    raise ValueError("Unknown shape preset.")


def edit_boundary(sofa, position, width, depth, operation):
    sofa = require_polygon(sofa)
    if (
        not all(np.isfinite(v) for v in (position, width, depth))
        or width <= 0
        or depth <= 0
    ):
        raise ValueError("Choose a positive edit width and depth.")
    v = np.asarray(sofa.exterior.coords)
    lengths = np.linalg.norm(np.diff(v, axis=0), axis=1)
    usable = lengths > 1e-12
    starts, edges, lengths = v[:-1][usable], np.diff(v, axis=0)[usable], lengths[usable]
    cumulative = np.cumsum(lengths)
    d = (position % 1) * cumulative[-1]
    i = min(np.searchsorted(cumulative, d, side="right"), len(lengths) - 1)
    tangent = edges[i] / lengths[i]
    anchor = starts[i] + (d - (cumulative[i - 1] if i else 0)) * tangent
    normal = np.array([tangent[1], -tangent[0]])
    low, high = (-depth, 1e-7) if operation == "Notch" else (-1e-7, depth)
    patch = Polygon(
        [
            anchor + u * tangent + w * normal
            for u, w in [
                (-width / 2, low),
                (width / 2, low),
                (width / 2, high),
                (-width / 2, high),
            ]
        ]
    )
    result = sofa.difference(patch) if operation == "Notch" else sofa.union(patch)
    return require_polygon(result)


def world_sofa(sofa, pose):
    x, y, angle = pose
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return affine_transform(sofa, [c, -s, s, c, x, y])


def local_region(region, pose):
    x, y, angle = pose
    c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
    return affine_transform(region, [c, s, -s, c, -c * x - s * y, s * x - c * y])


def polygons(geometry):
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 1e-12:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from polygons(part)


def largest_polygon(geometry):
    return max(polygons(geometry), key=lambda p: p.area, default=Polygon())


def radius(sofa):
    return float(np.linalg.norm(np.asarray(sofa.exterior.coords), axis=1).max())


def validate_route(route):
    route = np.asarray(route, dtype=float)
    if route.ndim != 2 or route.shape[1] != 3 or not 2 <= len(route) <= 100000:
        raise ValueError("A route needs 2-100,000 [x, y, angle in degrees] poses.")
    if not np.isfinite(route).all() or np.max(np.abs(route[:, :2])) > 10000:
        raise ValueError("Route positions must be finite and within +/-10,000.")
    if np.max(np.abs(route[:, 2])) > 36000:
        raise ValueError("Route angles exceed the supported range.")
    return route.copy()


def sample_route(route, distance=0.02, angle=1.0, cap=200000):
    route = validate_route(route)
    if not all(np.isfinite(v) and v > 0 for v in (distance, angle)):
        raise ValueError("Sampling steps must be positive.")
    changes = np.diff(route, axis=0)
    counts = np.maximum(
        1,
        np.maximum(
            np.ceil(np.linalg.norm(changes[:, :2], axis=1) / distance),
            np.ceil(np.abs(changes[:, 2]) / angle),
        ),
    ).astype(int)
    if counts.sum() + 1 > cap:
        raise ValueError(
            "Route requires too many samples; shorten it or increase the steps."
        )
    return np.vstack(
        [
            *[
                np.linspace(a, b, n, endpoint=False)
                for a, b, n in zip(route[:-1], route[1:], counts)
            ],
            route[-1:],
        ]
    )


def envelope(
    route,
    hallway=Hallway(),
    distance=0.10,
    angle=3.0,
    checkpoint=noop,
    cleanup_tolerance=0.0,
):
    route = validate_route(route)
    feasible = local_region(hallway.entrance, route[0]).intersection(
        local_region(hallway.exit, route[-1])
    )
    region = hallway.region
    for index, pose in enumerate(constraint_poses(route, distance, angle)):
        checkpoint()
        feasible = feasible.intersection(local_region(region, pose))
        if feasible.is_empty:
            return Polygon()
        if cleanup_tolerance and index % 128 == 127:
            feasible = feasible.simplify(cleanup_tolerance, preserve_topology=True)
    return largest_polygon(feasible)


def axial_translation(a, b):
    return abs(b[2] - a[2]) < 1e-12 and min(abs(b[0] - a[0]), abs(b[1] - a[1])) < 1e-12


def constraint_poses(route, distance, angle):
    yield route[0]
    for a, b in zip(route[:-1], route[1:]):
        if axial_translation(a, b):
            yield b
        else:
            yield from sample_route(np.array([a, b]), distance, angle)[1:]


def route_length(route, body_radius=1.0):
    delta = np.diff(np.asarray(route), axis=0)
    return float(
        np.sum(
            np.linalg.norm(delta[:, :2], axis=1)
            + body_radius * np.deg2rad(np.abs(delta[:, 2]))
        )
    )


def _translation_sweep(sofa, a, b):
    first, last = world_sofa(sofa, a), world_sofa(sofa, b)
    d = np.asarray(b[:2]) - a[:2]
    parts = [first, last]
    for ring in [first.exterior, *first.interiors]:
        v = np.asarray(ring.coords)
        for p, q in zip(v[:-1], v[1:]):
            strip = Polygon([p, q, q + d, p + d])
            if strip.area > 1e-16:
                parts.append(strip)
    return unary_union(parts)


def verify_motion(
    sofa, route, hallway=Hallway(), resolution=0.00001, max_depth=18, checkpoint=noop
):
    sofa, route = require_polygon(sofa), validate_route(route)
    if not np.isfinite(resolution) or resolution <= 0 or not 1 <= max_depth <= 24:
        raise ValueError("Invalid verification resolution or depth.")
    region, wall, r = hallway.region, hallway.region.boundary, radius(sofa)
    cache, collisions, unresolved = {}, [], 0
    max_outside, min_clearance, lower_clearance = 0.0, float("inf"), float("inf")

    def pose_info(q):
        nonlocal max_outside, min_clearance
        checkpoint()
        key = tuple(q)
        if key not in cache:
            moved = world_sofa(sofa, q)
            fits = region.covers(moved)
            outside = 0.0 if fits else float(moved.difference(region).area)
            clearance = float(moved.distance(wall)) if fits else 0.0
            max_outside = max(max_outside, outside)
            min_clearance = min(min_clearance, clearance)
            cache[key] = (fits, clearance)
            if not fits:
                collisions.append(list(map(float, q)))
        return cache[key]

    stack = [(a, b, 0) for a, b in zip(route[:-1], route[1:])]
    intervals = 0
    while stack:
        a, b, depth = stack.pop()
        fa, ca = pose_info(a)
        fb, cb = pose_info(b)
        intervals += 1
        if not fa or not fb:
            continue
        bound = float(np.linalg.norm(b[:2] - a[:2]) + r * np.deg2rad(abs(b[2] - a[2])))
        if min(ca, cb) > 0.5 * bound + 1e-10:
            lower_clearance = min(lower_clearance, min(ca, cb) - 0.5 * bound)
            continue
        if axial_translation(a, b):
            lower_clearance = min(lower_clearance, min(ca, cb))
            continue
        if abs(b[2] - a[2]) < 1e-12:
            swept = _translation_sweep(sofa, a, b)
            if region.covers(swept):
                lower_clearance = min(lower_clearance, float(swept.distance(wall)))
                continue
        middle = (a + b) / 2
        fm, cm = pose_info(middle)
        if not fm:
            continue
        if bound <= resolution or depth >= max_depth or intervals >= 200000:
            unresolved += 1
            lower_clearance = 0.0
        else:
            stack.extend([(a, middle, depth + 1), (middle, b, depth + 1)])
    entrance = hallway.entrance.covers(world_sofa(sofa, route[0]))
    exit_fit = hallway.exit.covers(world_sofa(sofa, route[-1]))
    sampled = not collisions and entrance and exit_fit
    passed = sampled and unresolved == 0
    return {
        "area": float(sofa.area),
        "samples_checked": len(cache),
        "colliding_samples": len(collisions),
        "max_outside_area": max_outside,
        "first_collision_pose": collisions[0] if collisions else None,
        "endpoints_in_separate_arms": bool(entrance and exit_fit),
        "sampled_feasible": bool(sampled),
        "motion_bound_pass": bool(passed),
        "unresolved_intervals": unresolved,
        "minimum_sampled_clearance": float(min_clearance),
        "clearance_lower_bound": (
            float(lower_clearance) if passed and np.isfinite(lower_clearance) else 0.0
        ),
        "resolution": resolution,
        "status": (
            "motion_bound_pass"
            if passed
            else ("unresolved_contact" if sampled else "collision")
        ),
        "limitation": "Floating-point polygon geometry; no exact-arithmetic certificate or global optimality guarantee.",
    }


def reconstruct(route, hallway=Hallway(), resolution=0.003, checkpoint=noop):
    route = validate_route(route)
    coarse = envelope(route, hallway, 0.05, 1.0, checkpoint)
    if coarse.is_empty:
        raise ValueError("This movement has no positive-area connected sofa.")
    r = radius(coarse)
    pieces = [route[:1]]
    maximum_motion = 0.0
    for a, b in zip(route[:-1], route[1:]):
        if axial_translation(a, b):
            pieces.append(np.array([b]))
            continue
        motion = np.linalg.norm(b[:2] - a[:2]) + r * np.deg2rad(abs(b[2] - a[2]))
        n = max(1, int(np.ceil(motion / resolution)))
        if n > 100000:
            raise ValueError("Movement requires excessive reconstruction samples.")
        maximum_motion = max(maximum_motion, motion / n)
        pieces.append(np.linspace(a, b, n + 1)[1:])
    poses = np.vstack(pieces)
    raw = envelope(
        poses,
        hallway,
        distance=resolution * 1.01,
        angle=np.rad2deg(resolution / max(r, 1e-8)) * 1.01,
        checkpoint=checkpoint,
        cleanup_tolerance=1e-10,
    )
    if raw.is_empty:
        raise ValueError("This movement has no positive-area connected sofa.")
    cleanup_tolerance = 1e-6
    margin = float(0.5 * maximum_motion + 1e-5)
    raw_area = float(raw.area)
    raw = raw.simplify(cleanup_tolerance, preserve_topology=True)
    sofa = largest_polygon(raw.buffer(-margin, join_style=2, mitre_limit=1e6))
    sofa = sofa.simplify(cleanup_tolerance, preserve_topology=True)
    sofa = require_polygon(sofa)
    report = verify_motion(sofa, route, hallway, resolution / 4, checkpoint=checkpoint)
    report.update(
        reconstruction_area=raw_area,
        inward_margin=margin,
        boundary_cleanup_tolerance=cleanup_tolerance,
    )
    return sofa, report


def load_json(path):
    path = Path(path)
    if path.stat().st_size > 25_000_000:
        raise ValueError("JSON file is too large (25 MB limit).")
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return make_sofa(data), None, Hallway(), {}
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object or a list of polygon vertices.")
    if data.get("type") == "Polygon":
        return require_polygon(shape(data)), None, Hallway(), {}
    if "geometry" in data:
        sofa = require_polygon(shape(data["geometry"]))
    elif "vertices" in data:
        sofa = make_sofa(data["vertices"], data.get("interior_rings", []))
    else:
        raise ValueError("JSON needs polygon geometry or vertices.")
    route = (
        validate_route(data["route_waypoints"]) if data.get("route_waypoints") else None
    )
    hallway = Hallway(float(data.get("hallway_length", 5.0)))
    return sofa, route, hallway, data


def result_payload(sofa, route, hallway, **details):
    from datetime import datetime, timezone

    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "geometry": mapping(require_polygon(sofa)),
        "vertices": list(sofa.exterior.coords),
        "interior_rings": [list(r.coords) for r in sofa.interiors],
        "route_waypoints": None if route is None else validate_route(route).tolist(),
        "hallway_width": 1.0,
        "hallway_length": hallway.length,
        "motion_interpolation": "linear x, y, unwrapped angle in degrees",
        **details,
    }


def save_json(path, data):
    path = Path(path)
    payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
