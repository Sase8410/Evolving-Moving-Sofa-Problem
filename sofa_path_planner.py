from dataclasses import asdict, dataclass
import time

import numpy as np

from sofa_geometry import (
    Hallway,
    _translation_sweep,
    axial_translation,
    noop,
    radius,
    require_polygon,
    result_payload,
    route_length,
    validate_route,
    verify_motion,
    world_sofa,
)


@dataclass(frozen=True)
class PlannerSettings:
    seconds: float = 30.0
    iterations: int = 12000
    seed: int = 42
    step: float = 0.20
    shortcut_attempts: int = 80
    solution_limit: int = 8


class BudgetExceeded(Exception):
    pass


def angle_delta(a, b):
    return (b - a + 180.0) % 360.0 - 180.0


def delta_pose(a, b):
    delta = np.asarray(b) - a
    delta[2] = angle_delta(a[2], b[2])
    return delta


def unwrap_route(route):
    q = np.asarray(route, dtype=float).copy()
    q[:, 2] = np.rad2deg(np.unwrap(np.deg2rad(q[:, 2])))
    return q


class CollisionOracle:
    def __init__(self, sofa, hallway, checkpoint=noop):
        self.sofa, self.region, self.checkpoint = sofa, hallway.region, checkpoint
        self.wall, self.r = self.region.boundary, radius(sofa)
        self.cache = {}

    def state(self, q):
        self.checkpoint()
        key = tuple(q)
        if key not in self.cache:
            body = world_sofa(self.sofa, q)
            fits = self.region.covers(body)
            self.cache[key] = (fits, float(body.distance(self.wall)) if fits else 0.0)
            if len(self.cache) > 30000:
                value = self.cache[key]
                self.cache.clear()
                self.cache[key] = value
        return self.cache[key]

    def edge(self, a, b):
        b = np.asarray(a) + delta_pose(a, b)
        stack = [(np.asarray(a), b, 0)]
        while stack:
            p, q, depth = stack.pop()
            fp, cp = self.state(p)
            fq, cq = self.state(q)
            if not fp or not fq:
                return False
            bound = np.linalg.norm(q[:2] - p[:2]) + self.r * np.deg2rad(
                abs(q[2] - p[2])
            )
            if min(cp, cq) > 0.5 * bound + 1e-10:
                continue
            if axial_translation(p, q):
                continue
            if abs(q[2] - p[2]) < 1e-12 and self.region.covers(
                _translation_sweep(self.sofa, p, q)
            ):
                continue
            if depth >= 14 or bound < 2e-5:
                return False
            middle = (p + q) / 2
            if not self.state(middle)[0]:
                return False
            stack.extend([(p, middle, depth + 1), (middle, q, depth + 1)])
        return True


class Tree:
    def __init__(self, roots, capacity):
        self.q = np.empty((capacity, 3))
        self.parents = np.empty(capacity, dtype=int)
        self.n = len(roots)
        self.q[: self.n] = roots
        self.parents[: self.n] = -1

    def add(self, q, parent):
        if self.n >= len(self.q):
            return None
        index = self.n
        self.n += 1
        self.q[index] = q
        self.parents[index] = parent
        return index

    def nearest(self, q, r):
        d = self.q[: self.n] - q
        d[:, 2] = np.deg2rad((d[:, 2] + 180) % 360 - 180) * r
        return int(np.argmin(np.sum(d * d, axis=1)))

    def branch(self, index):
        branch = []
        while index >= 0:
            branch.append(self.q[index].copy())
            index = self.parents[index]
        return branch[::-1]


def endpoint_poses(sofa, hallway):
    v = np.asarray(sofa.convex_hull.exterior.coords)
    edges = np.diff(v, axis=0)
    if len(edges) > 96:
        edges = edges[np.linspace(0, len(edges) - 1, 96, dtype=int)]
    rectangle = np.asarray(sofa.minimum_rotated_rectangle.exterior.coords)
    edges = np.vstack((edges, np.diff(rectangle, axis=0)))
    angles = list(np.arange(-180.0, 180.0, 10.0))
    angles.extend(-np.rad2deg(np.arctan2(edges[:, 1], edges[:, 0])))
    angles.extend(np.asarray(angles[:]) + 90.0)
    angles = np.unique(np.round((np.asarray(angles) + 180) % 360 - 180, 8))
    starts, goals = [], []
    for angle in angles:
        turned = world_sofa(sofa, [0, 0, angle])
        x0, y0, x1, y1 = turned.bounds
        start = np.array([-0.3 - x1, 0.5 - (y0 + y1) / 2, angle])
        goal = np.array([0.5 - (x0 + x1) / 2, 1.3 - y0, angle])
        if y1 - y0 <= 1 and hallway.entrance.covers(world_sofa(sofa, start)):
            starts.append(start)
        if x1 - x0 <= 1 and hallway.exit.covers(world_sofa(sofa, goal)):
            goals.append(goal)
    return starts, goals


def _shortcut(route, oracle, rng, attempts):
    route = unwrap_route(route)
    for _ in range(attempts):
        oracle.checkpoint()
        if len(route) < 3:
            break
        a, b = sorted(rng.choice(len(route), 2, replace=False))
        if b - a < 2:
            continue
        if oracle.edge(route[a], route[b]):
            candidate = unwrap_route(np.vstack((route[: a + 1], route[b:])))
            if (
                route_length(candidate, oracle.r)
                < route_length(route, oracle.r) - 1e-10
            ):
                route = candidate
    return route


def plan_path(
    sofa,
    settings=PlannerSettings(),
    hallway=None,
    initial_route=None,
    checkpoint=noop,
    progress=lambda **kw: None,
):
    sofa = require_polygon(sofa)
    if (
        not np.isfinite(settings.seconds)
        or settings.seconds <= 0
        or settings.iterations < 1
        or not np.isfinite(settings.step)
        or settings.step <= 0
        or settings.solution_limit < 1
        or settings.shortcut_attempts < 0
    ):
        raise ValueError("Invalid planner budget or step size.")
    r = radius(sofa)
    hallway = hallway or Hallway(max(5.0, 2 * r + 3.0))
    rng = np.random.default_rng(settings.seed)
    started = time.monotonic()
    deadline = started + settings.seconds
    candidates = []
    history = []
    iteration = 0

    def check_budget():
        checkpoint()
        if time.monotonic() > deadline:
            raise BudgetExceeded

    oracle = CollisionOracle(sofa, hallway, check_budget)

    def remember(route):
        report = verify_motion(sofa, route, hallway, checkpoint=checkpoint)
        if not report["motion_bound_pass"]:
            return
        candidates.append(
            (
                report["clearance_lower_bound"],
                -route_length(route, r),
                route.copy(),
                report,
            )
        )
        best = max(candidates, key=lambda c: c[:2])
        history.append(
            {
                "iteration": iteration,
                "solutions": len(candidates),
                "clearance_lower_bound": best[0],
                "weighted_travel": -best[1],
            }
        )
        progress(
            kind="candidate",
            sofa=sofa,
            route=best[2],
            report=best[3],
            message=f"Found {len(candidates)} checked route(s); refining movement.",
            history=history.copy(),
        )

    starts, goals = endpoint_poses(sofa, hallway)
    if initial_route is not None:
        initial_route = validate_route(initial_route)
        if hallway.entrance.covers(world_sofa(sofa, initial_route[0])):
            starts.append(initial_route[0])
        if hallway.exit.covers(world_sofa(sofa, initial_route[-1])):
            goals.append(initial_route[-1])
    if not starts or not goals:
        return {
            "success": False,
            "message": "No fitting entrance/exit orientation found. The shape was not changed.",
            "hallway_length": hallway.length,
            "iterations": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
    try:
        if initial_route is not None:
            report = verify_motion(
                sofa, initial_route, hallway, checkpoint=check_budget
            )
            if report["motion_bound_pass"]:
                remember(initial_route)
                remember(
                    _shortcut(initial_route, oracle, rng, settings.shortcut_attempts)
                )
        capacity = settings.iterations * 4 + len(starts) + len(goals) + 100
        left, right = Tree(starts, capacity), Tree(goals, capacity)
        active_is_start = True
        all_roots = np.vstack((starts, goals))
        xy_low = np.min(all_roots[:, :2], axis=0) - 0.25
        xy_high = np.max(all_roots[:, :2], axis=0) + 0.25

        def extend(tree, target):
            j = tree.nearest(target, r)
            d = delta_pose(tree.q[j], target)
            size = np.linalg.norm(d[:2]) + r * np.deg2rad(abs(d[2]))
            if size < 1e-10:
                return j, True
            q = tree.q[j] + d * min(1.0, settings.step / size)
            q[2] = (q[2] + 180) % 360 - 180
            if not oracle.edge(tree.q[j], q):
                return None, False
            return tree.add(q, j), size <= settings.step

        for iteration in range(1, settings.iterations + 1):
            check_budget()
            if len(candidates) >= settings.solution_limit:
                break
            if rng.random() < 0.16:
                target = right.q[rng.integers(right.n)].copy()
            elif rng.random() < 0.70:
                target = np.array(
                    [*rng.uniform([-0.7, 0.0], [1.0, 1.7]), rng.uniform(-180, 180)]
                )
            else:
                target = np.array(
                    [*rng.uniform(xy_low, xy_high), rng.uniform(-180, 180)]
                )
            index, _ = extend(left, target)
            if index is not None:
                for _ in range(50):
                    other, reached = extend(right, left.q[index])
                    if other is None:
                        break
                    if reached:
                        a, b = left.branch(index), right.branch(other)
                        route = np.asarray(a + list(reversed(b))[1:])
                        if not active_is_start:
                            route = route[::-1]
                        route = unwrap_route(route)
                        remember(route)
                        try:
                            remember(
                                _shortcut(
                                    route, oracle, rng, settings.shortcut_attempts
                                )
                            )
                        except BudgetExceeded:
                            pass
                        break
            left, right = right, left
            active_is_start = not active_is_start
            if iteration % 100 == 0:
                progress(
                    kind="status",
                    message=f"Path search: {iteration} attempts, {left.n+right.n} states.",
                )
    except BudgetExceeded:
        pass
    if not candidates:
        return {
            "success": False,
            "message": "No valid path found within this search. This does not prove the shape cannot fit.",
            "hallway_length": hallway.length,
            "iterations": iteration,
            "elapsed_seconds": time.monotonic() - started,
        }
    clearance, neg_length, route, report = max(candidates, key=lambda c: c[:2])
    return result_payload(
        sofa,
        route,
        hallway,
        mode="fixed_shape",
        success=True,
        verification_report=report,
        settings=asdict(settings),
        history=history,
        iterations=iteration,
        elapsed_seconds=time.monotonic() - started,
        objective="Find feasibility; compare conservative clearance lower bound, then weighted travel.",
        weighted_travel=-neg_length,
        limitation="Best found within a finite search; no global path-optimality or infeasibility proof. Shape unchanged.",
    )
