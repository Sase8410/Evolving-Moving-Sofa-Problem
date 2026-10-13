from dataclasses import asdict, dataclass
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from sofa_geometry import (
    Hallway,
    envelope,
    noop,
    reconstruct,
    result_payload,
    validate_route,
)


@dataclass(frozen=True)
class DiscoverySettings:
    seconds: float = 90.0
    evaluations: int = 6000
    population: int = 10
    restarts: int = 3
    seed: int = 42
    control_counts: tuple = (5, 9, 17)
    reconstruction_step: float = 0.002


class StageFinished(Exception):
    pass


def controls_to_route(controls, hallway=Hallway(7.0)):

    controls = np.asarray(controls, dtype=float)
    n = len(controls)
    times = np.linspace(0, 1, n)
    p = CubicSpline(times, controls, axis=0, bc_type="natural")(
        np.linspace(0, 1, 8 * (n - 1) + 1)
    )

    p[0, 1:] = [0.5, 0.0]
    p[-1, [0, 2]] = [0.5, 90.0]
    return np.vstack(
        ([-hallway.length + 2.0, 0.5, 0.0], p, [0.5, hallway.length - 2.0, 90.0])
    )


def initial_controls(n, rng, index):
    t = np.linspace(0, 1, n)
    if index == 0:
        xy = np.column_stack((0.5 * t, 0.5 + t))
    elif index == 1:
        xy = np.full((n, 2), 0.5)
    else:

        xy = np.column_stack(
            (
                rng.uniform(-0.4, 0.4) * (1 - t) + 0.5 * t,
                0.5 * (1 - t) + rng.uniform(0.7, 1.8) * t,
            )
        )
        xy += np.sin(np.pi * t)[:, None] * rng.normal(0, 0.22, (n, 2))
    a = 90 * t
    if index > 1:
        a += np.sin(np.pi * t) * rng.normal(0, 12, n)
    c = np.column_stack((xy, a))
    c[0, 1:] = [0.5, 0.0]
    c[-1, [0, 2]] = [0.5, 90.0]
    return c


def control_layout(n):
    mask = np.ones((n, 3), dtype=bool)
    mask[0, 1:] = False
    mask[-1, [0, 2]] = False
    lower = np.tile([-2.0, -1.0, -60.0], (n, 1))
    upper = np.tile([2.0, 3.0, 150.0], (n, 1))
    return mask, lower[mask], upper[mask]


def refine_controls(controls, n):
    c = CubicSpline(
        np.linspace(0, 1, len(controls)), controls, axis=0, bc_type="natural"
    )(np.linspace(0, 1, n))
    c[0, 1:] = [0.5, 0.0]
    c[-1, [0, 2]] = [0.5, 90.0]
    return c


def controls_from_route(route, n):
    route = validate_route(route)
    if len(route) < 4 or abs(route[0, 2]) > 1e-6 or abs(route[-1, 2] - 90) > 1e-6:
        raise ValueError(
            "Discovery resume needs a 0-to-90 degree route; use Find path for other rotations."
        )
    turn = route[1:-1].copy()
    if abs(turn[0, 2]) > 1e-6 or abs(turn[-1, 2] - 90) > 1e-6:
        raise ValueError("Discovery resume needs straight entrance/exit segments.")

    shift = 0.5 - turn[0, 1]
    rad = np.deg2rad(turn[:, 2])
    turn[:, :2] += np.column_stack((-np.sin(rad) * shift, np.cos(rad) * shift))
    if abs(turn[-1, 0] - 0.5) > 2e-4:
        raise ValueError(
            "This route uses different entrance/exit offsets. Start a new discovery or use Find path."
        )
    return refine_controls(turn, n)


def discover(
    settings=DiscoverySettings(),
    initial_route=None,
    checkpoint=noop,
    progress=lambda **kw: None,
):
    counts = tuple(settings.control_counts)
    if (
        not counts
        or any(int(n) != n or not 3 <= n <= 33 for n in counts)
        or any(a >= b for a, b in zip(counts, counts[1:]))
        or settings.population < 4
        or settings.restarts < 1
        or settings.evaluations < 20
        or not np.isfinite(settings.seconds)
        or settings.seconds <= 0
        or not 0.0001 <= settings.reconstruction_step <= 0.02
    ):
        raise ValueError("Invalid discovery settings.")
    rng = np.random.default_rng(settings.seed)
    hallway = Hallway(7.0)
    started = time.monotonic()
    deadline = started + settings.seconds
    history = []
    archive = []
    champions = []
    evaluations = 0
    global_best = -1.0
    controls = None
    last_emit = 0.0
    stage_count = len(counts)
    if initial_route is not None:
        initial_route = validate_route(initial_route)
        controls_from_route(initial_route, counts[0])
        score = envelope(initial_route, hallway, checkpoint=checkpoint).area
        if score > 0:
            champions.append(
                (
                    float(score),
                    initial_route.copy(),
                    controls_from_route(initial_route, counts[0]),
                )
            )

    for stage, n in enumerate(counts):
        checkpoint()
        if stage and (
            time.monotonic() > deadline or evaluations >= settings.evaluations
        ):
            break
        stage_deadline = started + settings.seconds * (stage + 1) / stage_count
        stage_limit = int(settings.evaluations * (stage + 1) / stage_count)
        mask, lower, upper = control_layout(n)
        template = initial_controls(n, rng, 0)
        seeds = [initial_controls(n, rng, i) for i in range(settings.population)]
        if controls is not None:
            seeds[0] = refine_controls(controls, n)
            for i in range(1, settings.population // 2):
                seeds[i] = seeds[0] + rng.normal(0, [0.025, 0.025, 1.5], (n, 3))
        elif initial_route is not None:
            seeds[0] = controls_from_route(initial_route, n)
        vectors = [np.clip(c[mask], lower, upper) for c in seeds]
        scores = []
        stage_best = [-1.0, vectors[0].copy()]
        cache = {}

        def decode(v):
            c = template.copy()
            c[mask] = v
            return c

        def objective(v):
            nonlocal evaluations, global_best, last_emit
            checkpoint()
            if (
                evaluations >= stage_limit or time.monotonic() > stage_deadline
            ) and evaluations > 0:
                raise StageFinished
            key = tuple(v)
            if key in cache:
                return -cache[key]
            controls_here = decode(v)
            route = controls_to_route(controls_here, hallway)

            angle = (3.0, 1.5, 0.75)[min(stage, 2)]
            sofa = envelope(
                route,
                hallway,
                distance=0.14 / (stage + 1),
                angle=angle,
                checkpoint=checkpoint,
            )
            area = float(sofa.area)
            evaluations += 1
            cache[key] = area
            if area > stage_best[0]:
                stage_best[:] = [area, v.copy()]
            if area > global_best or (evaluations % 100 == 0):
                global_best = max(global_best, area)
                row = {
                    "evaluation": evaluations,
                    "stage": stage + 1,
                    "control_points": n,
                    "stage_best_sampled_area": stage_best[0],
                    "search_angle_degrees": angle,
                }
                history.append(row)
            if area > 0 and (
                not archive or area > min(a[0] for a in archive) or len(archive) < 5
            ):
                if not any(np.array_equal(route, a[1]) for a in archive):
                    archive.append((area, route.copy(), controls_here.copy()))
                    archive.sort(key=lambda a: -a[0])
                    archive[:] = archive[:5]
            now = time.monotonic()
            if now - last_emit > 0.7 and stage_best[0] > 0:
                last_emit = now
                best_route = controls_to_route(decode(stage_best[1]), hallway)
                best_sofa = envelope(
                    best_route,
                    hallway,
                    distance=0.14 / (stage + 1),
                    angle=angle,
                    checkpoint=checkpoint,
                )
                progress(
                    kind="candidate",
                    sofa=best_sofa,
                    route=best_route,
                    report=None,
                    history=history.copy(),
                    message=f"Discovery stage {stage+1}/{stage_count}: {n} controls, {evaluations} evaluations. Candidate awaiting checks.",
                )
            return -area

        try:
            scores = [-objective(v) for v in vectors]

            for generation in range(8):
                order = np.argsort(scores)[::-1]
                vectors = [vectors[i] for i in order]
                scores = [scores[i] for i in order]
                for i in range(2, len(vectors)):
                    a, b, c = rng.choice(len(vectors), 3, replace=False)
                    child = vectors[a] + rng.uniform(0.3, 0.8) * (
                        vectors[b] - vectors[c]
                    )
                    blend = rng.random(len(child)) < 0.7
                    child = np.where(blend, child, vectors[i])
                    noise = np.tile([0.015, 0.015, 0.6], (n, 1))[mask]
                    child = np.clip(
                        child + rng.normal(size=len(child)) * noise, lower, upper
                    )
                    score = -objective(child)
                    if score > scores[i]:
                        vectors[i], scores[i] = child, score
            order = np.argsort(scores)[::-1]

            for restart in range(min(settings.restarts, len(vectors))):
                remaining = max(
                    30,
                    (stage_limit - evaluations) // max(1, settings.restarts - restart),
                )
                seed = stage_best[1] if restart == 0 else vectors[order[restart]]
                minimize(
                    objective,
                    seed,
                    method="Powell",
                    bounds=list(zip(lower, upper)),
                    options={"maxfev": remaining, "xtol": 0.0004, "ftol": 1e-7},
                )
        except StageFinished:
            pass
        controls = decode(stage_best[1])
        if stage_best[0] > 0:

            champion = (
                stage_best[0],
                controls_to_route(controls, hallway),
                controls.copy(),
            )
            champions.append(champion)
            archive.append(champion)

    if not archive:
        raise ValueError("No positive-area candidate found in this search budget.")
    progress(
        kind="status",
        message="Reconstructing finalists and checking their full movements...",
    )
    finalists = []
    attempts = []

    unique = []
    for item in list(reversed(champions)) + archive:
        if not any(np.array_equal(item[1], a[1]) for a in unique):
            unique.append(item)
    for score, route, c in unique[:4]:
        checkpoint()
        try:
            sofa, report = reconstruct(
                route, hallway, settings.reconstruction_step, checkpoint
            )
        except ValueError as error:
            attempts.append({"search_area": score, "error": str(error)})
            continue
        attempts.append({"search_area": score, "verification_report": report})
        if report["motion_bound_pass"]:
            finalists.append((float(sofa.area), sofa, route, c, report, score))
    if not finalists:
        return {
            "success": False,
            "message": "Search produced candidates, but none passed the final motion check.",
            "history": history,
            "finalist_checks": attempts,
            "evaluations": evaluations,
        }
    area, sofa, route, controls, report, score = max(finalists, key=lambda a: a[0])
    return result_payload(
        sofa,
        route,
        hallway,
        mode="discovery",
        success=True,
        settings=asdict(settings),
        controls=controls.tolist(),
        search_area=score,
        verification_report=report,
        history=history,
        finalist_checks=attempts,
        evaluations=evaluations,
        elapsed_seconds=time.monotonic() - started,
        initialization=(
            "imported route"
            if initial_route is not None
            else "generic and randomized movements"
        ),
        objective="Maximize area of a connected rigid polygon completing the hallway turn.",
        limitation="Finite control family and search budget; sampled search scores are approximate. No Gerver geometry, target-shape loss, or guaranteed convergence. Final polygon includes a reported inward margin.",
    )
