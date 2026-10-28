import importlib.util
from pathlib import Path
from time import perf_counter
import numpy as np
from .motion import Motion
from .search import finalize


def legacy_module():
    path = Path(__file__).resolve().parents[1]/"legacy_v1"/"sofa_freeform_optimizer_gui.py"
    spec = importlib.util.spec_from_file_location("attached_v1_freeform", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def route_motion(route):
    poses = route.copy()
    poses[:, 2] = np.deg2rad(poses[:, 2])
    return Motion(np.linspace(0, 1, len(route)), poses, "linear")


def split_route(motion):
    return [Motion([0, 1], [a, b], "linear")
            for a, b in zip(motion.poses[:-1], motion.poses[1:])]


def optimize_v1(cfg, progress=lambda row: None, checkpoint=lambda: None):
    cfg.validate()
    old = legacy_module()
    started = perf_counter()
    route = old.refine_waypoints(old.validate_route(old.REFERENCE_ROUTE))
    initial = route.copy()
    rng = np.random.default_rng(cfg.seed)
    best = old.build_sofa(route, checkpoint=checkpoint)
    archive = [(best.area, route.copy())]
    history = []
    iterations = cfg.budget-1
    for i in range(cfg.budget):
        checkpoint()
        if i:
            child = route.copy()
            j = int(rng.integers(1, len(route)-1))
            scale = .07*(1-i/iterations)+.008
            delta = rng.normal(0, scale, 2)
            child[j, :2] += delta
            if j+1 < len(route)-1 and rng.random() < .4:
                child[j+1, :2] += .5*delta
            child[1:-1, :2] = np.clip(child[1:-1, :2],
                initial[1:-1, :2]-.6, initial[1:-1, :2]+.6)
            candidate = old.build_sofa(child, checkpoint=checkpoint)
            candidate_area = candidate.area
            if candidate.area > best.area:
                best, route = candidate, child
                archive.append((best.area, route.copy()))
                archive = sorted(archive, key=lambda p: -p[0])[:cfg.finalists]
        else:
            candidate_area = best.area
        row = {"evaluation": i+1, "best_search_area": float(best.area),
               "candidate_area": float(candidate_area), "seconds": perf_counter()-started,
               "stage": "legacy_1plus1", "controls": len(route)}
        history.append(row)
        if (i+1) % 10 == 0 or i == 0:
            progress(row)
    archive = [(area, route_motion(r)) for area, r in archive]
    return finalize(archive, cfg, history, started, checkpoint,
                    "V1-attached-Step8-common-validation", parts_fn=split_route,
                    extra={"source": "legacy_v1/sofa_freeform_optimizer_gui.py",
                           "original_search_distance": .06, "original_search_angle_degrees": 3.,
                           "validation_changed": True})
