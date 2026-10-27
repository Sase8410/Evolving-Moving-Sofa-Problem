import numpy as np
from .geometry import HALLWAY, ENTRANCE, EXIT, world, radius_bound, straight_arm


def verify(sofa, parts, max_depth=18, dense_samples=257, checkpoint=lambda: None):
    report = {"status": "invalid", "area": float(sofa.area),
              "intervals_passed": 0, "unresolved_intervals": 0,
              "poses_checked": 0, "max_outside_area": 0.,
              "arithmetic": "floating point; not a formal certificate"}
    if sofa.is_empty or not sofa.is_valid or sofa.geom_type != "Polygon":
        return report
    report["endpoints_in_separate_arms"] = bool(
        ENTRANCE.covers(world(sofa, parts[0](0))) and EXIT.covers(world(sofa, parts[-1](1))))
    if not report["endpoints_in_separate_arms"]:
        report["status"] = "endpoint_failure"
        return report
    radius = radius_bound(sofa)
    for motion in parts:
        arm = straight_arm(motion)
        if arm is not None:
            checkpoint()
            if all(arm.covers(world(sofa, motion(t))) for t in [0., 1.]):
                report["intervals_passed"] += 1
                report["poses_checked"] += 2
                continue
        for t in np.linspace(0, 1, dense_samples):
            checkpoint()
            moved = world(sofa, motion(t))
            report["poses_checked"] += 1
            if not HALLWAY.covers(moved):
                report["status"] = "collision"
                report["max_outside_area"] = max(report["max_outside_area"],
                                                 moved.difference(HALLWAY).area)
                return report
        todo = [(a, b, 0) for a, b in zip(motion.times[:-1], motion.times[1:])]
        while todo:
            checkpoint()
            a, b, depth = todo.pop()
            m = (a+b)/2
            moved = world(sofa, motion(m))
            report["poses_checked"] += 1
            if not HALLWAY.covers(moved):
                report["status"] = "collision"
                report["max_outside_area"] = moved.difference(HALLWAY).area
                return report
            clearance = moved.distance(HALLWAY.boundary)
            bound = motion.speed_bound(a, b, radius)*(b-a)/2
            if clearance > bound+1e-10:
                report["intervals_passed"] += 1
            elif depth >= max_depth:
                report["unresolved_intervals"] += 1
                report["status"] = "inconclusive"
                return report
            else:
                todo.extend([(a, m, depth+1), (m, b, depth+1)])
    report["status"] = "bounded_numerical_pass"
    return report
