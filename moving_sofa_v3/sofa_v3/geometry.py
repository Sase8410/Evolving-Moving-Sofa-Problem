import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

LENGTH = 8.0
HALLWAY = unary_union([box(-LENGTH, 0, 1, 1), box(0, 0, 1, LENGTH)])
ENTRANCE = box(-LENGTH, 0, 0, 1)
EXIT = box(0, 1, 1, LENGTH)
SEARCH_BOX = box(-3, -2, 3, 1)
EPS = 1e-8


def straight_arm(motion):
    if motion.kind != "linear" or len(motion.times) != 2:
        return None
    p, q = motion.poses
    if p[2] == q[2] and p[2] == 0:
        return box(-LENGTH, 0, 1, 1)
    if p[2] == q[2] and p[2] == np.pi/2:
        return box(0, 0, 1, LENGTH)
    return None


def world(shape, pose):
    x, y, angle = pose
    c, s = np.cos(angle), np.sin(angle)
    return affine_transform(shape, [c, -s, s, c, x, y])


def local(region, pose):
    x, y, angle = pose
    c, s = np.cos(angle), np.sin(angle)
    return affine_transform(region, [c, s, -s, c, -c*x-s*y, s*x-c*y])


def polygons(g):
    if isinstance(g, Polygon):
        if not g.is_empty and g.area > 0:
            yield g
    elif hasattr(g, "geoms"):
        for part in g.geoms:
            yield from polygons(part)


def largest(g):
    return max(polygons(g), key=lambda x: x.area, default=Polygon())


def radius_bound(g):
    return max((float(np.linalg.norm(np.asarray(p.exterior.coords), axis=1).max())
                for p in polygons(g)), default=0.)


def inset_hallway(delta):
    if not 0 <= delta < .5:
        return Polygon()
    return unary_union([box(-LENGTH+delta, delta, 1-delta, 1-delta),
                        box(delta, delta, 1-delta, LENGTH-delta)])


def endpoint_region(parts, margin=0):
    return SEARCH_BOX.intersection(local(ENTRANCE.buffer(-margin), parts[0](0))).intersection(
        local(EXIT.buffer(-margin), parts[-1](1)))


def reconstruct(parts, samples=49, checkpoint=lambda: None, all_components=False):
    feasible = endpoint_region(parts)
    for motion in parts:
        times = np.unique(np.r_[motion.times, np.linspace(0, 1, samples)])
        if motion.kind == "linear" and len(motion.times) == 2:
            times = [0., 1.]
        for pose in motion(times):
            checkpoint()
            feasible = feasible.intersection(local(HALLWAY, pose))
            if feasible.is_empty:
                return Polygon()
    return feasible if all_components else largest(feasible)


def bounded_intervals(motion, radius, target, checkpoint=lambda: None):
    todo = list(zip(motion.times[:-1], motion.times[1:]))[::-1]
    while todo:
        checkpoint()
        a, b = todo.pop()
        bound = motion.speed_bound(a, b, radius)*(b-a)/2
        if bound > target:
            if b-a < 1e-12:
                raise RuntimeError("Cannot refine motion to requested tolerance.")
            m = (a+b)/2
            todo.extend([(m, b), (a, m)])
        else:
            yield a, b, bound


def reconstruct_conservative(parts, target=.002, checkpoint=lambda: None):
    if not 1e-6 <= target <= .05:
        raise ValueError("Reconstruction target must be between 1e-6 and .05.")
    feasible = reconstruct(parts, 97, checkpoint, all_components=True)
    radius = radius_bound(feasible)
    feasible = feasible.intersection(endpoint_region(parts, EPS))
    count = 0
    for motion in parts:
        arm = straight_arm(motion)
        if arm is not None:
            for t in [0., 1.]:
                checkpoint()
                feasible = feasible.intersection(local(arm.buffer(-EPS), motion(t)))
            count += 1
            continue
        for a, b, bound in bounded_intervals(motion, radius, target, checkpoint):
            feasible = feasible.intersection(local(inset_hallway(bound+EPS), motion((a+b)/2)))
            count += 1
            if feasible.is_empty:
                break
        if feasible.is_empty:
            break
    return largest(feasible), {"intervals": count, "target_displacement": target,
                              "radius_bound": radius, "roundoff_margin": EPS}


def contact_refinement_time(turn, sofa):
    scores = []
    for a, b in zip(turn.times[:-1], turn.times[1:]):
        ts = np.linspace(a, b, 7)[1:-1]
        losses = [sofa.difference(local(HALLWAY, turn(t))).area for t in ts]
        mid = (a+b)/2
        curvature = np.linalg.norm(turn(mid)-(turn(a)+turn(b))/2)
        scores.append((max(losses)+1e-3*curvature+1e-6*(b-a), mid))
    return max(scores)[1]
