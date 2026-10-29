pass
import numpy as np
from scipy.interpolate import CubicSpline, PPoly


class Motion:
    def __init__(self, times, poses, kind="cubic"):
        self.times = np.asarray(times, dtype=float)
        self.poses = np.asarray(poses, dtype=float)
        if (len(self.times) < 2 or self.poses.shape != (len(self.times), 3)
                or not np.isfinite(self.poses).all()
                or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times) <= 0)
                or self.times[0] != 0 or self.times[-1] != 1):
            raise ValueError("Motion requires finite poses and increasing times from 0 to 1.")
        if kind not in ("cubic", "linear"):
            raise ValueError("Unknown motion kind.")
        self.kind = kind
        if kind == "cubic":
            self.curve = CubicSpline(self.times, self.poses, bc_type="natural", extrapolate=False)
        else:
            slopes = np.diff(self.poses, axis=0)/np.diff(self.times)[:, None]
            self.curve = PPoly(np.array([slopes, self.poses[:-1]]), self.times,
                               extrapolate=False)
        self.derivative = self.curve.derivative()

    def __call__(self, t):
        
        
        
        value = self.curve(t)
        times = np.asarray(t)
        if times.ndim == 0:
            if times == 0:
                return self.poses[0].copy()
            if times == 1:
                return self.poses[-1].copy()
        else:
            value[times == 0] = self.poses[0]
            value[times == 1] = self.poses[-1]
        return value

    def speed_bound(self, a, b, radius):
        pass
        maxima = np.zeros(3)
        d = self.derivative
        first = max(0, np.searchsorted(self.times, a, side="right")-1)
        last = min(len(self.times)-2, np.searchsorted(self.times, b, side="left")-1)
        for i in range(first, last+1):
            left, right = max(a, self.times[i]), min(b, self.times[i+1])
            if right <= left:
                continue
            u, v = left-self.times[i], right-self.times[i]
            for j in range(3):
                c = d.c[:, i, j]
                values = [abs(np.polyval(c, u)), abs(np.polyval(c, v))]
                if len(c) == 3 and abs(c[0]) > 1e-30:
                    z = -c[1]/(2*c[0])
                    if u < z < v:
                        values.append(abs(np.polyval(c, z)))
                maxima[j] = max(maxima[j], *values)
        return float((np.hypot(*maxima[:2]) + radius*maxima[2])*(1+1e-12)+1e-12)

    def insert(self, t):
        times = np.unique(np.r_[self.times, t])
        return Motion(times, self(times), self.kind)

    def to_dict(self):
        return {"kind": self.kind, "times": self.times.tolist(),
                "poses_radians": self.poses.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(data["times"], data["poses_radians"], data["kind"])


def seed_motion(controls=5):
    pass
    t = np.linspace(0, 1, controls)
    return Motion(t, np.c_[np.zeros(controls), np.ones(controls), t*np.pi/2])


def full_motion(turn):
    pass
    p, q = turn(0), turn(1)
    if abs(p[2]) > 1e-10 or abs(q[2]-np.pi/2) > 1e-10:
        raise ValueError("Turn must start at 0 and end at pi/2.")
    return [Motion([0, 1], [[-4, p[1], 0], p], "linear"), turn,
            Motion([0, 1], [q, [q[0], 5, np.pi/2]], "linear")]


def continuous_parts(parts, tolerance=1e-10):
    pass
    if not parts:
        return False
    for a,b in zip(parts,parts[1:]):
        p,q=a(1),b(0)
        angle=np.arctan2(np.sin(p[2]-q[2]),np.cos(p[2]-q[2]))
        if np.max(abs(p[:2]-q[:2])) > tolerance or abs(angle) > tolerance:
            return False
    return True
