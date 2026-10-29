pass
import numpy as np
from shapely.geometry import Polygon
from .motion import Motion, full_motion
from .geometry import reconstruct_conservative
from .verify import verify

HAMMERSLEY_AREA = np.pi/2+2/np.pi
GERVER_AREA = 2.2195316688


def hammersley_motion(controls=65, r=2/np.pi):
    
    
    t = np.linspace(0, 1, controls)
    angle = t*np.pi/2
    return Motion(t, np.c_[-r*np.cos(angle), 1+r*np.sin(angle), angle])


def hammersley_polygon(points=1001, r=2/np.pi):
    pass
    a = np.linspace(np.pi, 1.5*np.pi, points)
    b = np.linspace(1.5*np.pi, 2*np.pi, points)
    c = np.linspace(0, -np.pi, 2*points)
    return Polygon(np.vstack([
        np.c_[-r+np.cos(a), np.sin(a)],
        np.c_[r+np.cos(b), np.sin(b)],
        np.c_[r*np.cos(c), r*np.sin(c)],
    ]))


def reference_check(target=.001, checkpoint=lambda: None):
    turn = hammersley_motion()
    parts = full_motion(turn)
    sofa, construction = reconstruct_conservative(parts, target, checkpoint)
    report = verify(sofa, parts, checkpoint=checkpoint)
    return {"reference": "Hammersley (known analytic motion, not discovered)",
            "analytic_area": float(HAMMERSLEY_AREA),
            "reconstructed_area": float(sofa.area),
            "validated_area": float(sofa.area) if report["status"] == "bounded_numerical_pass" else None,
            "area_gap": float(HAMMERSLEY_AREA-sofa.area),
            "construction": construction, "verification": report,
            "motion": turn.to_dict()}, sofa
