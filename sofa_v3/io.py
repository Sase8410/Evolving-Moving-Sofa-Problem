pass
import json
import os
import platform
import sys
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


def provenance():
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root/"sofa_v3").glob("*.py")), root/"run_v3.py", root/"audit_export.py"]
    legacy = root/"legacy_v1"/"sofa_freeform_optimizer_gui.py"
    if legacy.exists():
        files.append(legacy)
    packages = {}
    for name in ["numpy", "scipy", "shapely", "matplotlib", "scikit-learn"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"python": sys.version, "platform": platform.platform(),
            "packages": packages,
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in files}}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(temp, path)


def load_result(path):
    pass
    from shapely.geometry import shape
    from .motion import Motion, full_motion, continuous_parts
    import math
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    sofa = shape(data["geometry"])
    turn = Motion.from_dict(data["motion"])
    parts = [Motion.from_dict(p) for p in data.get("full_motion", [])] or full_motion(turn)
    if not continuous_parts(parts):
        raise ValueError("Saved motion segments are not continuous.")
    if sofa.geom_type != "Polygon" or (not sofa.is_empty and not sofa.is_valid):
        raise ValueError("Result must contain a valid polygon.")
    area = data.get("validated_area")
    if area is not None and (not math.isfinite(area) or area <= 0 or
                            abs(area-sofa.area) > 1e-8*max(1.,area)):
        raise ValueError("Saved validated area disagrees with the geometry.")
    data["full_motion"] = [p.to_dict() for p in parts]
    return data
