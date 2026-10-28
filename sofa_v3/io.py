import json
import os
import platform
import sys
import hashlib
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError


def provenance():
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root/"sofa_v3").glob("*.py")), root/"run_v3.py"]
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
