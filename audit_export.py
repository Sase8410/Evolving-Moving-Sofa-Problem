pass
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from shapely.geometry import shape, box
from shapely.ops import unary_union
from shapely.affinity import rotate, translate
from sofa_v3.io import save_json


def audit(file, samples=5001, checkpoint=lambda: None):
    if samples < 2:
        raise ValueError("Use at least two audit samples.")
    result=json.loads(Path(file).read_text())
    sofa=shape(result["geometry"])
    hallway=unary_union([box(-8,0,1,1),box(0,0,1,8)])
    entrance,exit=box(-8,0,0,1),box(0,1,1,8)
    count=collisions=0
    outside=0.
    endpoints=[]
    continuous=True
    previous=None
    for part in result["full_motion"]:
        poses=np.asarray(part["poses_radians"])
        if previous is not None:
            angle=np.arctan2(np.sin(previous[2]-poses[0,2]),np.cos(previous[2]-poses[0,2]))
            continuous=continuous and bool(np.max(abs(previous[:2]-poses[0,:2]))<=1e-10 and abs(angle)<=1e-10)
        previous=poses[-1]
        times=np.asarray(part["times"])
        curve=(CubicSpline(times,poses,bc_type="natural") if part["kind"]=="cubic"
               else interp1d(times,poses,axis=0))
        grid=np.unique(np.r_[0.,1.,(np.arange(samples)+.371)/samples])
        for t in grid:
            checkpoint()
            x,y,angle=curve(t)
            moved=translate(rotate(sofa,float(angle),origin=(0,0),use_radians=True),xoff=x,yoff=y)
            count+=1
            if not hallway.covers(moved):
                collisions+=1
                outside=max(outside,moved.difference(hallway).area)
            if t==0: endpoints.append(moved)
            if t==1: endpoints.append(moved)
    endpoint_fit=bool(entrance.covers(endpoints[0]) and exit.covers(endpoints[-1]))
    report={"source_file":str(file),"area":float(sofa.area),"samples_checked":count,
            "colliding_samples":collisions,"max_outside_area":outside,
            "endpoints_in_separate_arms":endpoint_fit,
            "continuous_segments":continuous,
            "sampled_audit_passed":bool(collisions==0 and endpoint_fit and continuous and sofa.is_valid and not sofa.is_empty and sofa.geom_type=="Polygon"),
            "method":"Separate SciPy interpolators and shapely.affinity rotate/translate; shifted time grid.",
            "limitation":"Additional floating-point sampled audit, not a formal certificate."}
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Independent dense audit of a saved result using Shapely affinity transforms.\n\nThis is an additional sampled check, not a replacement for bounded validation.\nExample: python audit_export.py results/discovery_4000_seed42.json\n')
    parser.add_argument("file",type=Path)
    parser.add_argument("--samples",type=int,default=5001)
    args=parser.parse_args()
    if args.samples<2: parser.error("Use at least two samples.")
    report=audit(args.file,args.samples)
    save_json(args.file.with_name(args.file.stem+"_audit.json"),report)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["sampled_audit_passed"] else 1)
