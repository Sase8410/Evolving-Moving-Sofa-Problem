pass
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import PathPatch, Polygon
from shapely.geometry import shape
from sofa_v3.io import load_result
from sofa_v3.motion import Motion
from sofa_v3.geometry import HALLWAY, world
from sofa_v3.gui import polygon_path


def render(source, output):
    data=load_result(source)
    if data.get("validated_area") is None:
        raise ValueError("Choose a result with a completed passing validation.")
    sofa=shape(data["geometry"])
    parts=[Motion.from_dict(p) for p in data["full_motion"]]
    poses=np.vstack([p(np.linspace(0,1,n,endpoint=True)) for p,n in zip(parts,[30,90,30])])
    if len(parts)!=3:
        poses=np.vstack([p(np.linspace(0,1,30)) for p in parts])
    fig,(ax,detail)=plt.subplots(1,2,figsize=(10,6),gridspec_kw={"width_ratios":[1.2,1]},layout="constrained")
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle("Moving Sofa V3.2 · discovered shape and motion",fontsize=16,weight="bold")
    ax.add_patch(Polygon(np.asarray(HALLWAY.exterior.coords),facecolor="#e2e8f0",edgecolor="#64748b"))
    ax.set(xlim=(-6,1.3),ylim=(-.3,6.5),aspect="equal",title="Entry → turn → exit")
    patch=PathPatch(polygon_path(world(sofa,poses[0])),facecolor="#0d9488",edgecolor="#115e59")
    ax.add_patch(patch)
    detail.add_patch(PathPatch(polygon_path(sofa),facecolor="#0d9488",edgecolor="#115e59"))
    a,b,c,d=sofa.bounds
    detail.set(xlim=(a-.15,c+.15),ylim=(b-.3,d+.3),aspect="equal",
               title=f"Validated area: {data['validated_area']:.6f}")
    fig.text(.78,.13,"Rigid geometry · continuous motion\nFloating-point numerical check; not a formal certificate",
             ha="center",fontsize=9)
    def update(i):
        patch.set_path(polygon_path(world(sofa,poses[i])))
        return patch,
    animation=FuncAnimation(fig,update,frames=len(poses),interval=50,blit=False)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    writer=PillowWriter(fps=20) if output.suffix.lower()==".gif" else FFMpegWriter(fps=20,codec="libx264",extra_args=["-pix_fmt","yuv420p"])
    animation.save(output,writer=writer,dpi=100)
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description='Render a saved result to an MP4 (requires ffmpeg) or animated GIF.\n\npython render_demo.py results/v32_release_study/convergence_best.json assets/demo.mp4\n')
    parser.add_argument("source",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args();render(args.source,args.output)
