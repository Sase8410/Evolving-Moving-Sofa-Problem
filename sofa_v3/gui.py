import json
import queue
import threading
import hashlib
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch, Polygon as Patch
from matplotlib.widgets import Button, TextBox, CheckButtons
from shapely.geometry import shape, mapping
from shapely.geometry.polygon import orient
from .geometry import HALLWAY, world
from .motion import Motion, full_motion
from .search import Config, optimize
from .reference import reference_check
from .io import save_json, provenance


def polygon_path(poly):
    if poly.is_empty:
        return PlotPath(np.zeros((1, 2)), [PlotPath.MOVETO])
    vertices, codes = [], []
    for ring in [orient(poly).exterior, *orient(poly).interiors]:
        coords = np.asarray(ring.coords)
        vertices.extend(coords)
        codes.extend([PlotPath.MOVETO]+[PlotPath.LINETO]*(len(coords)-2)+[PlotPath.CLOSEPOLY])
    return PlotPath(vertices, codes)


class Window:
    def __init__(self):
        self.events, self.stop = queue.Queue(), threading.Event()
        self.worker = None
        self.result = None
        self.playing = False
        self.frame = 0
        self.frames = None
        self.fig = plt.figure(figsize=(13, 9), facecolor="#f8fafc")
        self.fig.canvas.manager.set_window_title("Moving Sofa V3.1 — Neural Search")
        self.fig.text(.055, .95, "Moving Sofa V3.1 · Neural Search", fontsize=22, weight="bold", color="#0f172a")
        self.fig.text(.055, .915, "Learn from simulations · rank motion proposals · validate with geometry", fontsize=11)
        self.ax = self.fig.add_axes([.045, .39, .46, .49])
        self.ax.add_patch(Patch(np.asarray(HALLWAY.exterior.coords), fc="#e2e8f0", ec="#64748b"))
        self.ax.set(xlim=(-5.5, 1.5), ylim=(-.5, 6), title="Rigid motion through the hallway")
        self.ax.set_aspect("equal")
        self.local = self.fig.add_axes([.56, .64, .39, .23])
        self.local.set_aspect("equal")
        self.local.set_title("Final boundary")
        self.plot = self.fig.add_axes([.59, .43, .35, .13])
        self.plot.set(xlabel="Evaluation", ylabel="Sampled area")
        self.plot.grid(alpha=.2)
        self.status = self.fig.text(.055, .335, "Ready. Train + search NN collects data, learns, then guides proposals.", fontsize=10)
        self.nn_status = self.fig.text(.055, .295, "NN: waiting for a run. Predictions are not collision checks.", fontsize=10, color="#115e59")
        self.metrics = self.fig.text(.56, .585, "No result yet.", fontsize=10)
        self.fig.text(.055, .025, "Experimental numerical results. Floating-point validation is not a formal certificate.", fontsize=9, color="#475569")
        self.budget = TextBox(self.fig.add_axes([.13, .22, .09, .045]), "Budget ", initial="1500")
        self.seed = TextBox(self.fig.add_axes([.31, .22, .07, .045]), "Seed ", initial="0")
        self.seeds = TextBox(self.fig.add_axes([.52, .22, .06, .045]), "Compare seeds ", initial="30")
        self.reuse = CheckButtons(self.fig.add_axes([.65, .215, .30, .055]), ["Reuse saved NN learning"], [True])
        self.buttons = []
        actions = [("Train + search NN", lambda _: self.start("neural")),
                   ("Search V3 baseline", lambda _: self.start("search")),
                   ("Compare NN / V3", lambda _: self.start("benchmark")),
                   ("Hammersley check", lambda _: self.start("reference")),
                   ("Stop", self.cancel), ("Play / pause", self.play),
                   ("Save result", self.save), ("Load result", self.load)]
        for i, (label, callback) in enumerate(actions):
            x = .055+(i % 4)*.23
            btn = Button(self.fig.add_axes([x, .135-(i//4)*.065, .20, .045]), label)
            btn.on_clicked(callback)
            self.buttons.append(btn)
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.fig.canvas.mpl_connect("close_event", self.close)

    def checkpoint(self):
        if self.stop.is_set():
            raise InterruptedError("Stopped by user.")

    def start(self, mode):
        if self.worker is not None:
            return
        try:
            cfg = Config(seed=int(self.seed.text), budget=int(self.budget.text))
            cfg.validate()
            seeds = int(self.seeds.text) if mode == "benchmark" else 30
            if seeds < 1:
                raise ValueError("Comparison needs at least one seed.")
            if mode == "neural":
                from .neural import signature
                reuse = self.reuse.get_status()[0]
                tag = hashlib.sha256(json.dumps(signature(cfg.samples),sort_keys=True).encode()).hexdigest()[:10]
                if not reuse:
                    tag = "fresh_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                directory = Path(__file__).resolve().parents[1]/"results"/"nn_learning"/tag
                cfg = replace(cfg, surrogate=True, nn_directory=str(directory), nn_reuse=bool(reuse))
        except ValueError as error:
            self.status.set_text(str(error))
            self.fig.canvas.draw_idle()
            return
        self.stop.clear()
        self.playing = False
        self.status.set_text("Running "+mode+"…")
        self.nn_status.set_text("NN: collecting / loading learning data…" if mode == "neural" else
                                "Comparison uses fresh learning per seed." if mode == "benchmark" else
                                "NN disabled for this run.")
        self.plot.clear()
        self.plot.set(xlabel="Evaluation", ylabel="Sampled area")
        self.progress_x, self.progress_y = [], []
        def work():
            try:
                if mode == "reference":
                    result, sofa = reference_check(cfg.target, self.checkpoint)
                    result["geometry"] = mapping(sofa)
                elif mode == "benchmark":
                    from .experiments import benchmark
                    folder = Path(__file__).resolve().parents[1]/"results"/("comparison_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                    benchmark(cfg, seeds, folder, progress=lambda msg:self.events.put(("status",msg)),
                              checkpoint=self.checkpoint, neural=True)
                    self.events.put(("status", "Comparison saved to "+str(folder)))
                    return
                else:
                    result = optimize(cfg, lambda row:self.events.put(("progress",row)), self.checkpoint)
                result["environment"] = provenance()
                if mode == "neural":
                    path = Path(cfg.nn_directory)/("result_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".json")
                    save_json(path, result)
                self.events.put(("done",result))
            except Exception as error:
                self.events.put(("status",str(error)))
            finally:
                self.events.put(("finished",None))
        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()

    def display(self, result):
        self.result = result
        self.sofa = shape(result["geometry"])
        parts = ([Motion.from_dict(d) for d in result["full_motion"]]
                 if "full_motion" in result else full_motion(Motion.from_dict(result["motion"])))
        self.frames = np.vstack([m(np.linspace(0,1,120)) for m in parts])
        self.frame = 0
        if hasattr(self,"patch"):
            self.patch.remove()
        self.patch = PathPatch(polygon_path(world(self.sofa,self.frames[0])), fc="#0d9488", alpha=.9)
        self.ax.add_patch(self.patch)
        self.local.clear()
        self.local.set_aspect("equal")
        self.local.add_patch(PathPatch(polygon_path(self.sofa), fc="#0d9488", ec="#115e59", lw=.7))
        if not self.sofa.is_empty:
            a,b,c,d = self.sofa.bounds
            self.local.set(xlim=(a-.1,c+.1),ylim=(b-.1,d+.1))
        self.local.set_title("Known reference" if "reference" in result else "Discovered boundary")
        check = result.get("verification", result.get("selected_finalist",{}).get("verification",{}))
        area = result.get("validated_area")
        self.metrics.set_text(f"Validated area: {area:.6f}\n{check.get('status','unknown')}" if area is not None
                              else f"No validated result: {check.get('status','unknown')}")
        self.status.set_text("Known Hammersley reference — not a discovery result." if "reference" in result
                             else "Finished. NN data, model (when trained), and result saved automatically." if "neural" in result
                             else "Finished. Play the motion or save the result.")
        self.show_neural(result.get("neural"))
        if "history" in result:
            self.plot.clear()
            self.plot.plot([r["evaluation"] for r in result["history"]],
                           [r["best_search_area"] for r in result["history"]],color="#2563eb")
            self.plot.set(xlabel="Evaluation",ylabel="Sampled area")
        self.fig.canvas.draw_idle()

    def tick(self):
        changed = False
        while True:
            try:
                kind,data = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "done":
                self.display(data)
            elif kind == "finished":
                self.worker = None
            elif kind == "status":
                self.status.set_text(data)
            elif kind == "progress":
                self.progress_x.append(data["evaluation"])
                self.progress_y.append(data["best_search_area"])
                self.plot.clear()
                self.plot.plot(self.progress_x,self.progress_y,color="#2563eb")
                self.plot.set(xlabel="Evaluation",ylabel="Sampled area")
                self.status.set_text(f"{data['stage']}: evaluation {data['evaluation']}; sampled area {data['best_search_area']:.6f}. Final validation pending.")
                self.show_neural(data.get("neural"))
        if self.playing and self.frames is not None:
            self.patch.set_path(polygon_path(world(self.sofa,self.frames[self.frame])))
            self.frame = (self.frame+1)%len(self.frames)
            changed = True
        if changed:
            self.fig.canvas.draw_idle()

    def show_neural(self, info):
        if not info:
            self.nn_status.set_text("NN disabled for this result.")
            return
        rmse = info.get("holdout_rmse")
        diagnostic = f" | holdout RMSE {rmse:.3f}" if rmse is not None else ""
        self.nn_status.set_text(f"NN: {info['status']} | {info['unique_rows']} unique samples "
            f"({info['loaded_rows']} reused) | {info['fits_this_run']} training rounds | "
            f"{info['guided_evaluations']} guided evaluations"+diagnostic)

    def cancel(self,_=None):
        self.stop.set()

    def play(self,_=None):
        if self.worker is None and self.frames is not None:
            self.playing = not self.playing

    def save(self,_=None):
        if self.result is None or self.worker is not None:
            return
        path = Path(__file__).resolve().parents[1]/"results"/("sofa_v3_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f")+".json")
        try:
            save_json(path,self.result)
            self.status.set_text("Saved "+str(path))
        except OSError as error:
            self.status.set_text(str(error))
        self.fig.canvas.draw_idle()

    def load(self,_=None):
        if self.worker is not None:
            return
        root = None
        try:
            from tkinter import Tk,filedialog
            root = Tk(); root.withdraw()
            name = filedialog.askopenfilename(filetypes=[("V3 JSON result","*.json")])
            if name:
                self.display(json.loads(Path(name).read_text(encoding="utf-8")))
        except Exception as error:
            self.status.set_text("Could not load result: "+str(error))
        finally:
            if root is not None:
                root.destroy()

    def close(self,_=None):
        self.stop.set()
        self.timer.stop()


def main():
    window = Window()
    plt.show()
    return window
