from pathlib import Path
import queue
import threading
import textwrap

import matplotlib.pyplot as plt
from matplotlib.path import Path as PlotPath
from matplotlib.patches import PathPatch
from matplotlib.widgets import Button, RadioButtons, Slider, TextBox
import numpy as np
from shapely.geometry import Polygon, shape
from shapely.geometry.polygon import orient

from sofa_geometry import (
    Cancelled,
    Hallway,
    centered,
    edit_boundary,
    load_json,
    reconstruct,
    make_sofa,
    preset,
    result_payload,
    sample_route,
    save_json,
    verify_motion,
    world_sofa,
)
from sofa_path_planner import PlannerSettings, plan_path
from sofa_discovery import DiscoverySettings, discover

BLUE = "#2563eb"
INK = "#172554"
MUTED = "#64748b"


def polygon_path(sofa):
    if sofa.is_empty:
        return PlotPath([[0.0, 0.0]], [PlotPath.MOVETO])
    sofa = orient(sofa, sign=1)
    vertices, codes = [], []
    for ring in [sofa.exterior, *sofa.interiors]:
        points = np.asarray(ring.coords)
        vertices.extend(points)
        codes.extend(
            [PlotPath.MOVETO]
            + [PlotPath.LINETO] * (len(points) - 2)
            + [PlotPath.CLOSEPOLY]
        )
    return PlotPath(vertices, codes)


class SofaWindow:
    def __init__(self):
        self.sofa = preset("Rectangle")
        self.hallway = Hallway()
        self.route = None
        self.result = None
        self.report = None
        self.loaded_metadata = {}
        self.worker = None
        self.events = queue.Queue()
        self.stop_event = threading.Event()
        self.playing = False
        self.poses = None
        self.index = 0
        self.closed = False
        self.drawing = False
        self.draw_points = []
        self.drag_index = None
        self.drag_vertices = None
        self.undo_stack = []
        self.features_window = None
        self.widgets = []
        self.buttons = {}
        self.fig = plt.figure(figsize=(15, 9.5), facecolor="#f8fafc")
        self.fig.canvas.manager.set_window_title("Moving Sofa | Shape + Motion Lab")
        self.fig.text(
            0.045, 0.947, "Moving Sofa", fontsize=23, weight="bold", color=INK
        )
        self.fig.text(0.205, 0.948, "SHAPE + MOTION LAB", fontsize=11, color=MUTED)
        self.fig.text(
            0.045,
            0.914,
            "Draw a shape and find its movement, or discover both automatically.",
            fontsize=11,
            color=MUTED,
        )
        self.ax = self.fig.add_axes([0.045, 0.29, 0.39, 0.58])
        self.local = self.fig.add_axes([0.49, 0.615, 0.26, 0.255])
        self.chart = self.fig.add_axes([0.50, 0.29, 0.25, 0.12])
        for ax in (self.ax, self.local):
            ax.set_aspect("equal")
            ax.grid(alpha=0.15)
        self.ax.set_title(
            "Rigid movement through a unit-width hallway", fontsize=11, pad=12
        )
        self.local.set_title(
            "Shape editor • drag a vertex to change it", fontsize=10, pad=12
        )
        self.hall_patch = PathPatch(
            polygon_path(self.hallway.region), fc="white", ec="#94a3b8", lw=2
        )
        self.ax.add_patch(self.hall_patch)
        self.body = PathPatch(polygon_path(Polygon()), fc=BLUE, ec=INK, alpha=0.8)
        self.outline = PathPatch(polygon_path(self.sofa), fc="#bfdbfe", ec=BLUE, lw=1.5)
        self.ax.add_patch(self.body)
        self.local.add_patch(self.outline)
        (self.route_line,) = self.ax.plot([], [], color=BLUE, lw=1, alpha=0.65)
        (self.vertices_line,) = self.local.plot([], [], "o", color=BLUE, ms=3)
        (self.draft_line,) = self.local.plot([], [], "o--", color="#f97316", ms=4)
        (self.history_line,) = self.chart.plot([], [], color=BLUE, lw=1.6)
        self.chart.grid(alpha=0.15)
        self.chart.set_title("Search progress", fontsize=10)
        self.chart.tick_params(labelsize=8)
        self.metrics = self.fig.text(
            0.49, 0.56, "", fontsize=10, va="top", linespacing=1.5, color=INK
        )
        self.status = self.fig.text(
            0.045, 0.12, "Ready.", fontsize=10, color=INK, va="top"
        )
        self.fig.text(
            0.045,
            0.041,
            "Numerical motion checks • unresolved wall contact is reported • global optimality is not guaranteed",
            fontsize=9,
            color=MUTED,
        )
        self.fig.text(0.79, 0.89, "SEARCH", fontsize=11, weight="bold", color=INK)
        self.mode = RadioButtons(
            self.fig.add_axes([0.785, 0.785, 0.18, 0.09]),
            ["Find path", "Discover sofa"],
        )
        self.widgets.append(self.mode)
        self.budget = self.slider("Seconds", 0.74, 5, 600, 90, 5)
        self.seed = TextBox(
            self.fig.add_axes([0.855, 0.682, 0.11, 0.035]), "Seed  ", initial="42"
        )
        self.widgets.append(self.seed)
        self.button("Run", [0.79, 0.621, 0.08, 0.042], self.start)
        self.button("Stop", [0.885, 0.621, 0.08, 0.042], self.cancel)
        self.button(
            "Continue", [0.79, 0.564, 0.08, 0.039], lambda _: self.start(resume=True)
        )
        self.button("Refine area", [0.885, 0.564, 0.08, 0.039], self.refine_area)
        self.fig.text(0.79, 0.525, "SHAPE", fontsize=11, weight="bold", color=INK)
        self.shape_choice = RadioButtons(
            self.fig.add_axes([0.785, 0.401, 0.18, 0.11]),
            ["Rectangle", "Notched", "Curved"],
        )
        self.widgets.append(self.shape_choice)
        self.width = self.slider("Width", 0.366, 0.2, 4.0, 1.5, 0.05)
        self.height = self.slider("Height", 0.321, 0.15, 2.0, 0.6, 0.05)
        self.button("Apply preset", [0.79, 0.258, 0.175, 0.037], self.apply_preset)
        self.button("Draw polygon", [0.79, 0.204, 0.175, 0.037], self.toggle_draw)
        self.button("Notch / extrude", [0.79, 0.152, 0.175, 0.037], self.feature_editor)
        for i, (label, callback) in enumerate(
            [
                ("Load JSON", self.load),
                ("Save JSON", self.save),
                ("Check route", self.check_route),
                ("Play / pause", self.play),
                ("Undo shape", self.undo),
            ]
        ):
            self.button(label, [0.045 + i * 0.144, 0.204, 0.13, 0.039], callback)
        self.scrubber = Slider(
            self.fig.add_axes([0.10, 0.161, 0.61, 0.018]), "Motion", 0, 1, valinit=0.0
        )
        self.scrubber.on_changed(self.scrub)
        self.fig.canvas.mpl_connect("button_press_event", self.mouse_down)
        self.fig.canvas.mpl_connect("motion_notify_event", self.mouse_move)
        self.fig.canvas.mpl_connect("button_release_event", self.mouse_up)
        self.fig.canvas.mpl_connect("key_press_event", self.key)
        self.fig.canvas.mpl_connect("close_event", self.close)
        self.timer = self.fig.canvas.new_timer(interval=40)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self.draw_shape()
        self.update_controls()
        self.tell(
            "Choose Find path to move the current shape, or Discover sofa to start an independent search."
        )

    def slider(self, label, y, low, high, value, step):
        widget = Slider(
            self.fig.add_axes([0.845, y, 0.12, 0.019]),
            label,
            low,
            high,
            valinit=value,
            valstep=step,
        )
        widget.label.set_fontsize(9)
        widget.valtext.set_fontsize(9)
        self.widgets.append(widget)
        return widget

    def button(self, label, rect, callback):
        widget = Button(
            self.fig.add_axes(rect), label, color="white", hovercolor="#dbeafe"
        )
        widget.label.set_fontsize(9)
        widget.on_clicked(callback)
        self.buttons[label] = widget

    def tell(self, message):
        self.status.set_text(textwrap.fill(str(message), width=105))
        self.fig.canvas.draw_idle()

    def update_controls(self):
        busy = self.worker is not None
        for widget in self.widgets:
            widget.active = not busy
        for name, widget in self.buttons.items():
            enabled = not busy
            if name == "Stop":
                enabled = busy
            elif name in ("Play / pause", "Check route", "Continue"):
                enabled = not busy and self.route is not None and not self.drawing
            elif name == "Refine area":
                enabled = (
                    not busy
                    and self.result is not None
                    and self.result.get("mode") == "discovery"
                )
            elif name == "Undo shape":
                enabled = not busy and bool(self.undo_stack)
            elif name == "Run":
                enabled = not busy and not self.drawing
            widget.set_active(enabled)
            widget.label.set_color(INK if enabled else "#94a3b8")

    def set_shape(self, sofa, remember=True):
        if self.features_window is not None:
            plt.close(self.features_window)
        if remember:
            self.undo_stack.append(self.sofa)
            self.undo_stack = self.undo_stack[-20:]
        self.sofa = sofa
        self.result = None
        self.report = None
        self.route = None
        self.loaded_metadata = {}
        self.poses = None
        self.playing = False
        self.index = 0
        self.drawing = False
        self.draft_line.set_data([], [])
        self.buttons["Draw polygon"].label.set_text("Draw polygon")
        self.hallway = Hallway()
        self.draw_shape()
        self.update_controls()

    def draw_shape(self):
        self.outline.set_path(polygon_path(self.sofa))
        v = np.asarray(self.sofa.exterior.coords)
        if len(v) <= 250:
            self.vertices_line.set_data(v[:, 0], v[:, 1])
        else:
            self.vertices_line.set_data([], [])
        a, b, c, d = self.sofa.bounds
        padding = max(0.18, 0.12 * max(c - a, d - b))
        self.local.set(xlim=(a - padding, c + padding), ylim=(b - padding, d + padding))
        self.hall_patch.set_path(polygon_path(self.hallway.region))
        self.ax.set(
            xlim=(-self.hallway.length - 0.25, 1.5),
            ylim=(-0.4, self.hallway.length + 0.25),
        )
        if self.route is not None:
            self.route_line.set_data(self.route[:, 0], self.route[:, 1])
            self.body.set_path(polygon_path(world_sofa(self.sofa, self.route[0])))
        else:
            self.route_line.set_data([], [])
            self.body.set_path(polygon_path(world_sofa(self.sofa, [-2.5, 0.5, 0])))
        passed = self.report is not None and self.report.get("motion_bound_pass", False)
        self.body.set_facecolor("#16a34a" if passed else BLUE)
        state = "Not checked"
        extra = ""
        if self.report:
            state = {
                "motion_bound_pass": "Motion bounds passed (numerical)",
                "unresolved_contact": "Near contact: unresolved intervals",
                "collision": "Collision / incomplete passage",
            }.get(self.report["status"], "Not checked")
            extra = (
                f"\nChecked poses: {self.report['samples_checked']}"
                f"  |  Collisions: {self.report['colliding_samples']}"
                f"\nUnresolved intervals: {self.report['unresolved_intervals']}"
            )
        self.metrics.set_text(
            f"Area: {self.sofa.area:.6f} square units\n{state}{extra}\nGerver reference: approximately 2.2195"
        )
        self.fig.canvas.draw_idle()

    def apply_preset(self, _=None):
        if self.worker:
            return
        self.drawing = False
        self.draft_line.set_data([], [])
        self.buttons["Draw polygon"].label.set_text("Draw polygon")
        self.set_shape(
            preset(
                self.shape_choice.value_selected,
                self.width.val,
                self.height.val,
                self.height.val * 0.35,
            )
        )
        self.tell(
            "Shape updated. Find path searches without changing its size or boundary."
        )

    def undo(self, _=None):
        if self.worker or not self.undo_stack:
            return
        self.drawing = False
        self.draft_line.set_data([], [])
        self.buttons["Draw polygon"].label.set_text("Draw polygon")
        self.set_shape(self.undo_stack.pop(), remember=False)
        self.tell("Restored the previous shape.")

    def toggle_draw(self, _=None):
        if self.worker:
            return
        if self.drawing:
            try:
                sofa = centered(make_sofa(self.draw_points))
            except Exception as error:
                self.tell(error)
                return
            self.drawing = False
            self.draft_line.set_data([], [])
            self.buttons["Draw polygon"].label.set_text("Draw polygon")
            self.set_shape(sofa)
            self.tell("Polygon created. Choose Find path and Run.")
        else:
            self.drawing = True
            self.draw_points = []
            self.playing = False
            self.local.set(xlim=(-2.2, 2.2), ylim=(-1.3, 1.3))
            self.buttons["Draw polygon"].label.set_text("Finish polygon")
            self.tell(
                "Click boundary vertices in order in the shape editor. Right-click or Finish polygon closes it. Backspace removes a point; Esc cancels."
            )
        self.update_controls()

    def mouse_down(self, event):
        if self.worker or event.inaxes != self.local or event.xdata is None:
            return
        if self.drawing:
            if event.button == 3:
                self.toggle_draw()
                return
            if event.button == 1:
                self.draw_points.append([event.xdata, event.ydata])
                self.draw_draft()
        elif event.button == 1:
            vertices = np.asarray(self.sofa.exterior.coords)[:-1]
            pixels = self.local.transData.transform(vertices)
            distances = np.linalg.norm(pixels - [event.x, event.y], axis=1)
            index = int(np.argmin(distances))
            if distances[index] < 10:
                self.drag_index = index
                self.drag_vertices = vertices.copy()
                self.playing = False

    def mouse_move(self, event):
        if self.worker or self.drag_index is None or event.inaxes != self.local:
            return
        self.drag_vertices[self.drag_index] = [event.xdata, event.ydata]
        points = np.vstack((self.drag_vertices, self.drag_vertices[:1]))
        self.draft_line.set_data(points[:, 0], points[:, 1])
        self.fig.canvas.draw_idle()

    def mouse_up(self, event):
        if self.drag_index is None:
            return
        try:
            sofa = make_sofa(
                self.drag_vertices, [list(r.coords) for r in self.sofa.interiors]
            )
            self.set_shape(sofa)
            self.tell("Boundary edited. The previous route must be replanned.")
        except Exception as error:
            self.tell(f"Edit rejected: {error}")
        self.drag_index = None
        self.drag_vertices = None
        self.draft_line.set_data([], [])
        self.fig.canvas.draw_idle()

    def draw_draft(self):
        if self.draw_points:
            p = np.asarray(self.draw_points)
            self.draft_line.set_data(p[:, 0], p[:, 1])
        else:
            self.draft_line.set_data([], [])
        self.fig.canvas.draw_idle()

    def key(self, event):
        if self.worker:
            return
        if self.drawing and event.key == "backspace":
            self.draw_points = self.draw_points[:-1]
            self.draw_draft()
        elif self.drawing and event.key == "escape":
            self.drawing = False
            self.draw_points = []
            self.draw_draft()
            self.buttons["Draw polygon"].label.set_text("Draw polygon")
            self.draw_shape()
            self.update_controls()
            self.tell("Drawing cancelled.")

    def feature_editor(self, _=None):
        if self.worker or self.drawing:
            return
        if self.features_window is not None and plt.fignum_exists(
            self.features_window.number
        ):
            return
        base = self.sofa
        fig = plt.figure(figsize=(10, 6), facecolor="#f8fafc")
        self.features_window = fig
        fig.canvas.manager.set_window_title("Edit boundary | Notch / extrusion")
        ax = fig.add_axes([0.06, 0.23, 0.55, 0.65])
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        a, b, c, d = base.bounds
        ax.set(xlim=(a - 0.7, c + 0.7), ylim=(b - 0.7, d + 0.7))
        patch = PathPatch(polygon_path(base), fc="#bfdbfe", ec=BLUE)
        ax.add_patch(patch)
        pos = Slider(
            fig.add_axes([0.73, 0.62, 0.19, 0.03]), "Position", 0, 1, valinit=0.25
        )
        width = Slider(
            fig.add_axes([0.73, 0.51, 0.19, 0.03]), "Width", 0.02, 2.0, valinit=0.3
        )
        depth = Slider(
            fig.add_axes([0.73, 0.40, 0.19, 0.03]), "Depth", 0.005, 0.8, valinit=0.08
        )
        operation = RadioButtons(
            fig.add_axes([0.68, 0.71, 0.26, 0.17]), ["Extrude", "Notch"]
        )
        note = fig.text(
            0.06, 0.13, "Move the feature along the boundary, then Apply.", fontsize=10
        )
        current = [None]

        def preview(_=None):
            try:
                current[0] = edit_boundary(
                    base, pos.val, width.val, depth.val, operation.value_selected
                )
                patch.set_path(polygon_path(current[0]))
                note.set_text(
                    f"Preview area: {current[0].area:.6f}. Apply commits this edit."
                )
            except ValueError as error:
                current[0] = None
                patch.set_path(polygon_path(base))
                note.set_text(textwrap.fill(str(error), 80))
            fig.canvas.draw_idle()

        def apply(_):
            if current[0] is not None and not self.worker:
                self.set_shape(current[0])
                plt.close(fig)
                self.tell(
                    "Boundary edit applied. Choose Find path to search for a new movement."
                )

        for widget in (pos, width, depth):
            widget.on_changed(preview)
        operation.on_clicked(preview)
        button = Button(fig.add_axes([0.70, 0.24, 0.23, 0.06]), "Apply edit")
        button.on_clicked(apply)
        fig._sofa_widgets = (pos, width, depth, operation, button)
        preview()
        fig.show()

    def settings(self):
        try:
            seed = int(self.seed.text)
        except ValueError:
            raise ValueError("Seed must be a nonnegative integer.") from None
        if seed < 0:
            raise ValueError("Seed must be a nonnegative integer.")
        return float(self.budget.val), seed

    def launch(self, function):
        self.stop_event.clear()
        self.playing = False
        self.poses = None
        self.index = 0
        self.history_line.set_data([], [])
        if self.features_window is not None:
            plt.close(self.features_window)

        def checkpoint():
            if self.stop_event.is_set():
                raise Cancelled

        def progress(**data):
            self.events.put(("progress", data))

        def work():
            try:
                self.events.put(("done", function(checkpoint, progress)))
            except Cancelled:
                self.events.put(
                    (
                        "message",
                        "Stopped. The displayed candidate can be saved, but needs a new check before use.",
                    )
                )
            except Exception as error:
                self.events.put(("message", f"{type(error).__name__}: {error}"))
            finally:
                self.events.put(("finished", None))

        self.worker = threading.Thread(target=work, daemon=True)
        self.worker.start()
        self.update_controls()

    def start(self, _=None, resume=False):
        if self.worker or self.drawing:
            return
        try:
            seconds, seed = self.settings()
        except ValueError as error:
            self.tell(error)
            return
        if resume and self.route is None:
            self.tell("Load or discover a route first.")
            return
        sofa = self.sofa
        route = None if self.route is None else self.route.copy()
        hallway = self.hallway
        if resume or self.mode.value_selected == "Discover sofa":
            self.hallway = Hallway(7.0)
            self.tell(
                "Discovering shape and motion from generic movements. Final checks follow the search budget."
            )
            initial = route if resume else None
            self.launch(
                lambda check, progress: discover(
                    DiscoverySettings(
                        seconds=seconds,
                        seed=seed,
                        evaluations=max(1500, int(seconds * 90)),
                    ),
                    initial_route=initial,
                    checkpoint=check,
                    progress=progress,
                )
            )
        else:
            self.tell(
                "Searching translations and rotations for this fixed shape. Final checks follow the search budget."
            )
            from sofa_geometry import radius

            hallway = Hallway(max(hallway.length, 2 * radius(sofa) + 3))
            self.hallway = hallway
            self.draw_shape()
            self.launch(
                lambda check, progress: plan_path(
                    sofa,
                    PlannerSettings(seconds=seconds, seed=seed),
                    hallway=hallway,
                    initial_route=route,
                    checkpoint=check,
                    progress=progress,
                )
            )

    def check_route(self, _=None):
        if self.worker or self.route is None:
            return
        sofa, route, hallway = self.sofa, self.route.copy(), self.hallway
        old = self.result or self.loaded_metadata
        mode = "discovery" if old.get("mode") == "discovery" else "verification"
        old_settings = old.get("settings", {})
        if not isinstance(old_settings, dict):
            old_settings = {}
        try:
            step = float(old_settings.get("reconstruction_step", 0.002))
        except (TypeError, ValueError):
            step = 0.002
        if not 0.0001 <= step <= 0.02:
            step = 0.002
        self.tell("Checking unchanged geometry along the complete route...")

        def check(checkpoint, progress):
            report = verify_motion(sofa, route, hallway, checkpoint=checkpoint)
            return result_payload(
                sofa,
                route,
                hallway,
                mode=mode,
                success=True,
                verification_report=report,
                settings={**old_settings, "reconstruction_step": step},
            )

        self.launch(check)

    def refine_area(self, _=None):
        if self.worker or self.result is None or self.result.get("mode") != "discovery":
            return
        previous = self.result.copy()
        route = self.route.copy()
        hallway = self.hallway
        old_area = self.sofa.area
        step = max(0.000125, float(previous["settings"]["reconstruction_step"]) / 4.0)
        self.tell(
            f"Refining the discovered shape at motion step {step:g}. This may take longer than the original search."
        )

        def refine(checkpoint, progress):
            sofa, report = reconstruct(route, hallway, step, checkpoint)
            if not report["motion_bound_pass"]:
                return {
                    "success": False,
                    "message": "Finer reconstruction did not pass. The previous result has been retained.",
                }
            if sofa.area < old_area:
                return {
                    "success": False,
                    "message": "Finer reconstruction did not improve checked area. The previous result has been retained.",
                }
            metadata = {
                k: v
                for k, v in previous.items()
                if k
                not in (
                    "geometry",
                    "vertices",
                    "interior_rings",
                    "route_waypoints",
                    "schema_version",
                    "created_at",
                    "hallway_width",
                    "hallway_length",
                    "motion_interpolation",
                    "verification_report",
                    "settings",
                )
            }
            return result_payload(
                sofa,
                route,
                hallway,
                **metadata,
                verification_report=report,
                settings={**previous["settings"], "reconstruction_step": step},
            )

        self.launch(refine)

    def cancel(self, _=None):
        if self.worker:
            self.stop_event.set()
            self.tell("Stopping at the next geometry checkpoint...")

    def tick(self):
        changed = False
        while True:
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                break
            changed = True
            if kind == "progress":
                self.tell(data["message"])
                if data["kind"] == "candidate":
                    self.sofa = data["sofa"]
                    self.route = data["route"]
                    self.report = data.get("report")
                    self.result = None
                    self.poses = None
                    self.index = 0
                    self.draw_shape()
                    history = data.get("history", [])
                    if history:
                        if "evaluation" in history[0]:
                            x = [r["evaluation"] for r in history]
                            y = [r["stage_best_sampled_area"] for r in history]
                            self.chart.set_title(
                                "Sampled search area • not final area", fontsize=9
                            )
                        else:
                            x = list(range(1, len(history) + 1))
                            y = [r["clearance_lower_bound"] for r in history]
                            self.chart.set_title(
                                "Route clearance lower bound", fontsize=9
                            )
                        self.history_line.set_data(x, y)
                        self.chart.relim()
                        self.chart.autoscale_view()
            elif kind == "done":
                if data.get("success"):
                    self.result = data
                    self.sofa = shape(data["geometry"])
                    self.route = np.asarray(data["route_waypoints"])
                    self.hallway = Hallway(data["hallway_length"])
                    self.report = data["verification_report"]
                    self.poses = sample_route(self.route, 0.035, 1.5)
                    self.index = 0
                    self.draw_shape()
                    msg = f"Finished. Area {self.sofa.area:.6f}. "
                    if self.report["motion_bound_pass"]:
                        msg += "Motion bounds passed. Play or save the result."
                    else:
                        msg += "Movement did not pass all motion bounds; inspect the report before using it."
                    self.tell(msg)
                else:
                    self.tell(data["message"])
            elif kind == "message":
                self.tell(data)
            elif kind == "finished":
                self.worker = None
                self.update_controls()
        if self.playing and self.poses is not None:
            self.index = (self.index + 1) % len(self.poses)
            self.scrubber.eventson = False
            self.scrubber.set_val(self.index / max(1, len(self.poses) - 1))
            self.scrubber.eventson = True
            self.show_pose()
            changed = True
        if changed:
            self.fig.canvas.draw_idle()

    def show_pose(self):
        if self.poses is not None:
            moved = world_sofa(self.sofa, self.poses[self.index])
            self.body.set_path(polygon_path(moved))
            fits = self.hallway.region.covers(moved)
            self.body.set_facecolor(
                (
                    "#16a34a"
                    if self.report and self.report.get("motion_bound_pass")
                    else BLUE
                )
                if fits
                else "#ef4444"
            )

    def play(self, _=None):
        if self.worker or self.route is None:
            return
        if self.poses is None:
            self.poses = sample_route(self.route, 0.035, 1.5)
            self.index = 0
        self.playing = not self.playing

    def scrub(self, value):
        if self.worker or self.route is None:
            return
        if self.poses is None:
            self.poses = sample_route(self.route, 0.035, 1.5)
        self.playing = False
        self.index = int(round(value * (len(self.poses) - 1)))
        self.show_pose()
        self.fig.canvas.draw_idle()

    def file_dialog(self, save=False):
        from tkinter import Tk, filedialog

        root = Tk()
        root.withdraw()
        try:
            if save:
                return filedialog.asksaveasfilename(
                    parent=root,
                    title="Save sofa and motion",
                    defaultextension=".json",
                    filetypes=[("Sofa JSON", "*.json")],
                )
            return filedialog.askopenfilename(
                parent=root,
                title="Load shape or saved result",
                filetypes=[("Sofa JSON", "*.json")],
            )
        finally:
            root.destroy()

    def load(self, _=None):
        if self.worker:
            return
        try:
            path = self.file_dialog()
            if not path:
                return
            sofa, route, hallway, data = load_json(path)
            self.set_shape(sofa)
            self.route = route
            self.hallway = hallway
            self.loaded_metadata = data
            self.result = None
            self.report = None
            self.draw_shape()
            self.update_controls()
            self.tell(
                f"Loaded {Path(path).name}. Saved validation labels are untrusted; Check route or Find path to verify it."
            )
        except Exception as error:
            self.tell(f"Could not load: {error}")

    def save(self, _=None):
        if self.worker:
            return
        try:
            path = self.file_dialog(save=True)
            if not path:
                return
            data = self.result or result_payload(
                self.sofa,
                self.route,
                self.hallway,
                mode="custom_shape",
                verification_report=self.report,
                success=False,
            )
            save_json(path, data)
            self.tell(f"Saved {Path(path).name}.")
        except Exception as error:
            self.tell(f"Could not save: {error}")

    def close(self, _=None):
        self.closed = True
        self.stop_event.set()
        self.timer.stop()
        if self.features_window is not None:
            plt.close(self.features_window)


def main():
    window = SofaWindow()
    window.fig._sofa_window = window
    plt.show()
    return window


if __name__ == "__main__":
    main()
