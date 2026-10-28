import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.widgets import Button, RadioButtons, Slider

from sofa_path_simulation import (
    HALLWAY, TURNING_ROUTE, COLLISION_ROUTE, sample_path,
)
from sofa_shape_simulation import create_candidate, check_path


class SofaWindow:
    def __init__(self):
        self.running = False
        self.pending = False
        self.index = 0

        self.fig = plt.figure(figsize=(12, 8), facecolor="#f8fafc")
        self.fig.canvas.manager.set_window_title("Moving Sofa - Shape Controls")
        self.ax = self.fig.add_axes([0.06, 0.28, 0.60, 0.64])
        self.ax.set_facecolor("#e2e8f0")
        self.ax.add_patch(PolygonPatch(
            np.asarray(HALLWAY.exterior.coords), closed=True,
            facecolor="white", edgecolor="#334155", linewidth=2,
        ))
        self.path_line, = self.ax.plot([], [], "--", color="#64748b", lw=1)
        self.sofa_patch = PolygonPatch(
            [[0, 0], [0, 0], [0, 0]], closed=True,
            facecolor="#22c55e", edgecolor="#334155", alpha=0.85,
        )
        self.ax.add_patch(self.sofa_patch)
        self.origin, = self.ax.plot([], [], "ko", markersize=4)
        self.ax.set(
            xlim=(-5.5, 2), ylim=(-1, 5.5), xlabel="x", ylabel="y",
            title="Moving Sofa - Adjustable Shapes",
        )
        self.ax.set_aspect("equal")
        self.ax.grid(alpha=0.2)

        self.fig.text(0.73, 0.95, "SOFA SETTINGS", weight="bold", fontsize=12)
        self.shape = RadioButtons(
            self.fig.add_axes([0.73, 0.76, 0.22, 0.15]),
            ["rectangle", "notched", "curved"], active=2,
        )
        self.fig.text(0.73, 0.71, "Route", weight="bold")
        self.route = RadioButtons(
            self.fig.add_axes([0.73, 0.56, 0.22, 0.13]),
            ["Turn", "Collision demo"],
        )

        self.fig.text(0.73, 0.51, "Width")
        self.width = Slider(
            self.fig.add_axes([0.73, 0.47, 0.20, 0.025]),
            "", 0.5, 3.0, valinit=1.5, valstep=0.01, valfmt="%.2f",
        )
        self.fig.text(0.73, 0.43, "Height")
        self.height = Slider(
            self.fig.add_axes([0.73, 0.39, 0.20, 0.025]),
            "", 0.2, 1.4, valinit=0.6, valstep=0.01, valfmt="%.2f",
        )
        self.notch_label = self.fig.text(0.73, 0.35, "Notch depth")
        self.notch = Slider(
            self.fig.add_axes([0.73, 0.31, 0.20, 0.025]),
            "", 0.01, 0.59, valinit=0.25, valstep=0.01, valfmt="%.2f",
        )
        self.fig.text(0.73, 0.27, "Notch applies to notched / curved shapes.",
                      fontsize=8, color="#64748b")

        self.apply_button = Button(
            self.fig.add_axes([0.76, 0.20, 0.18, 0.05]), "Apply",
        )
        self.play_button = Button(
            self.fig.add_axes([0.12, 0.07, 0.14, 0.05]), "Play",
        )
        self.reset_button = Button(
            self.fig.add_axes([0.31, 0.07, 0.14, 0.05]), "Reset path",
        )
        self.status = self.fig.text(0.08, 0.23, "", va="top", fontsize=10)
        self.fig.text(
            0.5, 0.025,
            "Sampled checks only: collisions between samples may be missed.",
            ha="center", fontsize=9, color="#64748b",
        )

        self.shape.on_clicked(self.settings_changed)
        self.route.on_clicked(self.settings_changed)
        self.width.on_changed(self.settings_changed)
        self.height.on_changed(self.height_changed)
        self.notch.on_changed(self.settings_changed)
        self.apply_button.on_clicked(self.apply_settings)
        self.play_button.on_clicked(self.toggle_play)
        self.reset_button.on_clicked(self.reset_path)

        self.apply_settings()
        # Keeping the animation on this object prevents garbage collection.
        self.animation = FuncAnimation(
            self.fig, self.advance, init_func=lambda: (),
            interval=20, blit=False, cache_frame_data=False,
        )

    def settings_changed(self, _=None):
        self.running = False
        self.pending = True
        uses_notch = self.shape.value_selected != "rectangle"
        self.notch.set_active(uses_notch)
        self.notch_label.set_color("black" if uses_notch else "#94a3b8")
        self.update_display()

    def height_changed(self, _=None):
        # A notch must remain shallower than the sofa's height.
        limit = round(self.height.val - 0.01, 2)
        self.notch.valmax = limit
        self.notch.ax.set_xlim(self.notch.valmin, limit)
        if self.notch.val > limit:
            self.notch.set_val(limit)
        self.settings_changed()

    def apply_settings(self, _=None):
        self.running = False
        self.pending = True
        try:
            sofa = create_candidate(
                self.shape.value_selected,
                width=self.width.val, height=self.height.val,
                notch_depth=self.notch.val,
            )
            route = (TURNING_ROUTE if self.route.value_selected == "Turn"
                     else COLLISION_ROUTE)
            poses = sample_path(route)
            sofas, fits, outside = check_path(sofa, poses)
        except ValueError as error:
            self.play_button.set_active(False)
            self.play_button.label.set_text("Play")
            self.status.set_text(f"Cannot apply settings: {error}")
            self.fig.canvas.draw_idle()
            return

        self.sofa = sofa
        self.poses, self.sofas = poses, sofas
        self.fits, self.outside = fits, outside
        self.applied_shape = self.shape.value_selected
        self.applied_route = self.route.value_selected
        self.collisions = int(np.count_nonzero(~fits))
        self.index = 0
        self.pending = False
        self.path_line.set_data(poses[:, 0], poses[:, 1])
        self.update_display()

    def toggle_play(self, _=None):
        if self.pending:
            return
        if self.index == len(self.poses) - 1 or not self.fits[self.index]:
            self.index = 0
        self.running = not self.running and bool(self.fits[self.index])
        self.update_display()

    def reset_path(self, _=None):
        self.running = False
        self.index = 0
        self.update_display()

    def advance(self, _=None):
        if not self.running or self.pending:
            return
        if self.fits[self.index] and self.index < len(self.poses) - 1:
            self.index += 1
        if not self.fits[self.index] or self.index == len(self.poses) - 1:
            self.running = False
        self.update_display()

    def update_display(self):
        i = self.index
        self.sofa_patch.set_xy(np.asarray(self.sofas[i].exterior.coords))
        self.sofa_patch.set_facecolor("#22c55e" if self.fits[i] else "#ef4444")
        x, y, angle = self.poses[i]
        self.origin.set_data([x], [y])

        mode = "Playing" if self.running else "Ready / paused"
        if not self.fits[i]:
            mode = "Collision - stopped"
        elif i == len(self.poses) - 1:
            mode = "Finished"
        if self.pending:
            mode = "Settings changed - click Apply"

        result = (f"{self.collisions} colliding samples" if self.collisions
                  else "no sampled collisions")
        self.status.set_text(
            f"{mode}\n"
            f"Applied: {self.applied_shape} | Area: {self.sofa.area:.4f} square units\n"
            f"Route: {self.applied_route} | {result}\n"
            f"Sample {i + 1}/{len(self.poses)} | Angle: {angle:.1f} deg | "
            f"Outside area: {self.outside[i]:.5f}"
        )
        self.play_button.set_active(not self.pending)
        self.play_button.label.set_color("#94a3b8" if self.pending else "black")
        self.play_button.label.set_text("Pause" if self.running else "Play")
        self.fig.canvas.draw_idle()


def main():
    window = SofaWindow()
    plt.show()
    return window


if __name__ == "__main__":
    main()
