import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.widgets import Slider, Button
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HALLWAY_LENGTH = 5.0

hallway = unary_union([
    box(-HALLWAY_LENGTH, 0, 1, 1),
    box(0, 0, 1, HALLWAY_LENGTH),
])

SOFA_WIDTH = 1.5
SOFA_HEIGHT = 0.6

sofa_vertices = np.array([
    [-SOFA_WIDTH/2, -SOFA_HEIGHT/2],
    [SOFA_WIDTH/2, -SOFA_HEIGHT/2],
    [SOFA_WIDTH/2, SOFA_HEIGHT/2],
    [-SOFA_WIDTH/2, SOFA_HEIGHT/2],
])

def transform_sofa(x, y, angle_degrees):
    angle = np.deg2rad(angle_degrees)

    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])

    transformed = sofa_vertices @ rotation.T + np.array([x, y])
    return Polygon(transformed)

def sofa_fits(sofa):
    return hallway.covers(sofa)

fig, ax = plt.subplots(figsize=(9, 8))
plt.subplots_adjust(bottom=0.28)

ax.set_facecolor("#e2e8f0")

hallway_patch = PolygonPatch(
    np.asarray(hallway.exterior.coords),
    closed=True,
    facecolor="white",
    edgecolor="#334155",
    linewidth=2,
)
ax.add_patch(hallway_patch)

initial_x = -2.5
initial_y = 0.5
initial_angle = 0.0

initial_sofa = transform_sofa(initial_x, initial_y, initial_angle)

sofa_patch = PolygonPatch(
    np.asarray(initial_sofa.exterior.coords),
    closed=True,
    facecolor="#22c55e",
    edgecolor="#14532d",
    alpha=0.8,
    linewidth=2,
)
ax.add_patch(sofa_patch)

center_marker, = ax.plot(
    [initial_x], [initial_y], "ko", markersize=4
)

status_text = ax.text(
    0.02,
    0.97,
    "",
    transform=ax.transAxes,
    va="top",
    fontsize=11,
    bbox=dict(facecolor="white", edgecolor="#cbd5e1", alpha=0.95),
)

ax.set(
    xlim=(-HALLWAY_LENGTH - 0.5, 2),
    ylim=(-1, HALLWAY_LENGTH + 0.5),
    xlabel="x",
    ylabel="y",
    title="Moving Sofa — Hallway Simulation",
)
ax.set_aspect("equal")
ax.grid(alpha=0.2)

x_slider = Slider(
    fig.add_axes([0.18, 0.18, 0.65, 0.025]),
    "X position",
    -HALLWAY_LENGTH,
    1.0,
    valinit=initial_x,
)

y_slider = Slider(
    fig.add_axes([0.18, 0.13, 0.65, 0.025]),
    "Y position",
    0.0,
    HALLWAY_LENGTH,
    valinit=initial_y,
)

angle_slider = Slider(
    fig.add_axes([0.18, 0.08, 0.65, 0.025]),
    "Rotation",
    -180,
    180,
    valinit=initial_angle,
    valfmt="%0.1f°",
)

reset_button = Button(
    fig.add_axes([0.84, 0.02, 0.10, 0.035]),
    "Reset",
)

def update(_=None):
    x = x_slider.val
    y = y_slider.val
    angle = angle_slider.val

    sofa = transform_sofa(x, y, angle)
    fits = sofa_fits(sofa)
    outside_area = sofa.difference(hallway).area

    sofa_patch.set_xy(np.asarray(sofa.exterior.coords))
    sofa_patch.set_facecolor("#22c55e" if fits else "#ef4444")
    sofa_patch.set_edgecolor("#14532d" if fits else "#7f1d1d")

    center_marker.set_data([x], [y])

    status_text.set_text(
        f"{'VALID POSITION' if fits else 'COLLISION'}\n"
        f"Sofa area: {sofa.area:.3f}\n"
        f"Area outside hallway: {outside_area:.4f}"
    )

    fig.canvas.draw_idle()


def reset(_):
    x_slider.reset()
    y_slider.reset()
    angle_slider.reset()


x_slider.on_changed(update)
y_slider.on_changed(update)
angle_slider.on_changed(update)
reset_button.on_clicked(reset)

update()
plt.show()