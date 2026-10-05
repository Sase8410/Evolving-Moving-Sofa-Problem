import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.widgets import Button, RadioButtons
from matplotlib.animation import FuncAnimation
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

HALLWAY_LENGTH = 5.0

HALLWAY = unary_union([
    box(-HALLWAY_LENGTH, 0, 1, 1),
    box(0, 0, 1, HALLWAY_LENGTH),
])

SOFA_WIDTH = 1.5
SOFA_HEIGHT = 0.6

SOFA_VERTICES = np.array([
    [-SOFA_WIDTH/2, -SOFA_HEIGHT/2],
    [SOFA_WIDTH/2, -SOFA_HEIGHT/2],
    [SOFA_WIDTH/2, SOFA_HEIGHT/2],
    [-SOFA_WIDTH/2, SOFA_HEIGHT/2],
])

TURNING_ROUTE = np.array([
    [-2.5000, 0.5000,  0],
    [ 0.2300, 0.3200,  0],
    [ 0.1779, 0.5039, 15],
    [ 0.1805, 0.6548, 30],
    [ 0.2375, 0.7625, 45],
    [ 0.3452, 0.8195, 60],
    [ 0.4961, 0.8221, 75],
    [ 0.6800, 0.7700, 90],
    [ 0.5000, 2.5000, 90],
])

COLLISION_ROUTE = np.array([
    [-2.5, 0.5,  0],
    [ 0.5, 2.5, 90],
])

def transform_sofa(x, y, angle_degrees):
    angle = np.deg2rad(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return Polygon(SOFA_VERTICES @ rotation.T + [x, y])

def sample_path(waypoints, max_distance=0.01, max_angle=0.5):
    waypoints = np.asarray(waypoints, dtype=float)
    if (waypoints.ndim != 2 or waypoints.shape[1] != 3
            or len(waypoints) < 2 or not np.isfinite(waypoints).all()):
        raise ValueError("Provide at least two finite [x, y, angle] poses.")
    if not all(np.isfinite(v) and v > 0 for v in (max_distance, max_angle)):
        raise ValueError("Sampling limits must be finite and positive.")

    pieces = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        distance = np.linalg.norm(end[:2] - start[:2])
        angle_change = abs(end[2] - start[2])
        intervals = max(
            1,
            int(np.ceil(distance / max_distance)),
            int(np.ceil(angle_change / max_angle)),
        )
        pieces.append(np.linspace(start, end, intervals, endpoint=False))

    return np.vstack([*pieces, waypoints[-1:]])


def check_path(poses):
    sofas = [transform_sofa(*pose) for pose in poses]
    fits = np.array([HALLWAY.covers(sofa) for sofa in sofas], dtype=bool)
    outside_areas = np.array([
        sofa.difference(HALLWAY).area for sofa in sofas
    ])
    return sofas, fits, outside_areas


def animate_path(poses, sofas, fits, outside_areas, routes=None):
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(left=0.07, right=0.69, bottom=0.14, top=0.90)
    ax.set_facecolor("#e2e8f0")
    ax.add_patch(PolygonPatch(
        np.asarray(HALLWAY.exterior.coords), closed=True,
        facecolor="white", edgecolor="#334155", linewidth=2,
    ))
    path_line, = ax.plot(
        poses[:, 0], poses[:, 1], "--", color="#64748b", linewidth=1,
    )
    sofa_patch = PolygonPatch(
        np.asarray(sofas[0].exterior.coords), closed=True,
        facecolor="#22c55e", edgecolor="#334155", alpha=0.85,
    )
    ax.add_patch(sofa_patch)
    center, = ax.plot([], [], "ko", markersize=4)
    status = fig.text(0.74, 0.56, "", va="top", fontsize=10)
    summary = fig.text(0.74, 0.32, "", va="top", fontsize=10)
    ax.set(
        xlim=(-5.5, 2), ylim=(-1, 5.5), xlabel="x", ylabel="y",
        title="Moving Sofa - Step 2",
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.text(
        0.5, 0.055, "Sampled checks only; collisions between samples may be missed.",
        ha="center", fontsize=9,
    )

    fig.text(0.74, 0.89, "Choose a route", fontsize=12, weight="bold")
    selector = None
    if routes:
        selector = RadioButtons(
            fig.add_axes([0.74, 0.73, 0.23, 0.13]), tuple(routes), active=0,
        )
    else:
        fig.text(0.74, 0.80, "Supplied route", fontsize=11)

    play = Button(fig.add_axes([0.74, 0.64, 0.105, 0.05]), "Play")
    reset = Button(fig.add_axes([0.865, 0.64, 0.105, 0.05]), "Reset")
    state = {"index": 0, "running": False}

    def show_results():
        invalid = np.flatnonzero(~fits)
        if len(invalid):
            first = invalid[0]
            x, y, angle = poses[first]
            summary.set_text(
                f"PATH CHECK\n{len(poses)} samples checked\n"
                f"Colliding samples: {len(invalid)}\n"
                f"First collision: sample {first + 1}\n"
                f"x = {x:.3f}, y = {y:.3f}\n"
                f"Angle = {angle:.1f} deg"
            )
            summary.set_color("#b91c1c")
        else:
            summary.set_text(
                f"PATH CHECK\n{len(poses)} samples checked\n"
                "No sampled collisions"
            )
            summary.set_color("#166534")

    def draw():
        i = state["index"]
        x, y, angle = poses[i]
        sofa_patch.set_xy(np.asarray(sofas[i].exterior.coords))
        sofa_patch.set_facecolor("#22c55e" if fits[i] else "#ef4444")
        center.set_data([x], [y])
        label = "PLAYING" if state["running"] else "PAUSED"
        if not fits[i]:
            label = "COLLISION - stopped"
        elif i == len(poses) - 1:
            label = "FINISHED - no sampled collisions"
        status.set_text(
            f"{label}\n"
            f"Sample: {i + 1}/{len(poses)}\n"
            f"x = {x:.3f}, y = {y:.3f}\n"
            f"Angle = {angle:.1f} deg\n"
            f"Sofa area: {sofas[i].area:.3f}\n"
            f"Area outside hallway: {outside_areas[i]:.5f}"
        )
        play.label.set_text("Pause" if state["running"] else "Play")
        fig.canvas.draw_idle()

    def toggle_play(_):
        i = state["index"]
        if i == len(poses) - 1 or not fits[i]:
            state["index"] = 0
        state["running"] = not state["running"]
        draw()

    def restart(_):
        state.update(index=0, running=False)
        draw()

    def select_route(name):
        nonlocal poses, sofas, fits, outside_areas
        state.update(index=0, running=False)
        poses = sample_path(routes[name])
        sofas, fits, outside_areas = check_path(poses)
        path_line.set_data(poses[:, 0], poses[:, 1])
        show_results()
        draw()

    def advance(_):
        if not state["running"]:
            return
        i = state["index"]
        if fits[i] and i < len(poses) - 1:
            state["index"] += 1
        i = state["index"]
        if not fits[i] or i == len(poses) - 1:
            state["running"] = False
        draw()

    play.on_clicked(toggle_play)
    reset.on_clicked(restart)
    if selector is not None:
        selector.on_clicked(select_route)
    show_results()
    draw()
    animation = FuncAnimation(
        fig, advance, init_func=lambda: (), interval=20,
        blit=False, cache_frame_data=False,
    )
    fig.sofa_controls = (animation, play, reset, selector)
    plt.show()
    return animation, play, reset


def main():
    routes = {
        "Turning route": TURNING_ROUTE,
        "Collision demo": COLLISION_ROUTE,
    }
    poses = sample_path(TURNING_ROUTE)
    sofas, fits, outside_areas = check_path(poses)
    animate_path(poses, sofas, fits, outside_areas, routes=routes)


if __name__ == "__main__":
    main()