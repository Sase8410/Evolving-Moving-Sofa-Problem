import os

os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon, box, shape

from sofa_geometry import (
    Cancelled,
    Hallway,
    edit_boundary,
    envelope,
    load_json,
    make_sofa,
    preset,
    reconstruct,
    result_payload,
    sample_route,
    save_json,
    verify_motion,
    world_sofa,
)
from sofa_path_planner import PlannerSettings, plan_path, unwrap_route
from sofa_discovery import (
    DiscoverySettings,
    controls_to_route,
    discover,
    initial_controls,
)


class GeometryTests(unittest.TestCase):
    def test_detects_collision_between_valid_endpoints(self):
        body = box(-0.08, -0.08, 0.08, 0.08)
        route = np.array([[-0.8, 0.8, 0], [0.2, 1.8, 0]])
        self.assertTrue(
            all(Hallway().region.covers(world_sofa(body, p)) for p in route)
        )
        report = verify_motion(body, route)
        self.assertGreater(report["colliding_samples"], 0)
        self.assertFalse(report["motion_bound_pass"])

    def test_rotation_collision_between_valid_endpoint_poses(self):
        body = box(-0.9, -0.1, 0.9, 0.1)
        report = verify_motion(body, np.array([[-2, 0.5, 0], [-2, 0.5, 180]]))
        self.assertGreater(report["colliding_samples"], 0)

    def test_exact_contact_axis_translation_is_permitted(self):
        body = box(-0.5, -0.5, 0.5, 0.5)
        route = np.array([[-2, 0.5, 0], [0.5, 0.5, 0], [0.5, 2, 0]])
        report = verify_motion(body, route)
        self.assertTrue(report["motion_bound_pass"])
        self.assertEqual(report["minimum_sampled_clearance"], 0.0)

    def test_partial_turn_is_rejected(self):
        report = verify_motion(box(-0.1, -0.1, 0.1, 0.1), [[-3, 0.5, 0], [-2, 0.5, 0]])
        self.assertFalse(report["endpoints_in_separate_arms"])
        self.assertFalse(report["motion_bound_pass"])

    def test_self_intersection_and_disconnected_edit_rejected(self):
        with self.assertRaises(ValueError):
            make_sofa([[0, 0], [1, 1], [0, 1], [1, 0]])
        with self.assertRaises(ValueError):
            edit_boundary(box(-1, -0.3, 1, 0.3), 0.19, 0.2, 0.7, "Notch")

    def test_holes_survive_transform_and_json_roundtrip(self):
        body = Polygon(
            [[-1, -0.4], [1, -0.4], [1, 0.4], [-1, 0.4]],
            [[[-0.1, -0.1], [-0.1, 0.1], [0.1, 0.1], [0.1, -0.1]]],
        )
        moved = world_sofa(body, [0.2, 0.3, 37])
        self.assertAlmostEqual(body.area, moved.area)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "shape.json"
            save_json(path, result_payload(body, None, Hallway()))
            loaded, route, _, _ = load_json(path)
            self.assertTrue(body.equals(loaded))
            self.assertIsNone(route)
            self.assertEqual(len(loaded.interiors), 1)

    def test_legacy_json_and_nonmonotonic_angles(self):
        body = preset("Rectangle")
        route = np.array([[-2, 0.5, 0], [0, 0.5, 40], [0.2, 0.6, 25], [0.5, 3, 90]])
        self.assertTrue(np.any(np.diff(sample_route(route)[:, 2]) < 0))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.json"
            save_json(
                path,
                {
                    "vertices": list(body.exterior.coords),
                    "route_waypoints": route.tolist(),
                },
            )
            loaded, q, _, _ = load_json(path)
            self.assertTrue(body.equals(loaded))
            np.testing.assert_array_equal(q, route)
        q = unwrap_route([[0, 0, 179], [0, 0, -179]])
        self.assertAlmostEqual(q[1, 2] - q[0, 2], 2.0)

    def test_reconstruction_is_checked_on_an_independent_dense_grid(self):
        controls = initial_controls(5, np.random.default_rng(42), 0)
        route = controls_to_route(controls)
        body, report = reconstruct(route, Hallway(7), resolution=0.008)
        self.assertTrue(report["motion_bound_pass"])
        self.assertLess(body.area, report["reconstruction_area"])
        for pose in sample_route(route, 0.008, 0.4):
            self.assertTrue(Hallway(7).region.covers(world_sofa(body, pose)))

    def test_invalid_sampling_and_nonfinite_vertices(self):
        with self.assertRaises(ValueError):
            sample_route([[0, 0, 0], [1, 1, 90]], 0, 1)
        with self.assertRaises(ValueError):
            make_sofa([[0, 0], [1, 0], [np.nan, 1]])


class SearchTests(unittest.TestCase):
    def test_fixed_shape_finds_full_route_without_changing_polygon(self):
        body = preset("Rectangle", 1.5, 0.6)
        before = body.wkb
        result = plan_path(body, PlannerSettings(seconds=8, seed=42, solution_limit=1))
        self.assertTrue(result["success"])
        self.assertEqual(body.wkb, before)
        self.assertTrue(body.equals(shape(result["geometry"])))
        self.assertTrue(result["verification_report"]["motion_bound_pass"])

    def test_oversize_shape_reports_search_failure(self):
        result = plan_path(preset("Rectangle", 1.4, 1.4), PlannerSettings(seconds=2))
        self.assertFalse(result["success"])
        self.assertIn("No fitting", result["message"])

    def test_search_is_cancellable(self):
        def stop():
            raise Cancelled

        with self.assertRaises(Cancelled):
            discover(DiscoverySettings(seconds=2, evaluations=50), checkpoint=stop)


class GuiTests(unittest.TestCase):
    def setUp(self):
        from sofa_optimizer_gui import SofaWindow

        self.window = SofaWindow()

    def tearDown(self):
        import matplotlib.pyplot as plt

        self.window.close()
        plt.close("all")

    def test_mode_survives_busy_toggle_and_polygon_draw(self):
        w = self.window
        w.mode.set_active(1)
        w.worker = object()
        w.update_controls()
        self.assertEqual(w.mode.value_selected, "Discover sofa")
        self.assertFalse(w.mode.active)
        w.worker = None
        w.update_controls()
        self.assertTrue(w.mode.active)
        w.toggle_draw()
        w.draw_points = [[-0.6, -0.2], [0.6, -0.2], [0.4, 0.3], [-0.5, 0.4]]
        w.toggle_draw()
        self.assertFalse(w.drawing)
        self.assertTrue(w.sofa.is_valid)
        w.undo()
        self.assertAlmostEqual(w.sofa.area, 0.9)

    def test_loading_does_not_trust_saved_pass_label(self):
        w = self.window
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forged.json"
            save_json(
                path,
                result_payload(
                    preset("Rectangle"),
                    [[-2, 0.5, 0], [0.5, 2, 90]],
                    Hallway(),
                    verification_report={"motion_bound_pass": True},
                ),
            )
            with patch.object(w, "file_dialog", return_value=str(path)):
                w.load()
        self.assertIsNone(w.report)
        self.assertIsNone(w.result)

    def test_refinement_worker_returns_new_checked_geometry(self):
        w = self.window
        body, larger = box(-0.2, -0.2, 0.2, 0.2), box(-0.25, -0.25, 0.25, 0.25)
        route = np.array([[-2.0, 0.5, 0.0], [0.5, 0.5, 0.0], [0.5, 2.0, 0.0]])
        report = verify_motion(larger, route)
        w.set_shape(body)
        w.route = route
        w.result = result_payload(
            body,
            route,
            Hallway(),
            mode="discovery",
            success=True,
            settings={"reconstruction_step": 0.002},
            verification_report=verify_motion(body, route),
        )
        with patch("sofa_optimizer_gui.reconstruct", return_value=(larger, report)):
            w.refine_area()
            w.worker.join(timeout=2.0)
            w.tick()
        self.assertIsNone(w.worker)
        self.assertTrue(w.sofa.equals(larger))
        self.assertTrue(w.report["motion_bound_pass"])
        self.assertEqual(w.result["settings"]["reconstruction_step"], 0.0005)

    def test_new_candidate_invalidates_previous_animation(self):
        w = self.window
        w.poses = np.array([[-2.0, 0.5, 0.0], [0.5, 2.0, 90.0]])
        w.events.put(
            (
                "progress",
                {
                    "kind": "candidate",
                    "message": "New candidate",
                    "sofa": w.sofa,
                    "route": w.poses.copy(),
                    "report": None,
                },
            )
        )
        w.tick()
        self.assertIsNone(w.poses)
        self.assertIsNone(w.report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
