import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import threading
import numpy as np
from shapely.geometry import box, Polygon
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sofa_v3.motion import Motion, seed_motion, full_motion
from sofa_v3.geometry import (world, local, HALLWAY, reconstruct,
                              reconstruct_conservative, inset_hallway)
from sofa_v3.verify import verify
from sofa_v3.reference import hammersley_motion, hammersley_polygon, HAMMERSLEY_AREA
from sofa_v3.search import Config, optimize
from sofa_v3 import baseline


class MotionTests(unittest.TestCase):
    def test_transform_inverse_preserves_hole(self):
        sofa = box(-1,-1,1,1).difference(box(-.2,-.3,.2,.3))
        pose = [.33,.76,.72]
        restored = local(world(sofa,pose),pose)
        self.assertLess(sofa.symmetric_difference(restored).area,1e-12)
        self.assertEqual(len(restored.interiors),1)

    def test_inserting_knot_preserves_motion(self):
        m = Motion([0,.3,.7,1],[[0,1,0],[-.8,1.2,.9],[.1,1.8,.8],[0,1.5,np.pi/2]])
        finer=m.insert(.42)
        np.testing.assert_allclose(m(np.linspace(0,1,1001)),finer(np.linspace(0,1,1001)),atol=1e-12)

    def test_speed_bound_covers_cubic_overshoot(self):
        m=Motion([0,.3,.7,1],[[0,1,0],[1,-1,2],[-1,2,-1],[0,1,np.pi/2]])
        radius=2.
        for a,b in [(0,1),(.12,.61),(.71,.83)]:
            d=m.derivative(np.linspace(a,b,10001))
            actual=np.linalg.norm(d[:,:2],axis=1)+radius*abs(d[:,2])
            self.assertGreaterEqual(m.speed_bound(a,b,radius),actual.max())

    def test_invalid_motion(self):
        with self.assertRaises(ValueError):
            Motion([0,0,1],np.zeros((3,3)))
        with self.assertRaises(ValueError):
            Motion([0,1],[[0,0,0],[0,np.nan,0]])

    def test_exact_spline_endpoints_keep_exit_a_translation(self):
        from sofa_v3.geometry import straight_arm
        m=Motion([0,.25,.5,.75,1],[[0,1,0],[-.4,1.2,.3],[-.2,1.4,.9],[-.1,1.5,1.4],[0,1.6,np.pi/2]])
        np.testing.assert_array_equal(m(1),m.poses[-1])
        self.assertIsNotNone(straight_arm(full_motion(m)[-1]))


class GeometryTests(unittest.TestCase):
    def test_conservative_inset_clearance(self):
        inset=inset_hallway(.03)
        self.assertTrue(HALLWAY.covers(inset))
        self.assertGreaterEqual(inset.distance(HALLWAY.boundary),.03-1e-12)

    def test_reference_area_and_engine(self):
        p=hammersley_polygon(301)
        self.assertAlmostEqual(p.area,HAMMERSLEY_AREA,places=4)
        parts=full_motion(hammersley_motion())
        coarse=reconstruct(parts,97)
        sofa,_=reconstruct_conservative(parts,.005)
        self.assertGreater(sofa.area,2.16)
        self.assertLess(sofa.area,HAMMERSLEY_AREA)
        self.assertLess(sofa.area,coarse.area)
        self.assertEqual(verify(sofa,parts)["status"],"bounded_numerical_pass")

    def test_collision_between_dense_samples_is_rejected(self):
        sofa=box(-.1,-.1,.1,.1)
        motion=Motion([0,.4,.405,.41,1],
            [[-2,.5,0],[-1,.5,0],[-1,2,0],[-1,.5,0],[.5,2,0]],"linear")
        report=verify(sofa,[motion],dense_samples=2)
        self.assertEqual(report["status"],"collision")

    def test_inconclusive_is_not_pass(self):
        sofa=box(-.1,-.1,.1,.1)
        motion=Motion([0,1],[[-2,.5,0],[.5,2,0]],"linear")
        report=verify(sofa,[motion],max_depth=0,dense_samples=2)
        self.assertNotEqual(report["status"],"bounded_numerical_pass")

    def test_empty_shape_rejected(self):
        self.assertEqual(verify(Polygon(),full_motion(seed_motion()))["status"],"invalid")


class SearchTests(unittest.TestCase):
    def test_exact_budget_and_repeatability(self):
        cfg=Config(seed=3,budget=40,target=.01,finalists=1)
        a=optimize(cfg); b=optimize(cfg)
        self.assertEqual(a["search_evaluations"],40)
        self.assertEqual(b["search_evaluations"],40)
        self.assertEqual(a["motion"],b["motion"])
        self.assertEqual(a["validated_area"],b["validated_area"])
        self.assertEqual(a["selected_finalist"]["verification"]["status"],"bounded_numerical_pass")

    def test_cancellation(self):
        def stop(): raise InterruptedError()
        with self.assertRaises(InterruptedError):
            optimize(Config(budget=20),checkpoint=stop)

    def test_v1_adapter_matches_original_candidate_search(self):
        old=baseline.legacy_module()
        areas=[]
        original=old.build_sofa
        def capture(*args,**kwargs):
            result=original(*args,**kwargs)
            areas.append(result.area)
            return result
        class Events:
            def put(self,item):
                if item[0]=="status" and item[1].startswith("Rebuilding"):
                    raise old.Cancelled()
        with patch.object(old,"build_sofa",capture):
            old.optimize(old.REFERENCE_ROUTE,19,7,Events(),threading.Event())
        def return_history(archive,cfg,history,*args,**kwargs): return history
        with patch.object(baseline,"finalize",return_history):
            history=baseline.optimize_v1(Config(seed=7,budget=20,finalists=1))
        self.assertEqual(len(areas),20)
        np.testing.assert_allclose(areas,[r["candidate_area"] for r in history],atol=1e-12)


if __name__ == "__main__":
    unittest.main()
