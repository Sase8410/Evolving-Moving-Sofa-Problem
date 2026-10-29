import json
import tempfile
import unittest
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
from shapely.geometry import box
from sofa_v3.motion import Motion, seed_motion
from sofa_v3.search import Config, optimize, retain_diverse, motion_distance, finalize
from sofa_v3.neural import NeuralGuide, ranking_accuracy, features
from sofa_v3.calibration import ValidationCalibrator, descriptors
from sofa_v3.io import save_json, load_result


class ReleaseTests(unittest.TestCase):
    def test_disconnected_valid_segments_cannot_teleport(self):
        from sofa_v3.verify import verify
        from sofa_v3.motion import continuous_parts
        parts=[Motion([0,1],[[-2,.5,0],[-2,.5,0]],'linear'),
               Motion([0,1],[[.5,2,0],[.5,2,0]],'linear')]
        self.assertFalse(continuous_parts(parts))
        self.assertEqual(verify(box(-.1,-.1,.1,.1),parts)['status'],'discontinuous_motion')

    def test_diversity_merges_equivalent_knots_and_nearby_motion(self):
        m = seed_motion()
        p = m.poses.copy(); p[2,0] += .00001
        archive = retain_diverse([(2.,m)],(2.1,Motion(m.times,p)),12,.015)
        self.assertEqual(len(archive),1)
        self.assertEqual(archive[0][0],2.1)
        self.assertLess(motion_distance(m,m.insert(.42)),1e-12)
        p[2,0] += .4
        archive = retain_diverse(archive,(1.9,Motion(m.times,p)),12,.015)
        self.assertEqual(len(archive),2)

    def test_final_selection_uses_verified_area_not_coarse_rank(self):
        a=seed_motion(); p=a.poses.copy();p[2,0]=.2;b=Motion(a.times,p)
        def rebuild(parts,*args):
            return (box(0,0,1,1) if parts[1](.5)[0]<.1 else box(0,0,1.5,1)), {}
        with patch('sofa_v3.search.reconstruct_conservative',side_effect=rebuild), \
             patch('sofa_v3.search.verify',return_value={'status':'bounded_numerical_pass'}):
            r=finalize([(3.,a),(2.,b)],Config(finalists=2),[{'best_search_area':3.}],
                       perf_counter(),lambda:None,'test')
        self.assertEqual(r['validated_area'],1.5)
        self.assertEqual(r['selected_finalist']['search_area'],2.)

    def test_progressive_ranking_and_nested_grids(self):
        motions=[]
        for i in range(5):
            m=seed_motion(); p=m.poses.copy();p[2,0]=i*.1
            motions.append((5.-i,Motion(m.times,p)))
        def reconstruct(parts,samples,checkpoint):
            return box(0,0,1+parts[1](.5)[0],1)
        with patch('sofa_v3.search.reconstruct',side_effect=reconstruct), \
             patch('sofa_v3.search.reconstruct_conservative',return_value=(box(0,0,1,1),{})), \
             patch('sofa_v3.search.verify',return_value={'status':'bounded_numerical_pass'}):
            r=finalize(motions,Config(finalists=1),[{'best_search_area':5.}],
                       perf_counter(),lambda:None,'test')
        self.assertEqual(r['selected_finalist']['search_area'],1.)
        self.assertEqual([p['samples'] for p in r['selection']['promotions']],[97,193])
        np.testing.assert_array_equal(np.linspace(0,1,49),np.linspace(0,1,193)[::4])

    def test_expired_finalization_cannot_claim_success(self):
        r=finalize([(2.,seed_motion())],Config(time_limit=1),[],
                   perf_counter()-2,lambda:None,'test')
        self.assertIsNone(r['validated_area'])
        self.assertEqual(r['selected_finalist']['verification']['status'],'time_limit')
        self.assertTrue(r['timing']['finalization_interrupted'])

    def test_wall_time_limit_and_different_history_lengths(self):
        from sofa_v3.experiments import summarize
        r=optimize(Config(seed=8,budget=100000,time_limit=2,target=.02,finalists=1))
        self.assertLess(r['search_evaluations'],100000)
        self.assertLess(r['total_seconds'],3.)
        r['experiment_method']='V3'
        short=dict(r,history=r['history'][:1])
        with tempfile.TemporaryDirectory() as tmp:
            summary=summarize([r,short],tmp)
            self.assertEqual(summary['V3']['runs'],2)

    def test_rank_gate_rejects_reversed_top_candidates(self):
        y=np.arange(20.)
        self.assertEqual(ranking_accuracy(y,-y)[0],0.)
        self.assertEqual(ranking_accuracy(y,y)[0],1.)
        self.assertIsNone(ranking_accuracy(np.ones(20),y)[0])

    def test_fallback_after_guidance_underperforms(self):
        g=NeuralGuide()
        for _ in range(16):
            g.record_outcome('nn_ranked',0.)
            g.record_outcome('exploration',.01)
        self.assertEqual(g.fallbacks,1)
        self.assertEqual(g.cooldown,64)
        original=np.ones(3)
        chosen,source=g.choose(original,None,None,None,None)
        self.assertEqual(source,'evolution_fallback')
        np.testing.assert_array_equal(chosen,original)
        g.close()

    def test_validated_calibration_requires_data_and_learns_loss(self):
        self.assertFalse(ValidationCalibrator().report['ready'])
        rng=np.random.default_rng(9); rows=[]
        for _ in range(160):
            m=seed_motion();p=m.poses.copy();p[1:-1,0]=rng.normal(0,.15,3)
            x=features(Motion(m.times,p)); coarse=2.+rng.uniform(0,.2)
            d=descriptors([x],[coarse])[0]
            loss=.005+.05*(coarse-2.)+.01*d[1]
            rows.append({'features':x.tolist(),'sampled_area':coarse,'validated_area':coarse-loss})
        cal=ValidationCalibrator(rows);cal.fit()
        self.assertTrue(cal.report['ready'])
        self.assertLess(cal.report['holdout_rmse'],cal.report['constant_baseline_rmse'])
        self.assertTrue(np.all(cal.loss([rows[0]['features']],[2.1])>=0))

    def test_save_load_rejects_false_area_and_preserves_old_results(self):
        m=seed_motion()
        from sofa_v3.motion import full_motion
        from shapely.geometry import mapping
        data={'motion':m.to_dict(),'geometry':mapping(box(0,0,1,1)),
              'validated_area':1.,'full_motion':[p.to_dict() for p in full_motion(m)]}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'result.json';save_json(path,data)
            self.assertEqual(load_result(path)['validated_area'],1.)
            data['validated_area']=2.;save_json(path,data)
            with self.assertRaisesRegex(ValueError,'disagrees'):
                load_result(path)

    def test_validation_target_isolates_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            NeuralGuide(directory=tmp,target=.002).close()
            with self.assertRaises(ValueError):
                NeuralGuide(directory=tmp,target=.001)


if __name__ == '__main__':
    unittest.main()
