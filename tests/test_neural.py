pass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sofa_v3.motion import seed_motion
from sofa_v3.search import Config, Parameterization, optimize
from sofa_v3.neural import NeuralGuide, features


def teach(guide, n=160):
    pass
    rng = np.random.default_rng(88)
    param = Parameterization(seed_motion())
    for i in range(n):
        x = param.encode(param.base)+rng.normal(0, .08, len(param.lo))
        motion = param.decode(np.clip(x,param.lo,param.hi))
        target = 1.5+.8*x[0]-.3*x[1]+.2*x[2]**2
        guide.observe(motion, target, i+1, "synthetic_test")


class NeuralTests(unittest.TestCase):
    def test_knot_insertion_has_same_feature_representation(self):
        motion = seed_motion()
        np.testing.assert_allclose(features(motion), features(motion.insert(.42)), atol=1e-12)

    def test_learns_and_numeric_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = NeuralGuide(epochs=80,directory=tmp)
            try:
                teach(g)
                self.assertTrue(g.maybe_fit())
                self.assertTrue(g.report["ready"])
                self.assertLess(g.report["holdout_rmse"],g.report["constant_baseline_rmse"])
                self.assertGreaterEqual(g.report["top_half_ranking_accuracy"],.60)
                param = Parameterization(seed_motion())
                base = param.encode(param.base)
                population = np.tile(base,(20,1))
                for i in range(20):
                    g.choose(base,population,np.ones(20),param,np.random.default_rng(i))
                self.assertGreater(g.guided,0)
                x = np.array([r["features"] for r in g.rows[:10]])
                before = g.predict(x)
                size = len(g.rows)
                g.observe(seed_motion(),1.5,161,"test")
                g.observe(seed_motion(),1.5,162,"test")
                self.assertEqual(len(g.rows),size+1)
            finally:
                g.close()
            other = NeuralGuide(directory=tmp)
            try:
                self.assertEqual(other.loaded_models,3)
                self.assertEqual(other.loaded_rows,size+1)
                np.testing.assert_allclose(other.predict(x),before,atol=0,rtol=0)
            finally:
                other.close()

    def test_incompatible_data_rejected_and_lock_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            NeuralGuide(directory=tmp).close()
            with self.assertRaisesRegex(ValueError,"different geometry"):
                NeuralGuide(samples=97,directory=tmp)
            self.assertFalse((Path(tmp)/"active.lock").exists())
            NeuralGuide(directory=tmp).close()

    def test_concurrent_writer_rejected_without_removing_existing_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = NeuralGuide(directory=tmp)
            try:
                with self.assertRaisesRegex(ValueError,"already in use"):
                    NeuralGuide(directory=tmp)
                self.assertTrue((Path(tmp)/"active.lock").exists())
            finally:
                first.close()

    def test_incomplete_last_append_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            g = NeuralGuide(directory=tmp)
            g.observe(seed_motion(),1.5,1,"test")
            g.close()
            path = Path(tmp)/"evaluations.jsonl"
            with path.open("ab") as f:
                f.write(b'{"incomplete":')
            loaded = NeuralGuide(directory=tmp)
            loaded.observe(seed_motion(),1.5,2,"test")
            loaded.close()
            self.assertEqual(len([json.loads(s) for s in path.read_text().splitlines()]),2)

    def test_training_is_cancellable(self):
        calls = [0]
        def stop():
            calls[0] += 1
            if calls[0] > 5:
                raise InterruptedError()
        guide = NeuralGuide(checkpoint=stop)
        teach(guide)
        with self.assertRaises(InterruptedError):
            guide.maybe_fit()

    def test_nn_budget_guidance_real_labels_and_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(seed=2,budget=220,surrogate=True,target=.01,finalists=1,nn_directory=tmp)
            result = optimize(cfg)
            rows = [json.loads(s) for s in (Path(tmp)/"evaluations.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows),cfg.budget)
            self.assertEqual(result["search_evaluations"],cfg.budget)
            
            self.assertGreater(result["neural"]["fits_this_run"],0)
            self.assertEqual(result["neural"]["guided_evaluations"],sum(
                h.get("candidate_source","").startswith("nn_") for h in result["history"]))
            np.testing.assert_allclose([r["sampled_area"] for r in rows],
                                       [h["candidate_area"] for h in result["history"]])
            from sofa_v3.geometry import reconstruct
            from sofa_v3.motion import Motion, full_motion
            for row in rows[::25]:
                sofa = reconstruct(full_motion(Motion.from_dict(row["motion"])),cfg.samples)
                self.assertAlmostEqual(sofa.area,row["sampled_area"],places=12)
            self.assertIsNotNone(result["validated_area"])
            self.assertEqual(result["selected_finalist"]["verification"]["status"],"bounded_numerical_pass")
            self.assertFalse((Path(tmp)/"active.lock").exists())
            
            warm = optimize(Config(seed=3,budget=20,surrogate=True,target=.01,
                                   finalists=1,nn_directory=tmp))
            self.assertGreater(warm["neural"]["loaded_rows"],0)
            self.assertEqual(warm["neural"]["loaded_models"],3)

    def test_tiny_budget_does_not_claim_trained_network(self):
        cfg = Config(seed=5,budget=20,surrogate=True,target=.01,finalists=1)
        nn = optimize(cfg)
        self.assertEqual(nn["neural"]["fits_this_run"],0)
        self.assertEqual(nn["neural"]["guided_evaluations"],0)
        self.assertFalse(nn["neural"]["ready"])
        plain = optimize(Config(seed=5,budget=20,target=.01,finalists=1))
        self.assertEqual(nn["motion"],plain["motion"])

    def test_cancel_closes_learning_journal_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            def stop(): raise InterruptedError()
            with self.assertRaises(InterruptedError):
                optimize(Config(surrogate=True,nn_directory=tmp),checkpoint=stop)
            self.assertFalse((Path(tmp)/"active.lock").exists())
            NeuralGuide(directory=tmp).close()

    def test_benchmark_cold_start_isolation_and_resume(self):
        from sofa_v3.experiments import benchmark
        seen = []
        def fake(cfg,**kwargs):
            seen.append(cfg)
            return {"settings":{"seed":cfg.seed},"search_evaluations":cfg.budget}
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sofa_v3.experiments.optimize",side_effect=fake), \
                 patch("sofa_v3.experiments.summarize",return_value={}):
                cfg = Config(budget=20,nn_directory="must_not_reuse")
                benchmark(cfg,2,tmp,progress=lambda _:None,neural=True)
                self.assertEqual(len(seen),4)
                neural = [c for c in seen if c.surrogate]
                self.assertTrue(all(not c.nn_reuse for c in neural))
                self.assertEqual(len({c.nn_directory for c in neural}),2)
                self.assertTrue(all(c.nn_directory != "must_not_reuse" for c in neural))
                benchmark(cfg,2,tmp,progress=lambda _:None,neural=True)
                self.assertEqual(len(seen),4)


if __name__ == "__main__":
    unittest.main()
