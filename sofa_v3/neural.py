pass
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import uuid
import numpy as np
from .motion import Motion
from .io import save_json
from .calibration import ValidationCalibrator

FEATURE_VERSION = "motion17-xyz-radians-v1"
FEATURE_COUNT = 51


def features(motion):
    return np.asarray(motion(np.linspace(0, 1, 17)), dtype=float).ravel()


def signature(samples, target=.002):
    root = Path(__file__).parent
    return {"features": FEATURE_VERSION, "samples": samples, "validation_target": target,
            "label": "sampled_search_area_not_validated",
            "source": {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in ("motion.py", "geometry.py", "neural.py", "calibration.py")}}


def ranking_accuracy(actual, predicted):
    pass
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    ids = np.argsort(actual)[-max(8, len(actual)//2):]
    a, p = actual[ids], predicted[ids]
    i, j = np.triu_indices(len(ids), 1)
    useful = abs(a[i]-a[j]) > 1e-6
    count = int(useful.sum())
    accuracy = float(np.mean(((a[i]-a[j])*(p[i]-p[j]))[useful] > 0)) if count else None
    return accuracy, count


def forward(x, weights):
    for i, (w, b) in enumerate(weights):
        x = x @ w + b
        if i < len(weights)-1:
            x = np.tanh(x)
    return x.ravel()


class NeuralGuide:
    def __init__(self, seed=0, samples=49, directory=None, reuse=True,
                 minimum=64, retrain_every=64, epochs=100, checkpoint=lambda: None, target=.002):
        
        try:
            from sklearn.neural_network import MLPRegressor
            from threadpoolctl import threadpool_limits
        except ImportError as error:
            raise RuntimeError("NN mode needs scikit-learn. Install requirements.txt.") from error
        self.mlp_class, self.thread_limits = MLPRegressor, threadpool_limits
        self.seed, self.checkpoint = seed, checkpoint
        self.minimum, self.retrain_every, self.epochs = minimum, retrain_every, epochs
        self.schema = signature(samples, target)
        self.directory = Path(directory) if directory else None
        self.rows, self.keys = [], set()
        self.models, self.scaling = [], None
        self.report = {"status": "collecting", "ready": False}
        self.fit_count = self.last_fit_size = 0
        self.loaded_rows = self.loaded_models = 0
        self.guided = self.exploration = self.proposals = 0
        self.training_seconds = 0.
        self.last_prediction_error = None
        self.run_id = uuid.uuid4().hex
        self.journal = None
        self.lock_path = None
        self.lock_fd = None
        self.outcomes, self.cooldown, self.fallbacks = [], 0, 0
        self.calibrator = ValidationCalibrator()
        try:
            if self.directory:
                self.directory.mkdir(parents=True, exist_ok=True)
                lock_path = self.directory/"active.lock"
                try:
                    self.lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError as error:
                    raise ValueError("Learning folder is already in use. If a previous process crashed, "
                                     "close it and remove active.lock, or select a new folder.") from error
                self.lock_path = lock_path
                os.write(self.lock_fd, str(os.getpid()).encode())
                schema_path = self.directory/"schema.json"
                if schema_path.exists():
                    if json.loads(schema_path.read_text()) != self.schema:
                        raise ValueError("Learning data uses different geometry or samples. Choose a new NN folder.")
                    if not reuse:
                        raise ValueError("Fresh learning requires a new empty NN folder.")
                    self._load_rows()
                    self.loaded_rows = len(self.rows)
                    self._load_model()
                    validated = self.directory/"validated_training.json"
                    if validated.exists():
                        self.calibrator = ValidationCalibrator(json.loads(validated.read_text()))
                        self.calibrator.fit()
                else:
                    if any(p.name != "active.lock" for p in self.directory.iterdir()):
                        raise ValueError("Choose an empty folder for new NN learning data.")
                    save_json(schema_path, self.schema)
                self.journal = (self.directory/"evaluations.jsonl").open("a", encoding="utf-8")
        except Exception:
            self.close()
            raise

    def close(self):
        if self.journal:
            self.journal.close()
            self.journal = None
        if self.lock_fd is not None:
            os.close(self.lock_fd)
            self.lock_fd = None
        if self.lock_path:
            self.lock_path.unlink(missing_ok=True)
            self.lock_path = None

    @staticmethod
    def key(x):
        return hashlib.sha256(np.round(x, 10).astype("<f8").tobytes()).hexdigest()

    def _remember(self, row):
        x = np.asarray(row["features"], dtype=float)
        y = float(row["sampled_area"])
        if x.shape != (FEATURE_COUNT,) or not np.isfinite(x).all() or not np.isfinite(y) or y < 0:
            raise ValueError("Invalid NN training row.")
        key = self.key(x)
        if key not in self.keys:
            self.rows.append(row)
            self.keys.add(key)

    def _load_rows(self):
        path = self.directory/"evaluations.jsonl"
        if not path.exists():
            return
        
        with path.open("rb+") as handle:
            valid_end = 0
            for raw in handle:
                if not raw.endswith(b"\n"):
                    handle.truncate(valid_end)
                    break
                self._remember(json.loads(raw))
                valid_end += len(raw)

    def observe(self, motion, area, evaluation, stage):
        x = features(motion)
        if self.models:
            mean, _ = self.predict(np.array([x]))
            self.last_prediction_error = float(abs(mean[0]-area))
        row = {"run_id": self.run_id, "seed": self.seed, "evaluation": evaluation,
               "stage": stage, "motion": motion.to_dict(), "features": x.tolist(),
               "sampled_area": area}
        self._remember(row)
        
        if self.journal:
            self.journal.write(json.dumps(row, allow_nan=False)+"\n")
            self.journal.flush()

    def maybe_fit(self, force=False):
        if self.cooldown and not force:
            return False
        n = len(self.rows)
        if n < self.minimum or (not force and self.models and n-self.last_fit_size < self.retrain_every):
            return False
        self.checkpoint()
        started = perf_counter()
        rows = self.rows[-2500:]
        x = np.array([r["features"] for r in rows])
        y = np.array([r["sampled_area"] for r in rows])
        
        held = np.array([int(self.key(v)[:8], 16) % 5 == 0 for v in x])
        if held.sum() < 8 or (~held).sum() < 24:
            self.report = {"status": "collecting held-out candidates", "ready": False}
            return False
        xm, xs = x[~held].mean(0), np.maximum(x[~held].std(0), .05)
        ym, ys = float(y[~held].mean()), max(float(y[~held].std()), .05)
        xx, yy = (x-xm)/xs, (y-ym)/ys
        models, losses, used_epochs = [], [], []
        train = np.flatnonzero(~held)
        with self.thread_limits(limits=1):
            for member in range(3):
                rng = np.random.default_rng(self.seed+member*1009+self.fit_count*7919)
                bootstrap = rng.choice(train, len(train), replace=True)
                mlp = self.mlp_class(hidden_layer_sizes=(48, 24), activation="tanh",
                    solver="adam", learning_rate_init=.003, alpha=.01,
                    batch_size=min(64, len(train)), random_state=int(rng.integers(2**31-1)))
                best_loss, best_weights, stale = np.inf, None, 0
                for epoch in range(self.epochs):
                    self.checkpoint()
                    mlp.partial_fit(xx[bootstrap], yy[bootstrap])
                    loss = float(np.mean((mlp.predict(xx[held])-yy[held])**2))
                    if loss < best_loss-1e-6:
                        best_loss, stale = loss, 0
                        best_weights = [(w.copy(), b.copy()) for w,b in zip(mlp.coefs_, mlp.intercepts_)]
                    else:
                        stale += 1
                    if epoch >= 24 and stale >= 12:
                        break
                if best_weights is None:
                    raise RuntimeError("NN training produced no finite model.")
                models.append(best_weights)
                losses.append(best_loss)
                used_epochs.append(epoch+1)
        self.models, self.scaling = models, (xm, xs, ym, ys)
        predicted, _ = self.predict(x[held])
        mse = float(np.mean((predicted-y[held])**2))
        base_mse = float(np.mean((ym-y[held])**2))
        rank_accuracy, rank_pairs = ranking_accuracy(y[held], predicted)
        ready = bool(np.isfinite(mse) and mse < .98*base_mse and rank_pairs >= 6
                     and rank_accuracy >= .60)
        self.fit_count += 1
        self.last_fit_size = n
        elapsed = perf_counter()-started
        self.training_seconds += elapsed
        self.report = {"status": "guiding" if ready else "trained; validation gate closed",
            "ready": ready, "training_rows": int((~held).sum()), "holdout_rows": int(held.sum()),
            "holdout_rmse": float(np.sqrt(mse)), "constant_baseline_rmse": float(np.sqrt(base_mse)),
            "top_half_ranking_accuracy": rank_accuracy, "ranking_pairs": rank_pairs,
            "epochs": used_epochs, "fit_seconds": elapsed,
            "note": "Candidate holdout selects epochs and gates guidance; not an independent test set. "
                    "Ensemble spread is a heuristic, not calibrated confidence."}
        if self.directory:
            self.save_model(self.directory/"model.npz")
            with (self.directory/"training.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"run_id": self.run_id, "fit": self.fit_count, **self.report})+"\n")
        return True

    def predict(self, x):
        xm, xs, ym, ys = self.scaling
        values = np.array([forward((x-xm)/xs, m)*ys+ym for m in self.models])
        return values.mean(0), values.std(0)

    def choose(self, original, population, costs, param, rng):
        if self.cooldown:
            self.cooldown -= 1
            return original, "evolution_fallback"
        if not self.report.get("ready"):
            return original, "evolution"
        
        if rng.random() < .30:
            self.exploration += 1
            return original, "exploration"
        span = param.hi-param.lo
        best = population[np.argmin(costs)]
        proposals = [original]
        for _ in range(15):
            if rng.random() < .65:
                child = best+rng.normal(0, rng.choice([.015, .04, .10]), len(best))*span
            else:
                a,b,c = population[rng.choice(len(population), 3, replace=False)]
                child = a+rng.uniform(.4,.9)*(b-c)
            proposals.append(np.clip(child, param.lo, param.hi))
        xx = np.array([features(param.decode(p)) for p in proposals])
        mean, spread = self.predict(xx)
        mean = mean-self.calibrator.loss(xx, mean)
        
        uncertain = rng.random() < .15
        scores = spread if uncertain else mean+.35*spread
        scores[~np.isfinite(scores)] = -np.inf
        selected = int(np.argmax(scores))
        self.guided += 1
        self.proposals += len(proposals)
        return proposals[selected], "nn_uncertainty" if uncertain else "nn_ranked"

    def record_outcome(self, source, gain):
        
        self.outcomes.append((source.startswith("nn_"), float(gain)))
        self.outcomes = self.outcomes[-80:]
        guided = [g for n,g in self.outcomes if n]
        ordinary = [g for n,g in self.outcomes if not n]
        if not self.cooldown and len(guided) >= 16 and len(ordinary) >= 16:
            if np.mean(guided) <= np.mean(ordinary)+1e-6:
                self.cooldown = 64
                self.fallbacks += 1
                self.outcomes.clear()

    def snapshot(self):
        return {**self.report, "unique_rows": len(self.rows), "loaded_rows": self.loaded_rows,
                "loaded_models": self.loaded_models, "fits_this_run": self.fit_count,
                "guided_evaluations": self.guided, "exploration_evaluations": self.exploration,
                "predictions_ranked": self.proposals, "training_seconds": self.training_seconds,
                "last_absolute_prediction_error": self.last_prediction_error,
                "fallback_count": self.fallbacks, "fallback_remaining": self.cooldown,
                "validated_loss_model": self.calibrator.report,
                "label": "sampled search area; not validated area",
                "directory": str(self.directory) if self.directory else None}

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xm,xs,ym,ys = self.scaling
        data = {"xm": xm, "xs": xs, "ym": np.array(ym), "ys": np.array(ys),
                "metadata": np.array(json.dumps({"schema": self.schema, "report": self.report,
                                                "rows_at_fit": self.last_fit_size}))}
        for i,model in enumerate(self.models):
            for j,(w,b) in enumerate(model):
                data[f"w{i}_{j}"], data[f"b{i}_{j}"] = w,b
        temp = path.with_suffix(".npz.tmp")
        with temp.open("wb") as f:
            np.savez_compressed(f, **data)
        os.replace(temp, path)

    def _load_model(self):
        path = self.directory/"model.npz"
        if not path.exists():
            return
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            if meta["schema"] != self.schema:
                raise ValueError("NN checkpoint schema does not match this geometry.")
            xm,xs,ym,ys = z["xm"],z["xs"],float(z["ym"]),float(z["ys"])
            if (xm.shape != (FEATURE_COUNT,) or xs.shape != xm.shape or
                    not np.isfinite(np.r_[xm,xs,ym,ys]).all() or np.any(xs <= 0) or ys <= 0):
                raise ValueError("Invalid NN checkpoint scaling.")
            models = []
            shapes = [(FEATURE_COUNT,48),(48,24),(24,1)]
            for i in range(3):
                weights = []
                for j,expected in enumerate(shapes):
                    w,b = z[f"w{i}_{j}"],z[f"b{i}_{j}"]
                    if w.shape != expected or b.shape != (expected[1],) or not np.isfinite(w).all() or not np.isfinite(b).all():
                        raise ValueError("Invalid NN checkpoint weights.")
                    weights.append((w,b))
                models.append(weights)
        self.models, self.scaling = models, (xm,xs,ym,ys)
        self.report = meta["report"]
        self.last_fit_size = int(meta["rows_at_fit"])
        self.loaded_models = len(models)

    def record_validation(self, result):
        by_key = {self.key(r["features"]): r for r in self.calibrator.rows}
        for item in result["finalists"]:
            if item["verification"]["status"] != "bounded_numerical_pass":
                continue
            x = features(Motion.from_dict(item["motion"]))
            by_key[self.key(x)] = {"features": x.tolist(), "sampled_area": item["search_area"],
                "validated_area": item["reconstructed_area"], "run_id": self.run_id,
                "target": self.schema["validation_target"]}
        self.calibrator.rows = list(by_key.values())[-2500:]
        self.calibrator.report["rows"] = len(self.calibrator.rows)
        
        if self.directory:
            save_json(self.directory/"validated_training.json", self.calibrator.rows)
            save_json(self.directory/("validation_"+self.run_id+".json"), {
                "run_id": self.run_id, "validated_area": result["validated_area"],
                "motion": result["motion"], "finalists": result["finalists"],
                "neural": result["neural"]})
