pass
import hashlib
import numpy as np


def descriptors(features, sampled):
    p = np.asarray(features).reshape(-1, 17, 3)
    delta = np.diff(p, axis=1)
    return np.c_[np.asarray(sampled).reshape(-1),
                 np.linalg.norm(delta[:, :, :2], axis=2).sum(1),
                 abs(delta[:, :, 2]).sum(1),
                 np.linalg.norm(np.diff(delta, axis=1), axis=2).sum(1)]


class ValidationCalibrator:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.model = None
        self.report = {"ready": False, "rows": len(self.rows), "status": "collecting validated finalists"}

    def fit(self):
        if len(self.rows) < 32:
            return
        x = descriptors([r["features"] for r in self.rows], [r["sampled_area"] for r in self.rows])
        y = np.array([r["sampled_area"]-r["validated_area"] for r in self.rows])
        held = np.array([int(hashlib.sha256(np.round(r["features"],10).astype("<f8").tobytes()).hexdigest()[:8],16)%5==0 for r in self.rows])
        if held.sum() < 8 or (~held).sum() < 20:
            return
        mean, scale = x[~held].mean(0), np.maximum(x[~held].std(0), 1e-3)
        design = np.c_[np.ones(len(x)), (x-mean)/scale]
        penalty = np.diag([0., 1., 1., 1., 1.])
        coeff = np.linalg.solve(design[~held].T@design[~held]+penalty,
                                design[~held].T@y[~held])
        pred = np.clip(design[held]@coeff, 0., .5)
        mse = float(np.mean((pred-y[held])**2))
        baseline = float(np.mean((y[~held].mean()-y[held])**2))
        ready = bool(np.isfinite(mse) and mse < .95*baseline)
        self.report = {"ready": ready, "rows": len(x), "holdout_rows": int(held.sum()),
            "holdout_rmse": float(np.sqrt(mse)), "constant_baseline_rmse": float(np.sqrt(baseline)),
            "status": "correcting predicted reconstruction loss" if ready else "validation-loss gate closed",
            "note": "Within-dataset holdout gate; not an independent performance test."}
        self.model = (mean, scale, coeff) if ready else None

    def loss(self, features, sampled):
        if self.model is None:
            return np.zeros(len(features))
        mean, scale, coeff = self.model
        x = descriptors(features, sampled)
        return np.clip(np.c_[np.ones(len(x)), (x-mean)/scale]@coeff, 0., .5)
