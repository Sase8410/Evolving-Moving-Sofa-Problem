from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import mapping
from .motion import Motion, seed_motion, full_motion
from .geometry import (reconstruct, reconstruct_conservative,
                       contact_refinement_time)
from .verify import verify


@dataclass
class Config:
    seed: int = 0
    budget: int = 1500
    controls: int = 5
    samples: int = 49
    target: float = .002
    local: bool = True
    adaptive: bool = True
    fixed_angles: bool = False
    surrogate: bool = False
    finalists: int = 3
    nn_directory: str | None = None
    nn_reuse: bool = True
    nn_minimum: int = 64
    nn_retrain: int = 64
    nn_epochs: int = 100

    def validate(self):
        if self.budget < 20 or not 3 <= self.controls <= 17 or self.samples < 9:
            raise ValueError("Use budget >=20, 3-17 controls and >=9 search samples.")
        if not 1e-6 <= self.target <= .05 or not 1 <= self.finalists <= 5:
            raise ValueError("Invalid reconstruction tolerance or finalist count.")
        if self.seed < 0 or self.nn_minimum < 32 or self.nn_retrain < 1 or self.nn_epochs < 1:
            raise ValueError("Use a nonnegative seed, NN minimum >=32, retrain >=1 and epochs >=1.")


class Parameterization:
    def __init__(self, motion, fixed_angles=False):
        self.base = motion
        mask = np.ones_like(motion.poses, dtype=bool)
        mask[0, 1:] = False  # initial y=1, angle=0
        mask[-1, [0, 2]] = False  # final x=0, angle=pi/2
        if fixed_angles:
            mask[:, 2] = False
        self.mask = mask
        lo = np.tile([-1.5, -.5, -.4], (len(mask), 1))
        hi = np.tile([1.5, 2.5, np.pi/2+.4], (len(mask), 1))
        self.lo, self.hi = lo[mask], hi[mask]

    def encode(self, motion):
        return motion.poses[self.mask]

    def decode(self, x):
        poses = self.base.poses.copy()
        poses[self.mask] = x
        return Motion(self.base.times, poses)


class BudgetReached(Exception):
    pass


class Search:
    def __init__(self, config, progress=lambda row: None, checkpoint=lambda: None):
        config.validate()
        self.cfg, self.progress, self.checkpoint = config, progress, checkpoint
        self.rng = np.random.default_rng(config.seed)
        self.n = 0
        self.best = -1.
        self.best_motion = None
        self.archive, self.history = [], []
        self.started = perf_counter()
        self.stage_limit = config.budget
        self.stage = "global"
        self.guide = None
        self.nn_rng = np.random.default_rng(config.seed+104729)
        self.candidate_source = "evolution"
        if config.surrogate:
            from .neural import NeuralGuide
            self.guide = NeuralGuide(config.seed, config.samples, config.nn_directory,
                config.nn_reuse, config.nn_minimum, config.nn_retrain, config.nn_epochs,
                checkpoint)

    def evaluate(self, x, param):
        self.checkpoint()
        if self.n >= self.stage_limit:
            raise BudgetReached()
        motion = param.decode(x)
        sofa = reconstruct(full_motion(motion), self.cfg.samples, self.checkpoint)
        area = float(sofa.area)
        self.n += 1
        if self.guide:
            self.guide.observe(motion, area, self.n, self.stage)
        if area > self.best:
            self.best, self.best_motion = area, motion
        self.archive.append((area, motion))
        self.archive.sort(key=lambda pair: -pair[0])
        seen, unique = set(), []
        for entry in self.archive:
            key = (tuple(entry[1].times), entry[1].poses.tobytes())
            if key not in seen:
                unique.append(entry)
                seen.add(key)
        self.archive = unique[:self.cfg.finalists]
        row = {"evaluation": self.n, "best_search_area": self.best,
               "candidate_area": area, "seconds": perf_counter()-self.started,
               "stage": self.stage, "controls": len(motion.times)}
        if self.guide:
            row["candidate_source"] = self.candidate_source if self.stage == "global" else self.stage
            row["neural"] = self.guide.snapshot()
        self.history.append(row)
        if self.n == 1 or self.n % 10 == 0:
            self.progress(row)
        return -area

    def global_search(self, param):
        dimension = len(param.lo)
        popsize = min(max(16, 2*dimension), max(5, self.stage_limit-self.n))
        base = param.encode(param.base)
        span = param.hi-param.lo
        population = np.clip(base+self.rng.normal(0, .12, (popsize, dimension))*span,
                             param.lo, param.hi)
        population[0] = base
        population[-max(1, popsize//4):] = self.rng.uniform(param.lo, param.hi,
                                              (max(1, popsize//4), dimension))
        costs = np.array([self.evaluate(x, param) for x in population])
        while self.n < self.stage_limit:
            for i in range(popsize):
                self.checkpoint()
                if self.n >= self.stage_limit:
                    raise BudgetReached()
                if self.guide:
                    self.guide.maybe_fit()
                choices = np.delete(np.arange(popsize), i)
                a, b, c = self.rng.choice(choices, 3, replace=False)
                f = self.rng.uniform(.45, .9)
                anchor = population[np.argmin(costs)] if self.rng.random() < .6 else population[a]
                mutant = np.clip(anchor+f*(population[b]-population[c]), param.lo, param.hi)
                cross = self.rng.random(dimension) < .8
                cross[self.rng.integers(dimension)] = True
                child = np.where(cross, mutant, population[i])
                self.candidate_source = "evolution"
                if self.guide:
                    child, self.candidate_source = self.guide.choose(
                        child, population, costs, param, self.nn_rng)
                cost = self.evaluate(child, param)
                if cost <= costs[i]:
                    population[i], costs[i] = child, cost

    def run(self):
        cfg = self.cfg
        param = Parameterization(seed_motion(cfg.controls), cfg.fixed_angles)
        self.stage_limit = max(16, int(cfg.budget*.55)) if cfg.local else cfg.budget
        if not cfg.local and cfg.adaptive:
            self.stage_limit = max(16, int(cfg.budget*.8))
        try:
            self.global_search(param)
        except BudgetReached:
            pass
        if cfg.local:
            self.stage = "local"
            self.stage_limit = int(cfg.budget*.8) if cfg.adaptive else cfg.budget
            while self.n < self.stage_limit:
                before = self.n
                try:
                    minimize(lambda x: self.evaluate(x, param),
                        param.encode(self.best_motion), method="Powell",
                        bounds=list(zip(param.lo, param.hi)),
                        options={"maxfev": self.stage_limit-self.n, "xtol": 1e-4, "ftol": 1e-6})
                except BudgetReached:
                    break
                if self.n == before:
                    break
        if cfg.adaptive and self.n < cfg.budget:
            self.stage = "adaptive"
            sofa = reconstruct(full_motion(self.best_motion), cfg.samples, self.checkpoint)
            refined = self.best_motion.insert(contact_refinement_time(self.best_motion, sofa))
            param = Parameterization(refined, cfg.fixed_angles)
            self.stage_limit = cfg.budget
            try:
                if cfg.local:
                    while self.n < cfg.budget:
                        minimize(lambda x: self.evaluate(x, param),
                            param.encode(refined), method="Powell",
                            bounds=list(zip(param.lo, param.hi)),
                            options={"maxfev": cfg.budget-self.n, "xtol": 1e-4, "ftol": 1e-6})
                        if len(self.best_motion.times) == len(refined.times):
                            refined = self.best_motion
                else:
                    self.global_search(param)
            except BudgetReached:
                pass
        if self.guide:
            self.guide.maybe_fit()
        extra = {"adaptive_probe_reconstructions": int(cfg.adaptive)}
        if self.guide:
            extra["neural"] = self.guide.snapshot()
        result = finalize(self.archive, cfg, self.history, self.started, self.checkpoint,
                        method="V3_NN" if self.guide else "V3", extra=extra)
        if self.guide:
            self.guide.record_validation(result)
        return result


def finalize(archive, cfg, history, started, checkpoint, method, extra=None, parts_fn=full_motion):
    search_seconds = perf_counter()-started
    checks, winners = [], []
    for coarse, turn in archive:
        checkpoint()
        parts = parts_fn(turn)
        sofa, construction = reconstruct_conservative(parts, cfg.target, checkpoint)
        verification = verify(sofa, parts, checkpoint=checkpoint)
        check = {"search_area": coarse, "reconstructed_area": float(sofa.area),
                 "construction": construction, "verification": verification}
        checks.append(check)
        winners.append((verification["status"] == "bounded_numerical_pass",
                        sofa.area, sofa, turn, check))
    passed, _, sofa, turn, check = max(winners, key=lambda x: (x[0], x[1]))
    result = {"version": "3.1.0-neural-experimental", "method": method,
              "settings": asdict(cfg), "search_evaluations": len(history),
              "search_seconds": search_seconds, "total_seconds": perf_counter()-started,
              "validated_area": float(sofa.area) if passed else None,
              "geometry": mapping(sofa), "motion": turn.to_dict(),
              "full_motion": [m.to_dict() for m in parts_fn(turn)],
              "best_search_area": max(x["best_search_area"] for x in history),
              "selected_finalist": check, "finalists": checks, "history": history,
              "limitations": ["Floating-point bounds, not an exact mathematical certificate.",
                              "Finite hallway, bounded translations and finite spline controls.",
                              "No claim of global convergence or independent Gerver reproduction."],
              **(extra or {})}
    return result


def optimize(config=None, progress=lambda row: None, checkpoint=lambda: None):
    search = Search(config or Config(), progress, checkpoint)
    try:
        return search.run()
    finally:
        if search.guide:
            search.guide.close()
