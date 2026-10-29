pass
from dataclasses import dataclass, asdict
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from shapely.geometry import mapping
from shapely.geometry import Polygon
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
    diverse: bool = True
    progressive: bool = True
    diversity_distance: float = .015
    time_limit: float | None = None

    def validate(self):
        if self.budget < 20 or not 3 <= self.controls <= 17 or self.samples < 9:
            raise ValueError("Use budget >=20, 3-17 controls and >=9 search samples.")
        if not 1e-6 <= self.target <= .05 or not 1 <= self.finalists <= 5:
            raise ValueError("Invalid reconstruction tolerance or finalist count.")
        if self.seed < 0 or self.nn_minimum < 32 or self.nn_retrain < 1 or self.nn_epochs < 1:
            raise ValueError("Use a nonnegative seed, NN minimum >=32, retrain >=1 and epochs >=1.")
        if not 0 <= self.diversity_distance <= .2:
            raise ValueError("Diversity distance must be between 0 and .2.")
        if self.time_limit is not None and (not np.isfinite(self.time_limit) or self.time_limit < 1):
            raise ValueError("Total time limit must be at least one second.")


def motion_distance(a, b):
    pass
    grid = np.linspace(0, 1, 33)
    return float(np.sqrt(np.mean((a(grid)-b(grid))**2)))


def retain_diverse(archive, entry, capacity, distance):
    
    
    grid = np.linspace(0, 1, 33)
    new = entry[1](grid)
    close = [float(np.sqrt(np.mean((new-old[1](grid))**2))) <= distance for old in archive]
    if any(near and old[0] >= entry[0] for near,old in zip(close,archive)):
        return archive
    kept = [old for near,old in zip(close,archive) if not near]
    return sorted([*kept,entry],key=lambda p:-p[0])[:capacity]


class Parameterization:
    def __init__(self, motion, fixed_angles=False):
        self.base = motion
        mask = np.ones_like(motion.poses, dtype=bool)
        mask[0, 1:] = False  
        mask[-1, [0, 2]] = False  
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
        self.stage_deadline = float("inf")
        self.guide = None
        self.nn_rng = np.random.default_rng(config.seed+104729)
        self.candidate_source = "evolution"
        if config.surrogate:
            from .neural import NeuralGuide
            self.guide = NeuralGuide(config.seed, config.samples, config.nn_directory,
                config.nn_reuse, config.nn_minimum, config.nn_retrain, config.nn_epochs,
                self.search_checkpoint, target=config.target)

    def search_checkpoint(self):
        self.checkpoint()
        if perf_counter() >= self.stage_deadline:
            raise BudgetReached()

    def set_stage(self, name, fraction):
        self.stage = name
        self.stage_limit = max(16, int(self.cfg.budget*fraction))
        self.stage_deadline = (self.started + .55*self.cfg.time_limit*fraction
                               if self.cfg.time_limit else float("inf"))

    def evaluate(self, x, param):
        self.search_checkpoint()
        if self.n >= self.stage_limit:
            raise BudgetReached()
        motion = param.decode(x)
        sofa = reconstruct(full_motion(motion), self.cfg.samples, self.search_checkpoint)
        area = float(sofa.area)
        self.n += 1
        if self.guide:
            self.guide.observe(motion, area, self.n, self.stage)
        if area > self.best:
            self.best, self.best_motion = area, motion
        capacity = max(12, self.cfg.finalists*4) if self.cfg.progressive else self.cfg.finalists
        self.archive = retain_diverse(self.archive, (area, motion), capacity,
            self.cfg.diversity_distance if self.cfg.diverse else 0.)
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
                self.search_checkpoint()
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
                if self.guide:
                    self.guide.record_outcome(self.candidate_source, max(0., costs[i]-cost))
                if cost <= costs[i]:
                    population[i], costs[i] = child, cost

    def run(self):
        cfg = self.cfg
        param = Parameterization(seed_motion(cfg.controls), cfg.fixed_angles)
        self.set_stage("global", .55 if cfg.local else .8 if cfg.adaptive else 1.)
        try:
            self.global_search(param)
        except BudgetReached:
            pass
        if cfg.local and self.best_motion is not None:
            self.set_stage("local", .8 if cfg.adaptive else 1.)
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
        adaptive_probes = 0
        if cfg.adaptive and self.n < cfg.budget and self.best_motion is not None:
            self.set_stage("adaptive", 1.)
            
            try:
                sofa = reconstruct(full_motion(self.best_motion), cfg.samples, self.search_checkpoint)
                adaptive_probes += 1
                refined = self.best_motion.insert(contact_refinement_time(self.best_motion, sofa))
                param = Parameterization(refined, cfg.fixed_angles)
                if cfg.local:
                    while self.n < cfg.budget:
                        before = self.n
                        minimize(lambda x: self.evaluate(x, param),
                            param.encode(refined), method="Powell",
                            bounds=list(zip(param.lo, param.hi)),
                            options={"maxfev": cfg.budget-self.n, "xtol": 1e-4, "ftol": 1e-6})
                        if len(self.best_motion.times) == len(refined.times):
                            refined = self.best_motion
                        if self.n == before:
                            break
                else:
                    self.global_search(param)
            except BudgetReached:
                pass
        if self.guide and cfg.time_limit is None:
            
            self.guide.maybe_fit()
        extra = {"adaptive_probe_reconstructions": adaptive_probes}
        if self.guide:
            extra["neural"] = self.guide.snapshot()
        result = finalize(self.archive, cfg, self.history, self.started, self.checkpoint,
                        method="V3_NN" if self.guide else "V3", extra=extra)
        if self.guide:
            self.guide.record_validation(result)
            result["neural"] = self.guide.snapshot()
        return result


def finalize(archive, cfg, history, started, checkpoint, method, extra=None, parts_fn=full_motion):
    search_seconds = perf_counter()-started
    deadline = started+cfg.time_limit if cfg.time_limit else float("inf")
    def final_checkpoint():
        checkpoint()
        if perf_counter() >= deadline:
            raise BudgetReached()
    checks, winners, promotions = [], [], []
    candidates = list(archive)
    timed_out = False
    
    
    if cfg.progressive and len(candidates) > cfg.finalists:
        try:
            for samples, keep in [(2*(cfg.samples-1)+1, max(6, cfg.finalists*2)),
                                  (4*(cfg.samples-1)+1, cfg.finalists)]:
                rescored = []
                for coarse, turn in candidates:
                    sofa = reconstruct(parts_fn(turn), samples, final_checkpoint)
                    rescored.append((float(sofa.area), coarse, turn))
                rescored.sort(key=lambda x: -x[0])
                promotions.append({"samples": samples, "evaluations": len(rescored),
                    "scores": [x[0] for x in rescored], "retained": min(keep,len(rescored))})
                candidates = [(coarse, turn) for _,coarse,turn in rescored[:keep]]
        except BudgetReached:
            timed_out = True
    for coarse, turn in candidates[:cfg.finalists]:
        try:
            final_checkpoint()
            parts = parts_fn(turn)
            sofa, construction = reconstruct_conservative(parts, cfg.target, final_checkpoint)
            verification = verify(sofa, parts, checkpoint=final_checkpoint)
        except BudgetReached:
            timed_out = True
            break
        check = {"search_area": coarse, "reconstructed_area": float(sofa.area),
                 "motion": turn.to_dict(), "construction": construction, "verification": verification}
        checks.append(check)
        winners.append((verification["status"] == "bounded_numerical_pass",
                        sofa.area, sofa, turn, check))
    if winners:
        passed, _, sofa, turn, check = max(winners, key=lambda x: (x[0], x[1]))
    else:
        passed, sofa = False, Polygon()
        turn = archive[0][1] if archive else seed_motion(cfg.controls)
        check = {"verification": {"status": "time_limit" if timed_out else "no_candidate"}}
    elapsed = perf_counter()-started
    result = {"version": "3.2.0", "method": method,
              "settings": asdict(cfg), "search_evaluations": len(history),
              "search_seconds": search_seconds, "total_seconds": elapsed,
              "validated_area": float(sofa.area) if passed else None,
              "geometry": mapping(sofa), "motion": turn.to_dict(),
              "full_motion": [m.to_dict() for m in parts_fn(turn)],
              "best_search_area": max((x["best_search_area"] for x in history), default=0.),
              "selected_finalist": check, "finalists": checks, "history": history,
              "selection": {"archive_size": len(archive), "promotions": promotions,
                  "promotion_evaluations": sum(p["evaluations"] for p in promotions),
                  "diversity_distance": cfg.diversity_distance if cfg.diverse else 0.,
                  "final_ranking": "passed bounded verification, then reconstructed area"},
              "timing": {"limit_seconds": cfg.time_limit, "finalization_interrupted": timed_out,
                  "search_fraction": .55 if cfg.time_limit else None,
                  "limit_kind": "cooperative wall time; atomic operations and output may overrun",
                  "overrun_seconds": max(0., elapsed-cfg.time_limit) if cfg.time_limit else 0.},
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
