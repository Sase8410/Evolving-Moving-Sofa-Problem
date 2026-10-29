pass
import csv
import json
from dataclasses import replace, asdict
from pathlib import Path
import uuid
import numpy as np
from .search import optimize
from .baseline import optimize_v1
from .io import save_json, provenance


def benchmark(config, seeds, directory, ablations=False, progress=print,
              checkpoint=lambda: None, neural=False):
    config.validate()
    if seeds < 1:
        raise ValueError("Need at least one seed.")
    if config.time_limit and not neural:
        raise ValueError("Time-limited comparison supports V3 vs NN; use compare-nn.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    methods = ({"V3": {"surrogate": False}, "V3_NN": {"surrogate": True}}
               if neural else {"V1": {"surrogate": False}, "V3": {"surrogate": False}})
    if ablations:
        methods.update({"V3_no_local": {"local": False, "surrogate": False},
                        "V3_no_adaptive": {"adaptive": False, "surrogate": False},
                        "V3_fixed_angles": {"fixed_angles": True, "surrogate": False}})
    environment = provenance()
    manifest = {"config": asdict(config), "seeds": seeds, "methods": methods,
                "environment": environment,
                "budget_definition": ("Common cooperative total wall-time cap with 55% search allocation. "
                    "Only completed validations count; cap overruns are reported. " if config.time_limit else "")+
                    "Candidate objective evaluations; initial candidates included."
                    " Finalist reconstruction/verification and adaptive probes are separate overhead.",
                "comparison": ("V3 vs V3+NN, fresh learning per seed, identical spline/search settings, "
                    "no pretraining. Includes training and final verification runtime." if neural else
                    "Attached Step 8 candidate search; common V3 final verification."),
                "reference_shapes_in_discovery": False}
    manifest_path = directory/"manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            raise ValueError("Output directory belongs to a different configuration or source version. Choose a new folder.")
    else:
        save_json(manifest_path, manifest)
    records = []
    for offset in range(seeds):
        seed = config.seed+offset
        
        names = list(methods) if offset % 2 == 0 else list(reversed(methods))
        for method in names:
            checkpoint()
            cfg = replace(config, seed=seed, **methods[method])
            file = directory/f"{method}_seed{seed:03d}.json"
            if file.exists():
                result = json.loads(file.read_text())
            else:
                progress(f"Seed {offset+1}/{seeds}: {method}")
                if cfg.surrogate:
                    
                    
                    state_dir = directory/"learning"/(f"seed{seed:03d}_"+uuid.uuid4().hex[:8])
                    cfg = replace(cfg, nn_directory=str(state_dir), nn_reuse=False)
                solver = optimize_v1 if method == "V1" else optimize
                result = solver(cfg, checkpoint=checkpoint)
                result["experiment_method"] = method
                if cfg.time_limit is None and result["search_evaluations"] != cfg.budget:
                    raise RuntimeError("Search did not use the declared evaluation budget.")
                save_json(file, result)
            records.append(result)
    summary = summarize(records, directory)
    progress("Benchmark complete: "+str(directory/"summary.json"))
    return summary


def summarize(records, directory):
    directory = Path(directory)
    summary = {}
    for method in sorted({r["experiment_method"] for r in records}):
        group = [r for r in records if r["experiment_method"] == method]
        areas = [r["validated_area"] for r in group if r["validated_area"] is not None]
        summary[method] = {"runs": len(group), "feasible_runs": len(areas),
            "feasible_rate": len(areas)/len(group),
            "best_validated_area": max(areas) if areas else None,
            "mean_validated_area": float(np.mean(areas)) if areas else None,
            "std_validated_area": float(np.std(areas, ddof=1)) if len(areas)>1 else None,
            "median_validated_area": float(np.median(areas)) if areas else None,
            "mean_area_failures_as_zero": float(np.mean([r["validated_area"] or 0 for r in group])),
            "mean_search_seconds": float(np.mean([r["search_seconds"] for r in group])),
            "mean_total_seconds": float(np.mean([r["total_seconds"] for r in group])),
            "mean_evaluations": float(np.mean([r["search_evaluations"] for r in group])),
            "finalization_timeouts": sum(r.get("timing",{}).get("finalization_interrupted",False) for r in group),
            "max_overrun_seconds": max(r.get("timing",{}).get("overrun_seconds",0.) for r in group)}
    by_key = {(r["experiment_method"], r["settings"]["seed"]):r for r in records}
    for first, second in [("V1", "V3"), ("V3", "V3_NN")]:
        deltas = []
        for method, seed in by_key:
            if method == first and (second, seed) in by_key:
                a, b = by_key[(method, seed)], by_key[(second, seed)]
                if a["validated_area"] is not None and b["validated_area"] is not None:
                    deltas.append(b["validated_area"]-a["validated_area"])
        if deltas:
            rng = np.random.default_rng(20261025)
            means = rng.choice(deltas, (10000, len(deltas))).mean(axis=1)
            summary[f"paired_{second}_minus_{first}"] = {"valid_pairs": len(deltas),
                "mean_delta": float(np.mean(deltas)), "wins": sum(d>0 for d in deltas),
                "ties": sum(d==0 for d in deltas),
                "bootstrap_95_percent_interval": np.quantile(means,[.025,.975]).tolist(),
                "note": "Conditional on both runs passing; small-budget results do not establish convergence."}
    save_json(directory/"summary.json", summary)
    fields = ["method", "seed", "budget", "validated_area", "search_area", "search_seconds", "total_seconds", "status",
              "nn_fits", "nn_guided_evaluations", "nn_training_seconds", "nn_loaded_rows"]
    with (directory/"runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(dict(method=r["experiment_method"], seed=r["settings"]["seed"],
                budget=r["search_evaluations"], validated_area=r["validated_area"],
                search_area=r["best_search_area"], search_seconds=r["search_seconds"],
                total_seconds=r["total_seconds"], status=r["selected_finalist"]["verification"]["status"],
                nn_fits=r.get("neural",{}).get("fits_this_run",0),
                nn_guided_evaluations=r.get("neural",{}).get("guided_evaluations",0),
                nn_training_seconds=r.get("neural",{}).get("training_seconds",0),
                nn_loaded_rows=r.get("neural",{}).get("loaded_rows",0)))
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10, 4), layout="constrained")
    FigureCanvasAgg(fig)
    ax, areas_ax = fig.subplots(1, 2)
    labels, values = [], []
    for method in sorted({r["experiment_method"] for r in records}):
        group = [r for r in records if r["experiment_method"] == method]
        count = min(len(r["history"]) for r in group)
        curves = np.array([[h["best_search_area"] for h in r["history"][:count]] for r in group])
        x = np.arange(1, count+1)
        if count == 0:
            continue
        line, = ax.plot(x, curves.mean(axis=0), label=method)
        ax.fill_between(x, np.quantile(curves,.25,axis=0), np.quantile(curves,.75,axis=0),
                        alpha=.15, color=line.get_color())
        values.append([r["validated_area"] for r in group if r["validated_area"] is not None])
        labels.append(method)
    ax.set(title="Search area: common evaluation prefix", xlabel="Objective evaluations", ylabel="Sampled area (unverified)")
    ax.legend(fontsize=8)
    areas_ax.boxplot([v or [np.nan] for v in values], tick_labels=labels)
    areas_ax.tick_params(axis="x", rotation=25)
    areas_ax.set(title="Final area after common bounded validation", ylabel="Validated area")
    fig.savefig(directory/"comparison.png", dpi=160)
    return summary
