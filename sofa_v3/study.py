pass
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import hashlib
import json
from shapely.geometry import mapping
from .motion import Motion
from .geometry import reconstruct_conservative
from .verify import verify
from .experiments import benchmark
from .io import save_json, load_result, provenance


def convergence(source, output, targets=(.004, .002, .001, .0005),
                checkpoint=lambda: None, progress=print):
    source, output = Path(source), Path(output)
    data = load_result(source)
    parts = [Motion.from_dict(p) for p in data["full_motion"]]
    rows, best, last_area = [], None, None
    for target in targets:
        checkpoint()
        progress(f"Convergence: reconstructing at target {target:g}")
        started = perf_counter()
        sofa, construction = reconstruct_conservative(parts, target, checkpoint)
        verification = verify(sofa, parts, checkpoint=checkpoint)
        passed = verification["status"] == "bounded_numerical_pass"
        area = float(sofa.area) if passed else None
        rows.append({"target": target, "validated_area": area,
            "change_from_previous": area-last_area if area is not None and last_area is not None else None,
            "seconds": perf_counter()-started, "construction": construction, "verification": verification})
        if area is not None and (best is None or area > best["validated_area"]):
            
            best = {"version": "3.2.0", "method": "fixed_motion_reconstruction",
                "source_file": str(source), "motion": data["motion"], "full_motion": data["full_motion"],
                "geometry": mapping(sofa), "validated_area": area,
                "selected_finalist": {"construction": construction, "verification": verification},
                "target": target, "environment": provenance()}
        last_area = area
    best_path = output.with_name(output.stem+"_best.json")
    if best:
        save_json(best_path, best)
    report = {"source_file": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "kind": "fixed trajectory; reconstruction convergence, not optimizer convergence",
        "rows": rows, "best_result": str(best_path) if best else None,
        "limitation": "Successive area differences do not certify the remaining error or global optimality.",
        "environment": provenance()}
    save_json(output, report)
    return report


def final_study(config, seeds, directory, seconds=30., progress=print, checkpoint=lambda: None):
    directory = Path(directory)
    cfg = replace(config, time_limit=None, nn_directory=None, nn_reuse=False)
    progress("Final study: equal candidate-evaluation budgets")
    evaluation = benchmark(cfg, seeds, directory/"equal_evaluations", neural=True,
                           progress=progress, checkpoint=checkpoint)
    progress("Final study: common total runtime cap (includes training and validation)")
    timed_cfg = replace(cfg, time_limit=seconds, budget=max(100000, cfg.budget))
    timed = benchmark(timed_cfg, seeds, directory/"equal_time", neural=True,
                      progress=progress, checkpoint=checkpoint)
    candidates = []
    for folder in ("equal_evaluations", "equal_time"):
        for path in (directory/folder).glob("V3*_seed*.json"):
            data = json.loads(path.read_text())
            if data["validated_area"] is not None:
                candidates.append((data["validated_area"], path))
    convergence_report = audit_report = None
    if candidates:
        best_file = max(candidates, key=lambda p:p[0])[1]
        convergence_report = convergence(best_file, directory/"convergence.json",
                                          checkpoint=checkpoint, progress=progress)
        if convergence_report["best_result"]:
            from audit_export import audit
            progress("Final study: independent shifted-grid audit")
            audit_report = audit(convergence_report["best_result"], 2001, checkpoint)
            save_json(directory/"audit.json", audit_report)
    report = {"evaluation_comparison": evaluation, "time_comparison": timed,
        "convergence": convergence_report, "audit": audit_report,
        "recommendation": "V3 remains default; NN is experimental unless independent evidence supports promotion.",
        "limits": ["Fresh learning per seed; does not measure warmed-up validation-loss calibration.",
                   "Time caps are cooperative; atomic operations and file output can overrun.",
                   "Convergence refines a selected fixed motion, not a proof of optimality."]}
    save_json(directory/"final_summary.json", report)
    progress("Final study saved to "+str(directory/"final_summary.json"))
    return report
