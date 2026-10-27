import argparse
from pathlib import Path
from sofa_v3.search import Config, optimize
from sofa_v3.io import save_json, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="gui", choices=["gui", "search", "reference", "benchmark", "compare-nn"])
    parser.add_argument("--budget", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--controls", type=int, default=5)
    parser.add_argument("--target", type=float, default=.002)
    parser.add_argument("--finalists", type=int, default=3)
    parser.add_argument("--surrogate", action="store_true")
    parser.add_argument("--nn-dir", type=Path, help="Persistent NN evaluations and numeric model checkpoint.")
    parser.add_argument("--fresh-nn", action="store_true", help="Start fresh; requires a new learning directory.")
    parser.add_argument("--fixed-angles", action="store_true", help="Keep angle(t) linear; optimize translation only.")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable the extra contact-guided spline knot.")
    parser.add_argument("--no-local", action="store_true", help="Use differential evolution without Powell refinement.")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mode == "gui":
        from sofa_v3.gui import main as gui_main
        return gui_main()
    nn_dir = args.nn_dir
    if args.surrogate and args.mode == "search" and nn_dir is None:
        from sofa_v3.neural import signature
        import hashlib, json, uuid
        tag = hashlib.sha256(json.dumps(signature(49),sort_keys=True).encode()).hexdigest()[:10]
        nn_dir = Path("results/nn_learning")/(uuid.uuid4().hex if args.fresh_nn else tag)
    cfg = Config(seed=args.seed, budget=args.budget, controls=args.controls,
                 target=args.target, finalists=args.finalists, surrogate=args.surrogate,
                 nn_directory=str(nn_dir) if nn_dir else None, nn_reuse=not args.fresh_nn,
                 fixed_angles=args.fixed_angles, adaptive=not args.no_adaptive,
                 local=not args.no_local)
    if args.mode == "reference":
        from sofa_v3.reference import reference_check
        from shapely.geometry import mapping
        result, sofa = reference_check(args.target)
        result["geometry"] = mapping(sofa)
    elif args.mode in ("benchmark", "compare-nn"):
        from sofa_v3.experiments import benchmark
        return benchmark(cfg, args.seeds, args.output or Path("results")/args.mode, args.ablations,
                         neural=args.mode == "compare-nn")
    else:
        result = optimize(cfg, progress=lambda row: print(
            f"{row['evaluation']:5d}  {row['stage']:9s}  sampled area {row['best_search_area']:.6f}", flush=True))
    result["environment"] = provenance()
    path = args.output or Path("results")/(args.mode+f"_seed{args.seed}.json")
    save_json(path, result)
    print(f"Saved {path}")
    print("Validated area:", result.get("validated_area"))


if __name__ == "__main__":
    main()
