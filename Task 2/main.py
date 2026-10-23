"""Run experiments for the real-valued genetic algorithm assignment."""

import json
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

from modules import (
    parse_arguments,
    plot_convergence,
    run_ga,
    FUNCTION_MAP,
    VARIANT_NAMES,
)


def run_experiments(args) -> dict:
    """Run all configured experiments and return a nested results dictionary.

    Returns
    -------
    dict
        `results[function_name][variant_name]` containing:
        - `statistics`: best, worst, mean, median, stddev of final fitness
        - `avg_convergence`: element-wise mean of best-so-far traces
    """
    results = {}

    for func_name in args.functions:
        func = FUNCTION_MAP[func_name]
        results[func_name] = {}

        for variant in args.variants:
            runs_best = []
            traces = []

            for run in range(args.n_runs):
                rng = np.random.default_rng(args.seed + run)

                best_fitness, _best_solution, trace = run_ga(
                    variant=variant,
                    objective=func,
                    bounds=func.bounds,
                    dim=args.dimension,
                    max_fes=args.max_fes,
                    rng=rng,
                    population_size=args.population_size,
                    elitism_ratio=args.elitism_ratio,
                    crossover_ratio=args.crossover_ratio,
                    mutation_probability=args.mutation_probability,
                    tournament_size=args.tournament_size,
                    sigma=args.sigma,
                )
                runs_best.append(best_fitness)
                traces.append(trace)

            # Align traces to same length (use max_fes as target)
            max_len = max(len(t) for t in traces)
            aligned = []
            for t in traces:
                if len(t) < max_len:
                    t = t + [t[-1]] * (max_len - len(t))
                aligned.append(t[:max_len])

            stats = {
                "best": min(runs_best),
                "worst": max(runs_best),
                "mean": mean(runs_best),
                "median": median(runs_best),
                "stddev": stdev(runs_best) if len(runs_best) > 1 else 0.0,
            }

            results[func_name][variant] = {
                "statistics": stats,
                "avg_convergence": np.mean(np.asarray(aligned), axis=0).tolist(),
            }

            variant_display = VARIANT_NAMES.get(variant, variant)
            print(
                f"{func_name:13} | {variant_display:35} | "
                + " | ".join(f"{key}={value:.4f}" for key, value in stats.items())
            )

    return results


def main() -> None:
    args = parse_arguments()

    print("=" * 110)
    print(
        f"Real-Valued GA Optimization | D={args.dimension} | "
        f"FES={args.max_fes} | Runs={args.n_runs} | Pop={args.population_size}"
    )
    print("=" * 110)

    results = run_experiments(args)

    if args.save_results:
        Path("results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("\nSaved: results.json")

    if not args.no_plots:
        plot_convergence(results)


if __name__ == "__main__":
    main()
