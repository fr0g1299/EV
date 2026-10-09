"""Run experiments for the binary genetic algorithm assignment."""

import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Callable

import numpy as np

from modules import parse_arguments, plot_convergence

Objective = Callable[[np.ndarray], int]


def one_max(chromosome: np.ndarray) -> int:
    return int(np.sum(chromosome))


def leading_ones(chromosome: np.ndarray) -> int:
    zero_positions = np.flatnonzero(chromosome == 0)
    return int(zero_positions[0]) if zero_positions.size else len(chromosome)


def select_parent(
    population: np.ndarray, fitness: np.ndarray, method: str, rng: np.random.Generator
) -> np.ndarray:
    if method == "rank":
        ranks = np.argsort(np.argsort(fitness)) + 1
        index = int(rng.choice(len(population), p=ranks / ranks.sum()))
    else:
        weights = fitness.astype(float) + 1e-12
        index = int(rng.choice(len(population), p=weights / weights.sum()))
    return population[index].copy()


def crossover(
    parent_a: np.ndarray, parent_b: np.ndarray, kind: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if kind == "uniform":
        mask = rng.random(len(parent_a)) < 0.5
        return np.where(mask, parent_a, parent_b), np.where(mask, parent_b, parent_a)
    point = int(rng.integers(1, len(parent_a)))
    return (
        np.concatenate((parent_a[:point], parent_b[point:])),
        np.concatenate((parent_b[:point], parent_a[point:])),
    )


def mutate(
    chromosome: np.ndarray, probability: float, rng: np.random.Generator
) -> np.ndarray:
    child = chromosome.copy()
    child[rng.random(len(child)) < probability] ^= 1
    return child


def run_ga(
    dimension: int,
    max_fes: int,
    objective: Objective,
    rng: np.random.Generator,
    population_size: int = 50,
    elitism_ratio: float = 0.10,
    crossover_ratio: float = 0.90,
    mutation_probability: float = 0.01,
    selection: str = "rank",
    crossover_type: str = "single-point",
) -> tuple[int, list[float]]:
    """Run one maximization experiment and return its best result and trace."""
    population = rng.integers(0, 2, size=(population_size, dimension), dtype=np.int8)
    fitness = np.array(
        [objective(chromosome) for chromosome in population], dtype=float
    )
    evaluations = min(population_size, max_fes)
    best = float(np.max(fitness[:evaluations]))
    trace = [best] * evaluations

    while evaluations < max_fes:
        elite_count = min(
            population_size - 1, max(1, round(elitism_ratio * population_size))
        )
        elite_indices = np.argsort(fitness)[-elite_count:][::-1]
        next_population = [population[index].copy() for index in elite_indices]
        while len(next_population) < population_size:
            parent_a = select_parent(population, fitness, selection, rng)
            parent_b = select_parent(population, fitness, selection, rng)
            if rng.random() < crossover_ratio:
                child_a, child_b = crossover(parent_a, parent_b, crossover_type, rng)
            else:
                child_a, child_b = parent_a, parent_b
            next_population.extend(
                (
                    mutate(child_a, mutation_probability, rng),
                    mutate(child_b, mutation_probability, rng),
                )
            )
        population = np.asarray(next_population[:population_size], dtype=np.int8)
        fitness = np.array(
            [objective(chromosome) for chromosome in population], dtype=float
        )
        for value in fitness:
            if evaluations >= max_fes:
                break
            best = max(best, float(value))
            trace.append(best)
            evaluations += 1
    return int(best), trace


def run_experiments(args) -> dict:
    objectives = {"one-max": one_max, "leading-ones": leading_ones}
    problems = list(objectives) if args.problem == "all" else [args.problem]
    results = {}
    for problem in problems:
        results[problem] = {}
        for dimension in args.dimensions:
            max_fes = args.max_fes_multiplier * dimension
            runs = []
            traces = []
            for run in range(args.n_runs):
                rng = np.random.default_rng(args.seed + run)
                best, trace = run_ga(
                    dimension,
                    max_fes,
                    objectives[problem],
                    rng,
                    args.population_size,
                    args.elitism_ratio,
                    args.crossover_ratio,
                    args.mutation_probability,
                    args.selection,
                    args.crossover,
                )
                runs.append(best)
                traces.append(trace)
            stats = {
                "best": max(runs),
                "worst": min(runs),
                "mean": mean(runs),
                "median": median(runs),
                "stddev": stdev(runs) if len(runs) > 1 else 0.0,
            }
            results[problem][f"D{dimension}"] = {
                "max_fes": max_fes,
                "statistics": stats,
                "avg_convergence": np.mean(np.asarray(traces), axis=0).tolist(),
            }
            print(
                f"{problem:13} D={dimension:3} | "
                + " | ".join(f"{key}={value:.2f}" for key, value in stats.items())
            )
    return results


def main() -> None:
    args = parse_arguments()
    results = run_experiments(args)
    if args.save_results:
        Path("results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print("Saved: results.json")
    if not args.no_plots:
        plot_convergence(results, args.dimensions)


if __name__ == "__main__":
    main()
