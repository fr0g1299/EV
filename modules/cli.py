"""
CLI Module

Contains command-line argument parsing and configuration utilities.
"""

import argparse


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Binary genetic algorithm for One-Max and LeadingOnes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        choices=["one-max", "leading-ones", "all"],
        default="all",
        help="Binary maximization problem to solve",
    )

    parser.add_argument(
        "-n",
        "--n-runs",
        type=int,
        default=10,
        help="Number of independent runs per configuration",
    )

    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        nargs="+",
        default=[10, 30, 100],
        help="Binary string lengths to test (e.g., -d 10 30 100)",
    )

    parser.add_argument(
        "-m",
        "--max-fes-multiplier",
        type=int,
        default=100,
        help="Maximum objective evaluations per dimension",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=77777,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=50,
        help="Number of chromosomes in the population",
    )

    parser.add_argument(
        "--elitism-ratio",
        type=float,
        default=0.10,
        help="Fraction of the best population copied to the next generation",
    )

    parser.add_argument(
        "--crossover-ratio",
        type=float,
        default=0.90,
        help="Probability that selected parents undergo crossover",
    )

    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.01,
        help="Probability of flipping each chromosome bit",
    )

    parser.add_argument(
        "--selection",
        choices=["roulette", "rank"],
        default="rank",
        help="Parent selection method",
    )

    parser.add_argument(
        "--crossover",
        choices=["single-point", "uniform"],
        default="single-point",
        help="Crossover operator; single-point is specified by the task",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save experiment results as results.json",
    )

    args = parser.parse_args()
    if args.n_runs < 1 or any(d < 2 for d in args.dimensions):
        parser.error("n-runs must be positive and dimensions must be at least 2")
    if args.population_size < 2:
        parser.error("population-size must be at least 2")
    if not 0 <= args.elitism_ratio <= 1 or not 0 <= args.crossover_ratio <= 1:
        parser.error("elitism-ratio and crossover-ratio must be between 0 and 1")
    if not 0 <= args.mutation_probability <= 1:
        parser.error("mutation-probability must be between 0 and 1")
    if args.max_fes_multiplier < 1:
        parser.error("max-fes-multiplier must be positive")
    return args
