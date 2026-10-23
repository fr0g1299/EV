"""
CLI Module

Contains command-line argument parsing and configuration utilities
for the real-valued genetic algorithm optimization project.
"""

import argparse

ALL_FUNCTIONS = [
    "sphere",
    "rastrigin",
    "ackley",
    "zakharov",
    "michalewicz",
    "rosenbrock",
    "schwefel",
]

ALL_VARIANTS = [
    "ieee754",
    "fixed_point",
    "bcd",
    "real_gaussian",
    "real_uniform",
]


def parse_arguments():
    """Parse command-line arguments for real-valued GA optimization."""
    parser = argparse.ArgumentParser(
        description="Genetic algorithm for real-valued parameter optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-f",
        "--functions",
        type=str,
        nargs="+",
        choices=ALL_FUNCTIONS + ["all"],
        default=["schwefel", "sphere", "rosenbrock"],
        help="Benchmark functions to test",
    )

    parser.add_argument(
        "-v",
        "--variants",
        type=str,
        nargs="+",
        choices=ALL_VARIANTS + ["all"],
        default=["all"],
        help="GA variants to test",
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
        "--dimension",
        type=int,
        default=10,
        help="Dimensionality of the search space",
    )

    parser.add_argument(
        "--max-fes",
        type=int,
        default=10000,
        help="Maximum function evaluations",
    )

    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )

    parser.add_argument(
        "--population-size",
        type=int,
        default=100,
        help="Population size",
    )

    parser.add_argument(
        "--elitism-ratio",
        type=float,
        default=0.10,
        help="Fraction of elites preserved each generation",
    )

    parser.add_argument(
        "--crossover-ratio",
        type=float,
        default=0.90,
        help="Crossover probability",
    )

    parser.add_argument(
        "--mutation-probability",
        type=float,
        default=0.05,
        help="Mutation probability (per bit for binary, per dimension for real)",
    )

    parser.add_argument(
        "--tournament-size",
        type=int,
        default=3,
        help="Tournament selection size",
    )

    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Standard deviation for Gaussian mutation (as fraction of search range). "
        "If not specified, defaults to 10%% of the search range.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )

    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results to JSON",
    )

    args = parser.parse_args()

    # Validation
    errors = []

    if args.n_runs < 1:
        errors.append("--n-runs must be >= 1")

    if args.dimension < 2:
        errors.append("--dimension must be >= 2")

    if args.population_size < 2:
        errors.append("--population-size must be >= 2")

    if not (0 <= args.elitism_ratio <= 1):
        errors.append("--elitism-ratio must be between 0 and 1")

    if not (0 <= args.crossover_ratio <= 1):
        errors.append("--crossover-ratio must be between 0 and 1")

    if not (0 <= args.mutation_probability <= 1):
        errors.append("--mutation-probability must be between 0 and 1")

    if args.max_fes < 1:
        errors.append("--max-fes must be >= 1")

    if args.tournament_size < 2:
        errors.append("--tournament-size must be >= 2")

    if args.sigma is not None and args.sigma <= 0:
        errors.append("--sigma must be > 0 when specified")

    if errors:
        parser.error("\n  ".join(["Invalid arguments:"] + errors))

    # Post-processing
    if "all" in args.functions:
        args.functions = list(ALL_FUNCTIONS)

    if "all" in args.variants:
        args.variants = list(ALL_VARIANTS)

    return args
