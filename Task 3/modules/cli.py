"""
CLI Module

Contains command-line argument parsing and configuration utilities
for the Differential Evolution (DE) optimization project.
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
    "rand_1",
    "best_1",
    "jde",
]


def parse_arguments():
    """Parse command-line arguments for Differential Evolution optimization."""
    parser = argparse.ArgumentParser(
        description="Differential Evolution (DE) for continuous parameter optimization",
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
        help="DE variants to test: rand_1, best_1, jde, or all",
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
        help="Population size (NP)",
    )

    parser.add_argument(
        "-F",
        "--f",
        type=float,
        default=0.5,
        help="Mutation scale factor F for Rand/1 and Best/1 (and initial F for jDE)",
    )

    parser.add_argument(
        "--cr",
        type=float,
        default=0.8,
        help="Crossover probability CR for Rand/1 and Best/1",
    )

    parser.add_argument(
        "--jde-cr",
        type=float,
        default=0.9,
        help="Initial crossover probability CR for jDE",
    )

    parser.add_argument(
        "--tau1",
        type=float,
        default=0.1,
        help="Probability of adaptation for scale factor F in jDE",
    )

    parser.add_argument(
        "--tau2",
        type=float,
        default=0.1,
        help="Probability of adaptation for crossover rate CR in jDE",
    )

    parser.add_argument(
        "--f-lower",
        type=float,
        default=0.1,
        help="Lower bound for adapted F in jDE",
    )

    parser.add_argument(
        "--f-upper",
        type=float,
        default=0.9,
        help="Upper bound for adapted F in jDE",
    )

    parser.add_argument(
        "--cr-lower",
        type=float,
        default=0.0,
        help="Lower bound for adapted CR in jDE",
    )

    parser.add_argument(
        "--cr-upper",
        type=float,
        default=1.0,
        help="Upper bound for adapted CR in jDE",
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

    if args.population_size < 4:
        errors.append("--population-size must be >= 4 (DE requires at least 4 vectors)")

    if not 0 <= args.cr <= 1:
        errors.append("--cr must be between 0 and 1")

    if not 0 <= args.jde_cr <= 1:
        errors.append("--jde-cr must be between 0 and 1")

    if args.f <= 0:
        errors.append("-F/--f must be > 0")

    if not 0 <= args.tau1 <= 1:
        errors.append("--tau1 must be between 0 and 1")

    if not 0 <= args.tau2 <= 1:
        errors.append("--tau2 must be between 0 and 1")

    if not 0 <= args.f_lower <= args.f_upper:
        errors.append("0 <= --f-lower <= --f-upper required")

    if not 0 <= args.cr_lower <= args.cr_upper <= 1:
        errors.append("0 <= --cr-lower <= --cr-upper <= 1 required")

    if args.max_fes < 1:
        errors.append("--max-fes must be >= 1")

    if errors:
        parser.error("\n  ".join(["Invalid arguments:"] + errors))

    # Post-processing
    if "all" in args.functions:
        args.functions = list(ALL_FUNCTIONS)

    if "all" in args.variants:
        args.variants = list(ALL_VARIANTS)

    return args
