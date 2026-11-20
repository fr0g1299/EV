"""
CLI Module

Contains command-line argument parsing and configuration utilities
for the Particle Swarm Optimization (PSO) project.
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
    "linear_star",
    "const_star",
    "const_ring",
]


def parse_arguments():
    """Parse command-line arguments for Particle Swarm Optimization."""
    parser = argparse.ArgumentParser(
        description="Particle Swarm Optimization (PSO) for continuous parameter optimization",
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
        help="PSO variants to test: linear_star, const_star, const_ring, or all",
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
        default=50,
        help="Swarm population size (NP, recommended 20-100)",
    )

    parser.add_argument(
        "--w-start",
        type=float,
        default=0.8,
        help="Starting inertia weight for linearly decreasing variant",
    )

    parser.add_argument(
        "--w-end",
        type=float,
        default=0.3,
        help="Ending inertia weight for linearly decreasing variant",
    )

    parser.add_argument(
        "--w-const",
        type=float,
        default=0.7,
        help="Inertia weight for constant inertia variants",
    )

    parser.add_argument(
        "--c1-linear",
        type=float,
        default=2.0,
        help="Cognitive acceleration coefficient for linear variant",
    )

    parser.add_argument(
        "--c2-linear",
        type=float,
        default=2.0,
        help="Social acceleration coefficient for linear variant",
    )

    parser.add_argument(
        "--c1-const",
        type=float,
        default=1.49618,
        help="Cognitive acceleration coefficient for constant variants",
    )

    parser.add_argument(
        "--c2-const",
        type=float,
        default=1.49618,
        help="Social acceleration coefficient for constant variants",
    )

    parser.add_argument(
        "--v-max-ratio",
        type=float,
        default=0.2,
        help="Maximum velocity ratio (v_max = ratio * Range)",
    )

    parser.add_argument(
        "--neighborhood-size",
        type=int,
        default=3,
        help="Neighborhood size for ring topology (must be odd, >= 3)",
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

    if args.population_size < 3:
        errors.append("--population-size must be >= 3")

    if args.max_fes < 1:
        errors.append("--max-fes must be >= 1")

    if args.w_start < 0 or args.w_end < 0 or args.w_const < 0:
        errors.append("Inertia weights (--w-start, --w-end, --w-const) must be >= 0")

    if (
        args.c1_linear < 0
        or args.c2_linear < 0
        or args.c1_const < 0
        or args.c2_const < 0
    ):
        errors.append("Acceleration coefficients (c1, c2) must be >= 0")

    if args.v_max_ratio <= 0:
        errors.append("--v-max-ratio must be > 0")

    if args.neighborhood_size < 3 or args.neighborhood_size % 2 == 0:
        errors.append("--neighborhood-size must be an odd integer >= 3")

    if errors:
        parser.error("\n  ".join(["Invalid arguments:"] + errors))

    # Post-processing
    if "all" in args.functions:
        args.functions = list(ALL_FUNCTIONS)

    if "all" in args.variants:
        args.variants = list(ALL_VARIANTS)

    return args
