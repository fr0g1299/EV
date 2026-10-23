"""
Modules Package

This package contains the GA engine, benchmark functions,
visualization utilities, and CLI tools for real-valued optimization.
"""

from modules.benchmark_functions import (
    BenchmarkFunction,
    Sphere,
    Rastrigin,
    Ackley,
    Zakharov,
    Michalewicz,
    Rosenbrock,
    Schwefel,
)
from modules.ga_engine import run_ga, VARIANT_NAMES
from modules.visualization import plot_convergence
from modules.cli import parse_arguments, ALL_FUNCTIONS, ALL_VARIANTS

# Map CLI function names to benchmark function instances
FUNCTION_MAP = {
    "sphere": Sphere(),
    "rastrigin": Rastrigin(),
    "ackley": Ackley(),
    "zakharov": Zakharov(),
    "michalewicz": Michalewicz(),
    "rosenbrock": Rosenbrock(),
    "schwefel": Schwefel(),
}


__all__ = [
    # Benchmark Functions
    "BenchmarkFunction",
    # GA Engine
    "run_ga",
    "VARIANT_NAMES",
    # Visualization
    "plot_convergence",
    # CLI
    "parse_arguments",
    "ALL_FUNCTIONS",
    "ALL_VARIANTS",
    # Mapping
    "FUNCTION_MAP",
]
