"""
Benchmark Functions Module

Contains implementations of six benchmark optimization functions.
"""

from typing import Tuple
import numpy as np


class BenchmarkFunction:
    """Base class for benchmark functions"""

    def __init__(
        self,
        name: str,
        bounds: Tuple[float, float],
        global_min: float,
    ):
        self.name = name
        self.bounds = bounds
        self.global_min = global_min

    def __call__(self, x: np.ndarray) -> float:
        raise NotImplementedError


class Sphere(BenchmarkFunction):
    """## Sphere function
    Global minimum: f(0, ..., 0) = 0
    Domain: [-100, 100]^D (often [-5.12, 5.12] is also used)
    """

    def __init__(self):
        super().__init__("Sphere", (-5.12, 5.12), 0.0)

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(x**2)


class Rastrigin(BenchmarkFunction):
    """## Rastrigin function
    Global minimum: f(0, ..., 0) = 0
    Domain: [-5.12, 5.12]^D
    """

    def __init__(self):
        super().__init__("Rastrigin", (-5.12, 5.12), 0.0)

    def __call__(self, x: np.ndarray) -> float:
        D = len(x)
        return 10 * D + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))


class Ackley(BenchmarkFunction):
    """## Ackley function
    Global minimum: f(0, ..., 0) = 0
    Domain: [-32.768, 32.768]^D (often [-5, 5] is also used)
    """

    def __init__(self):
        super().__init__("Ackley", (-32.768, 32.768), 0.0)

    def __call__(self, x: np.ndarray) -> float:
        D = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return (
            -20 * np.exp(-0.2 * np.sqrt(sum_sq / D)) - np.exp(sum_cos / D) + 20 + np.e
        )


class Zakharov(BenchmarkFunction):
    """## Zakharov function
    Global minimum: f(0, ..., 0) = 0
    Domain: [-5, 10]^D (often [-10, 10] is also used)
    """

    def __init__(self):
        super().__init__("Zakharov", (-10, 10), 0.0)

    def __call__(self, x: np.ndarray) -> float:
        D = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(0.5 * np.arange(1, D + 1) * x)
        return sum1 + sum2**2 + sum2**4


class Michalewicz(BenchmarkFunction):
    """## Michalewicz function
    Global minimum:
            \tD=2: f(2.20, 1.57) ≈ -1.8013
            \tD=5: ≈ -4.687658
            \tD=10: ≈ -9.66015

    Domain: [0, π]^D
    """

    def __init__(self, m: int = 10):
        super().__init__("Michalewicz", (0, np.pi), -9.66015)
        self.m = m

    def __call__(self, x: np.ndarray) -> float:
        D = len(x)
        i = np.arange(1, D + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * self.m))


class Rosenbrock(BenchmarkFunction):
    """## Rosenbrock function
    Global minimum: f(1, ..., 1) = 0
    Domain: [-5, 10]^D (often [-2.048, 2.048] is also used)
    """

    def __init__(self):
        super().__init__("Rosenbrock", (-2.048, 2.048), 0.0)

    def __call__(self, x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


class Schwefel(BenchmarkFunction):
    """## Schwefel function
    Global minimum: f(420.9687, ..., 420.9687) = 0
    Domain: [-500, 500]^D
    """

    def __init__(self):
        super().__init__("Schwefel", (-500.0, 500.0), 0.0)

    def __call__(self, x: np.ndarray) -> float:
        D = len(x)
        return 418.9829 * D - np.sum(x * np.sin(np.sqrt(np.abs(x))))
