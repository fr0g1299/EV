"""
Differential Evolution (DE) Engine

Implements three DE variants for continuous parameter optimization:
  1. DE/rand/1/bin: Classic Differential Evolution with random base vector
  2. DE/best/1/bin: Differential Evolution guided by the best population vector
  3. jDE: Self-Adaptive Differential Evolution

Common framework:
  - Binomial crossover (with guaranteed mutated coordinate)
  - Box-constraint boundary handling (clamping to search bounds)
  - One-to-one greedy selection (minimization)
  - Detailed convergence tracing (best-so-far recorded at each FES)
"""

from typing import Callable, Tuple, List
import numpy as np

# Type aliases
ObjectiveFunc = Callable[[np.ndarray], float]
Bounds = Tuple[float, float]

VARIANT_NAMES = {
    "rand_1": "DE/rand/1/bin",
    "best_1": "DE/best/1/bin",
    "jde": "jDE (Self-Adaptive)",
}


def init_population(
    population_size: int,
    dim: int,
    bounds: Bounds,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize a population uniformly distributed within search bounds.

    Parameters
    ----------
    population_size : int
        Number of individuals (chromosomes/vectors).
    dim : int
        Dimensionality of the problem.
    bounds : Bounds
        `(lower, upper)` search bounds.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Array of shape `(population_size, dim)` with float64 coordinates.
    """
    lo, hi = bounds
    return rng.uniform(lo, hi, size=(population_size, dim))


def mutate_rand1(
    pop: np.ndarray,
    target_idx: int,
    F: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a mutant vector using the Rand/1 strategy.

    v_i = x_{r1} + F * (x_{r2} - x_{r3})
    where r1, r2, r3 are mutually distinct and distinct from target_idx.
    """
    pop_size = len(pop)
    candidates = [i for i in range(pop_size) if i != target_idx]
    r1, r2, r3 = rng.choice(candidates, size=3, replace=False)
    return pop[r1] + F * (pop[r2] - pop[r3])


def mutate_best1(
    pop: np.ndarray,
    target_idx: int,
    best_idx: int,
    F: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a mutant vector using the Best/1 strategy.

    v_i = x_{best} + F * (x_{r1} - x_{r2})
    where r1, r2 are distinct from each other, distinct from target_idx,
    and distinct from best_idx if possible.
    """
    pop_size = len(pop)
    candidates = [i for i in range(pop_size) if i != target_idx and i != best_idx]
    if len(candidates) >= 2:
        r1, r2 = rng.choice(candidates, size=2, replace=False)
    else:
        # Fallback if population is very small
        candidates = [i for i in range(pop_size) if i != target_idx]
        r1, r2 = rng.choice(candidates, size=2, replace=False)

    return pop[best_idx] + F * (pop[r1] - pop[r2])


def binomial_crossover(
    target: np.ndarray,
    mutant: np.ndarray,
    CR: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perform binomial crossover between target and mutant vectors.

    Ensures at least one component is taken from the mutant vector (j_rand).
    """
    dim = len(target)
    j_rand = rng.integers(0, dim)
    mask = rng.random(size=dim) <= CR
    mask[j_rand] = True

    return np.where(mask, mutant, target)


def handle_bounds(vector: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Clamp coordinates to domain search bounds."""
    lo, hi = bounds
    return np.clip(vector, lo, hi)


def run_de(
    variant: str,
    objective: ObjectiveFunc,
    bounds: Bounds,
    dim: int,
    max_fes: int,
    rng: np.random.Generator,
    population_size: int = 100,
    F: float = 0.5,
    CR: float = 0.8,
    tau1: float = 0.1,
    tau2: float = 0.1,
    f_lower: float = 0.1,
    f_upper: float = 0.9,
    cr_lower: float = 0.0,
    cr_upper: float = 1.0,
) -> Tuple[float, np.ndarray, List[float]]:
    """Execute a single run of Differential Evolution.

    Parameters
    ----------
    variant : str
        Algorithm variant: 'rand_1', 'best_1', or 'jde'.
    objective : ObjectiveFunc
        Function to minimize.
    bounds : Bounds
        Domain bounds (min, max).
    dim : int
        Search space dimensionality.
    max_fes : int
        Maximum number of function evaluations.
    rng : np.random.Generator
        Random number generator instance.
    population_size : int, default=100
        Size of the population.
    F : float, default=0.5
        Mutation scale factor for rand_1 and best_1, initial F for jDE.
    CR : float, default=0.8
        Crossover probability for rand_1 and best_1 (for jDE, default initial CR is 0.9).
    tau1 : float, default=0.1
        Probability of F adaptation in jDE.
    tau2 : float, default=0.1
        Probability of CR adaptation in jDE.
    f_lower : float, default=0.1
        Lower bound of F in jDE self-adaptation.
    f_upper : float, default=0.9
        Upper bound of F in jDE self-adaptation.
    cr_lower : float, default=0.0
        Lower bound of CR in jDE self-adaptation.
    cr_upper : float, default=1.0
        Upper bound of CR in jDE self-adaptation.

    Returns
    -------
    best_fitness : float
        Best objective function value found.
    best_solution : np.ndarray
        Best parameter vector found.
    trace : List[float]
        Best-so-far fitness after each function evaluation.
    """
    if variant not in VARIANT_NAMES:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from {list(VARIANT_NAMES.keys())}"
        )

    # Initialize population
    population = init_population(population_size, dim, bounds, rng)

    # Initialize individual parameters for jDE
    if variant == "jde":
        # Task recommendation: initial CR: 0.9, initial F: 0.5
        f_pop = np.full(population_size, F, dtype=np.float64)
        cr_pop = np.full(population_size, CR, dtype=np.float64)

    fitness = np.empty(population_size, dtype=np.float64)
    trace: List[float] = []
    fes = 0
    best_fitness = np.inf
    best_solution = None

    # Evaluate initial population
    for i in range(population_size):
        if fes >= max_fes:
            break
        f_val = objective(population[i])
        fitness[i] = f_val
        fes += 1
        if f_val < best_fitness:
            best_fitness = f_val
            best_solution = population[i].copy()
        trace.append(best_fitness)

    if fes >= max_fes:
        return best_fitness, best_solution, trace

    # Generational evolution loop
    while fes < max_fes:
        best_idx = int(np.argmin(fitness))

        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)
        if variant == "jde":
            new_f_pop = np.empty_like(f_pop)
            new_cr_pop = np.empty_like(cr_pop)

        for i in range(population_size):
            if fes >= max_fes:
                # Copy remaining individuals unchanged
                new_population[i:] = population[i:]
                new_fitness[i:] = fitness[i:]
                if variant == "jde":
                    new_f_pop[i:] = f_pop[i:]
                    new_cr_pop[i:] = cr_pop[i:]
                break

            # Determine F and CR
            if variant == "jde":
                if rng.random() < tau1:
                    cur_f = f_lower + rng.random() * (f_upper - f_lower)
                else:
                    cur_f = f_pop[i]

                if rng.random() < tau2:
                    cur_cr = cr_lower + rng.random() * (cr_upper - cr_lower)
                else:
                    cur_cr = cr_pop[i]

                mutant = mutate_rand1(population, i, cur_f, rng)
            elif variant == "rand_1":
                cur_f = F
                cur_cr = CR
                mutant = mutate_rand1(population, i, cur_f, rng)
            else:  # best_1
                cur_f = F
                cur_cr = CR
                mutant = mutate_best1(population, i, best_idx, cur_f, rng)

            # Binomial crossover
            trial = binomial_crossover(population[i], mutant, cur_cr, rng)

            # Handle bounds
            trial = handle_bounds(trial, bounds)

            # Evaluate trial vector
            trial_fitness = objective(trial)
            fes += 1
            if trial_fitness < best_fitness:
                best_fitness = trial_fitness
                best_solution = trial.copy()
            trace.append(best_fitness)

            # Greedy selection (minimization)
            if trial_fitness <= fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                if variant == "jde":
                    new_f_pop[i] = cur_f
                    new_cr_pop[i] = cur_cr
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
                if variant == "jde":
                    new_f_pop[i] = f_pop[i]
                    new_cr_pop[i] = cr_pop[i]

        population = new_population
        fitness = new_fitness
        if variant == "jde":
            f_pop = new_f_pop
            cr_pop = new_cr_pop

    return best_fitness, best_solution, trace
