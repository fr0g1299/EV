"""
Genetic Algorithm Engine for Real-Valued Optimization

Implements 5 GA variants for minimization of benchmark functions:
  1. Binary GA with IEEE 754 encoding (32 bits/dim)
  2. Binary GA with Fixed-Point encoding (32 bits/dim)
  3. Binary GA with BCD encoding (32 bits/dim)
  4. Real-valued GA with Gaussian mutation
  5. Real-valued GA with Uniform random mutation

Common framework:
  - Tournament selection (minimization)
  - Elitism (configurable ratio, default 10%)
  - Convergence tracing (best-so-far at each FES)
"""

from typing import Callable, Tuple, List
import numpy as np

# Type aliases
ObjectiveFunc = Callable[[np.ndarray], float]
Bounds = Tuple[float, float]

# Constants
BITS_PER_DIM = 32
BCD_MAX = 99_999_999  # 8 BCD digits, each 0-9
FIXED_POINT_MAX = 2**32 - 1  # 4294967295
_BCD_POWERS = 10 ** np.arange(7, -1, -1, dtype=np.int64)
VARIANT_NAMES = {
    "ieee754": "IEEE 754 Binary",
    "fixed_point": "Fixed-Point Binary",
    "bcd": "BCD Binary",
    "real_gaussian": "Real-Valued (Gaussian Mutation)",
    "real_uniform": "Real-Valued (Uniform Mutation)",
}
_BINARY_VARIANTS = {"ieee754", "fixed_point", "bcd"}


# Encoding / Decoding helpers


def encode_ieee754(values: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Encode real values to IEEE 754 32-bit binary representation.

    Each value is clamped to bounds, cast to `np.float32`, and then its
    32-bit memory representation is unpacked into an array of 0/1 bits
    (MSB first).  The per-dimension bit arrays are concatenated.

    Parameters
    ----------
    values : np.ndarray
        Real-valued solution vector of shape `(dim,)`.
    bounds : Bounds
        `(lo, hi)` search domain shared by every dimension.

    Returns
    -------
    np.ndarray
        Binary chromosome of shape `(dim * 32,)` with dtype `np.uint8`.
    """
    lo, hi = bounds
    clamped = np.clip(values, lo, hi).astype(np.float32)

    uint_view = clamped.view(np.uint32)
    bits_list = []
    for u in uint_view:
        # MSB first
        bits = np.array([(u >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
        bits_list.append(bits)

    return np.concatenate(bits_list)


def decode_ieee754(bits: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Decode IEEE 754 binary back to real values.

    Reconstructs `np.float32` values from 32-bit groups, clamps to bounds,
    and replaces any NaN / Inf with midpoint of bounds.

    Parameters
    ----------
    bits : np.ndarray
        Binary chromosome of shape `(dim * 32,)`.
    bounds : Bounds
        `(lo, hi)` search domain.

    Returns
    -------
    np.ndarray
        Real-valued solution vector of shape `(dim,)`.
    """
    lo, hi = bounds
    bytes_arr = np.packbits(bits)
    f32 = bytes_arr.view(np.dtype(">f4"))
    midpoint = (lo + hi) / 2.0
    safe_f = np.where(np.isnan(f32) | np.isinf(f32), midpoint, f32)

    return np.clip(safe_f.astype(np.float64), lo, hi)


def encode_fixed_point(values: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Encode real values to fixed-point 32-bit binary.

    The mapping is linear: `integer = round((value - lo) / (hi - lo) * (2^32 - 1))`.
    Each integer is then unpacked to 32 bits (MSB first).

    Parameters
    ----------
    values : np.ndarray
        Real-valued solution vector of shape `(dim,)`.
    bounds : Bounds
        `(lo, hi)` search domain.

    Returns
    -------
    np.ndarray
        Binary chromosome of shape `(dim * 32,)` with dtype `np.uint8`.
    """
    lo, hi = bounds
    clamped = np.clip(values, lo, hi)

    # Normalize to [0, 1], then scale to [0, 2^32-1]
    normalised = (clamped - lo) / (hi - lo)
    integers = np.round(normalised * FIXED_POINT_MAX).astype(np.uint64)
    integers = np.clip(integers, 0, FIXED_POINT_MAX)
    bits_list = []
    for u in integers:
        bits = np.array([(int(u) >> (31 - i)) & 1 for i in range(32)], dtype=np.uint8)
        bits_list.append(bits)

    return np.concatenate(bits_list)


def decode_fixed_point(bits: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Decode fixed-point binary back to real values.

    Reverses the linear mapping: `value = lo + (integer / (2^32 - 1)) * (hi - lo)`.

    Parameters
    ----------
    bits : np.ndarray
        Binary chromosome of shape `(dim * 32,)`.
    bounds : Bounds
        `(lo, hi)` search domain.

    Returns
    -------
    np.ndarray
        Real-valued solution vector of shape `(dim,)`.
    """
    lo, hi = bounds
    bytes_arr = np.packbits(bits)
    u32 = bytes_arr.view(np.dtype(">u4"))

    return lo + (u32 / FIXED_POINT_MAX) * (hi - lo)


def encode_bcd(values: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Encode real values to BCD 32-bit binary.

    Each dimension is mapped linearly to an 8-digit decimal number
    (0 – 99 999 999) which is then encoded as 8 BCD nibbles (32 bits total,
    most-significant digit first).

    Parameters
    ----------
    values : np.ndarray
        Real-valued solution vector of shape `(dim,)`.
    bounds : Bounds
        `(lo, hi)` search domain.

    Returns
    -------
    np.ndarray
        Binary chromosome of shape `(dim * 32,)` with dtype `np.uint8`.
    """
    lo, hi = bounds
    clamped = np.clip(values, lo, hi)
    normalised = (clamped - lo) / (hi - lo)
    decimals = np.round(normalised * BCD_MAX).astype(np.int64)
    decimals = np.clip(decimals, 0, BCD_MAX)
    bits_list = []
    for dec in decimals:
        # Extract 8 BCD digits (most-significant first)
        digits = []
        val = int(dec)
        for _ in range(8):
            digits.append(val % 10)
            val //= 10
        digits.reverse()  # MSB first

        # Encode each digit as 4 bits
        nibble_bits = []
        for digit in digits:
            for b in range(3, -1, -1):
                nibble_bits.append((digit >> b) & 1)
        bits_list.append(np.array(nibble_bits, dtype=np.uint8))

    return np.concatenate(bits_list)


def decode_bcd(bits: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Decode BCD binary back to real values.  Repairs invalid BCD digits first.

    Parameters
    ----------
    bits : np.ndarray
        Binary chromosome of shape `(dim * 32,)`.  May contain invalid BCD
        nibbles after crossover / mutation.
    bounds : Bounds
        `(lo, hi)` search domain.

    Returns
    -------
    np.ndarray
        Real-valued solution vector of shape `(dim,)`.
    """
    lo, hi = bounds
    b = repair_bcd(bits.copy())
    nibbles = b.reshape(-1, 8, 4)
    digits = (
        nibbles[:, :, 0] * 8
        + nibbles[:, :, 1] * 4
        + nibbles[:, :, 2] * 2
        + nibbles[:, :, 3]
    )
    dec_vals = np.minimum(digits @ _BCD_POWERS, BCD_MAX)

    return lo + (dec_vals / BCD_MAX) * (hi - lo)


def repair_bcd(bits: np.ndarray) -> np.ndarray:
    """Repair invalid BCD digits (values 10–15) by clamping to 9.

    Each group of 4 consecutive bits is treated as one BCD nibble.
    If the nibble value exceeds 9, it is replaced with the bits for 9 (`1001`).

    Parameters
    ----------
    bits : np.ndarray
        Binary chromosome (length must be a multiple of 4).

    Returns
    -------
    np.ndarray
        Repaired chromosome (same shape, dtype `np.uint8`).
    """
    nibbles = bits.reshape(-1, 4)
    digits = nibbles[:, 0] * 8 + nibbles[:, 1] * 4 + nibbles[:, 2] * 2 + nibbles[:, 3]
    invalid = digits > 9
    if np.any(invalid):
        nibbles[invalid] = [1, 0, 0, 1]

    return bits


# Binary GA operators


def binary_single_point_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-point crossover on binary strings.

    A crossover point `k` is chosen uniformly in `[1, L-1]` where `L` is
    the chromosome length.  Child 1 gets `parent_a[:k] + parent_b[k:]` and
    child 2 gets the complement.

    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Binary chromosomes of equal length.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two offspring chromosomes.
    """
    length = len(parent_a)
    point = rng.integers(1, length)  # [1, length-1]
    child_a = np.concatenate([parent_a[:point], parent_b[point:]])
    child_b = np.concatenate([parent_b[:point], parent_a[point:]])

    return child_a, child_b


def binary_bit_flip_mutation(
    chromosome: np.ndarray,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bit-flip mutation on a binary chromosome.

    Each bit is flipped independently with the given probability.

    Parameters
    ----------
    chromosome : np.ndarray
        Binary chromosome (0/1 values, dtype `np.uint8`).
    probability : float
        Per-bit flip probability.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Mutated chromosome (copy).
    """
    mutant = chromosome.copy()
    mask = rng.random(len(mutant)) < probability
    mutant[mask] = 1 - mutant[mask]

    return mutant


# Real-valued GA operators


def real_single_point_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single-point crossover between dimensions for real-valued individuals.

    A crossover point `k` is chosen uniformly in `[1, D-1]`.
    Child 1 = `parent_a[:k]` ∥ `parent_b[k:]`.
    Child 2 = `parent_b[:k]` ∥ `parent_a[k:]`.

    Parameters
    ----------
    parent_a, parent_b : np.ndarray
        Real-valued vectors of shape `(dim,)`.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two offspring vectors.
    """
    dim = len(parent_a)
    if dim < 2:
        return parent_a.copy(), parent_b.copy()
    k = rng.integers(1, dim)  # [1, dim-1]
    child_a = np.concatenate([parent_a[:k], parent_b[k:]])
    child_b = np.concatenate([parent_b[:k], parent_a[k:]])

    return child_a, child_b


def gaussian_mutation(
    individual: np.ndarray,
    bounds: Bounds,
    sigma: float,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian mutation: `x' = x + N(0, sigma²)`, clamped to bounds.

    Each dimension is mutated independently with the given probability.

    Parameters
    ----------
    individual : np.ndarray
        Real-valued vector of shape `(dim,)`.
    bounds : Bounds
        `(lo, hi)` search domain.
    sigma : float
        Standard deviation of the Gaussian perturbation.
    probability : float
        Per-dimension mutation probability.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Mutated vector (copy), clamped to bounds.
    """
    lo, hi = bounds
    mutant = individual.copy()
    mask = rng.random(len(mutant)) < probability
    perturbation = rng.normal(0.0, sigma, size=int(mask.sum()))
    mutant[mask] += perturbation

    return np.clip(mutant, lo, hi)


def uniform_mutation(
    individual: np.ndarray,
    bounds: Bounds,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniform random mutation: replace with a random value from `[lo, hi]`.

    Each dimension is independently replaced with the given probability.

    Parameters
    ----------
    individual : np.ndarray
        Real-valued vector of shape `(dim,)`.
    bounds : Bounds
        `(lo, hi)` search domain.
    probability : float
        Per-dimension mutation probability.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    np.ndarray
        Mutated vector (copy).
    """
    lo, hi = bounds
    mutant = individual.copy()
    mask = rng.random(len(mutant)) < probability
    mutant[mask] = rng.uniform(lo, hi, size=int(mask.sum()))

    return mutant


# Selection


def tournament_selection(
    fitness: np.ndarray,
    tournament_size: int,
    rng: np.random.Generator,
) -> int:
    """Tournament selection (minimization).

    Pick tournament_size contestants uniformly at random and return the
    index of the one with the lowest fitness value.

    Parameters
    ----------
    fitness : np.ndarray
        Fitness values of the population, shape `(pop_size,)`.
    tournament_size : int
        Number of contestants.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    int
        Index of the tournament winner.
    """
    candidates = rng.choice(len(fitness), size=tournament_size, replace=False)
    winner = candidates[np.argmin(fitness[candidates])]
    return int(winner)


# Main GA runner


def _init_binary_population(
    variant: str,
    pop_size: int,
    dim: int,
    rng: np.random.Generator,
) -> list:
    """Initialise a population of random binary chromosomes.

    Parameters
    ----------
    variant : str
        GA variant name (used to apply BCD repair if needed).
    pop_size : int
        Number of individuals.
    dim : int
        Dimensionality.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    list
        List of binary chromosomes, each of shape `(dim * 32,)`, dtype `np.uint8`.
    """
    chrom_len = BITS_PER_DIM * dim
    population: list = []
    for _ in range(pop_size):
        bits = rng.integers(0, 2, size=chrom_len, dtype=np.uint8)
        # For BCD, repair immediately
        if variant == "bcd":
            bits = repair_bcd(bits)
        population.append(bits)

    return population


def _init_real_population(
    pop_size: int,
    dim: int,
    bounds: Bounds,
    rng: np.random.Generator,
) -> list:
    """Initialise a population of real-valued vectors uniformly in bounds.

    Parameters
    ----------
    pop_size : int
        Number of individuals.
    dim : int
        Dimensionality.
    bounds : Bounds
        `(lo, hi)` search domain.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    list
        List of real-valued vectors, each of shape `(dim,)`, dtype `np.float64`.
    """
    lo, hi = bounds

    return [rng.uniform(lo, hi, size=dim) for _ in range(pop_size)]


def _decode(variant: str, chromosome: np.ndarray, bounds: Bounds) -> np.ndarray:
    """Decode a single chromosome according to the variant."""
    if variant == "ieee754":
        return decode_ieee754(chromosome, bounds)
    elif variant == "fixed_point":
        return decode_fixed_point(chromosome, bounds)
    elif variant == "bcd":
        return decode_bcd(chromosome, bounds)
    else:
        # real-valued: chromosome is already the real vector
        return chromosome


def run_ga(
    variant: str,
    objective: ObjectiveFunc,
    bounds: Bounds,
    dim: int,
    max_fes: int,
    rng: np.random.Generator,
    population_size: int = 100,
    elitism_ratio: float = 0.10,
    crossover_ratio: float = 0.90,
    mutation_probability: float = 0.05,
    tournament_size: int = 3,
    sigma: float = None,
) -> Tuple[float, np.ndarray, List[float]]:
    """Run one complete GA experiment.

    Parameters
    ----------
    variant : str
        One of `"ieee754"`, `"fixed_point"`, `"bcd"`,
        `"real_gaussian"`, `"real_uniform"`.
    objective : ObjectiveFunc
        Objective (fitness) function to minimize.  Accepts a 1-D
        `np.ndarray` of shape `(dim,)` and returns a scalar float.
    bounds : Bounds
        `(lo, hi)` common to every dimension.
    dim : int
        Number of decision variables.
    max_fes : int
        Maximum number of function evaluations (stopping criterion).
    rng : np.random.Generator
        Seeded random generator for reproducibility.
    population_size : int
        Number of individuals in the population.
    elitism_ratio : float
        Fraction of the best individuals preserved each generation.
    crossover_ratio : float
        Probability that a selected pair undergoes crossover (otherwise both
        parents are copied verbatim).
    mutation_probability : float
        Per-bit (binary) or per-dimension (real) mutation probability.
    tournament_size : int
        Tournament size for selection.
    sigma : float or None
        Standard deviation for Gaussian mutation.  If `None`, defaults to
        `0.1 * (hi - lo)`.

    Returns
    -------
    best_fitness : float
        Best (lowest) fitness value found.
    best_solution : np.ndarray
        Corresponding solution vector of shape `(dim,)`.
    trace : List[float]
        Best-so-far fitness recorded at each function evaluation.
    """
    if variant not in VARIANT_NAMES:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from {list(VARIANT_NAMES.keys())}"
        )

    lo, hi = bounds
    is_binary = variant in _BINARY_VARIANTS

    # Default sigma for Gaussian mutation
    if sigma is None:
        sigma = 0.1 * (hi - lo)

    elitism_count = max(1, int(round(elitism_ratio * population_size)))

    # Initialise population
    if is_binary:
        population = _init_binary_population(variant, population_size, dim, rng)
    else:
        population = _init_real_population(population_size, dim, bounds, rng)

    # Evaluate initial population
    fitness = np.empty(population_size, dtype=np.float64)
    trace: List[float] = []
    fes = 0
    best_fitness = np.inf
    best_solution = None

    for i in range(population_size):
        if fes >= max_fes:
            break
        decoded = _decode(variant, population[i], bounds)
        f = objective(decoded)
        fitness[i] = f
        fes += 1
        if f < best_fitness:
            best_fitness = f
            best_solution = decoded.copy()
        trace.append(best_fitness)

    if fes >= max_fes:
        return best_fitness, best_solution, trace

    # Generational loop
    while fes < max_fes:
        # Sort by fitness (ascending – best first)
        order = np.argsort(fitness)
        sorted_pop = [population[i] for i in order]
        sorted_fit = fitness[order].copy()

        # Elitism: carry over the best individuals
        new_population = [ind.copy() for ind in sorted_pop[:elitism_count]]
        new_fitness_list = list(sorted_fit[:elitism_count])

        # Fill rest via selection, crossover, mutation
        while len(new_population) < population_size and fes < max_fes:
            # Select two parents
            idx_a = tournament_selection(fitness, tournament_size, rng)
            idx_b = tournament_selection(fitness, tournament_size, rng)
            parent_a = population[idx_a].copy()
            parent_b = population[idx_b].copy()

            # Crossover
            if rng.random() < crossover_ratio:
                if is_binary:
                    child_a, child_b = binary_single_point_crossover(
                        parent_a, parent_b, rng
                    )
                else:
                    child_a, child_b = real_single_point_crossover(
                        parent_a, parent_b, rng
                    )
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()

            # Mutation
            if is_binary:
                child_a = binary_bit_flip_mutation(child_a, mutation_probability, rng)
                child_b = binary_bit_flip_mutation(child_b, mutation_probability, rng)
                # Repair BCD after mutation
                if variant == "bcd":
                    child_a = repair_bcd(child_a)
                    child_b = repair_bcd(child_b)
            elif variant == "real_gaussian":
                child_a = gaussian_mutation(
                    child_a, bounds, sigma, mutation_probability, rng
                )
                child_b = gaussian_mutation(
                    child_b, bounds, sigma, mutation_probability, rng
                )
            else:  # real_uniform
                child_a = uniform_mutation(child_a, bounds, mutation_probability, rng)
                child_b = uniform_mutation(child_b, bounds, mutation_probability, rng)

            # Evaluate children and add to new population
            for child in (child_a, child_b):
                if len(new_population) >= population_size or fes >= max_fes:
                    break
                decoded = _decode(variant, child, bounds)
                f = objective(decoded)
                fes += 1
                if f < best_fitness:
                    best_fitness = f
                    best_solution = decoded.copy()
                trace.append(best_fitness)
                new_population.append(child)
                new_fitness_list.append(f)

        # Replace population
        population = new_population
        fitness = np.array(new_fitness_list, dtype=np.float64)

    return best_fitness, best_solution, trace
