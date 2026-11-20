"""
Particle Swarm Optimization (PSO) Engine

Implements three PSO variants for continuous parameter optimization:
  1. Linear w (Star): Linearly decreasing inertia weight (w = 0.8 -> 0.3),
     c1 = c2 = 2.0, global (star) topology.
  2. Const w (Star): Constant inertia weight (w = 0.7),
     c1 = c2 = 1.49618, global (star) topology.
  3. Const w (Ring): Constant inertia weight (w = 0.7),
     c1 = c2 = 1.49618, ring topology (neighborhood size of 3).

Common specifications:
  - Maximum velocity: v_max = 0.2 * Range.
  - Particles are allowed to fly outside search boundaries, but their
    fitness is NOT evaluated outside the search bounds.
  - Evaluation budget (MAXFES) used as the stopping criterion.
  - Detailed convergence tracing (best-so-far recorded at each FES).
"""

from typing import Callable, Tuple, List
import numpy as np

# Type aliases
ObjectiveFunc = Callable[[np.ndarray], float]
Bounds = Tuple[float, float]

VARIANT_NAMES = {
    "linear_star": "PSO (Linear w, Star)",
    "const_star": "PSO (Const w, Star)",
    "const_ring": "PSO (Const w, Ring, k=3)",
}


def init_swarm(
    population_size: int,
    dim: int,
    bounds: Bounds,
    v_max: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialize particle positions uniformly inside bounds and velocities in [-v_max, v_max].

    Parameters
    ----------
    population_size : int
        Number of particles in the swarm (NP).
    dim : int
        Dimensionality of the search space.
    bounds : Bounds
        Domain bounds (lower, upper).
    v_max : float
        Maximum allowable velocity per dimension.
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    positions : np.ndarray
        Array of shape (population_size, dim) with positions in [lo, hi].
    velocities : np.ndarray
        Array of shape (population_size, dim) with velocities in [-v_max, v_max].
    """
    lo, hi = bounds
    positions = rng.uniform(lo, hi, size=(population_size, dim))
    velocities = rng.uniform(-v_max, v_max, size=(population_size, dim))
    return positions, velocities


def get_ring_best(
    pbest_pos: np.ndarray,
    pbest_fit: np.ndarray,
    neighborhood_size: int = 3,
) -> np.ndarray:
    """Compute local best attractor positions in a ring topology.

    For neighborhood size 3, particle i considers neighbors (i-1) % NP, i, and (i+1) % NP.

    Parameters
    ----------
    pbest_pos : np.ndarray
        Personal best positions of all particles, shape (population_size, dim).
    pbest_fit : np.ndarray
        Personal best fitnesses of all particles, shape (population_size,).
    neighborhood_size : int, default=3
        Size of the neighborhood (must be odd, e.g., 3).

    Returns
    -------
    np.ndarray
        Local best attractor positions for each particle, shape (population_size, dim).
    """
    pop_size = len(pbest_pos)
    radius = neighborhood_size // 2
    offsets = np.arange(-radius, radius + 1)
    # Shape: (pop_size, neighborhood_size)
    neighbor_indices = (np.arange(pop_size)[:, None] + offsets[None, :]) % pop_size
    neighbor_fits = pbest_fit[neighbor_indices]
    best_slot = np.argmin(neighbor_fits, axis=1)
    best_indices = neighbor_indices[np.arange(pop_size), best_slot]
    return pbest_pos[best_indices]


def run_pso(
    variant: str,
    objective: ObjectiveFunc,
    bounds: Bounds,
    dim: int,
    max_fes: int,
    rng: np.random.Generator,
    population_size: int = 50,
    w_start: float = 0.8,
    w_end: float = 0.3,
    w_const: float = 0.7,
    c1_linear: float = 2.0,
    c2_linear: float = 2.0,
    c1_const: float = 1.49618,
    c2_const: float = 1.49618,
    v_max_ratio: float = 0.2,
    neighborhood_size: int = 3,
) -> Tuple[float, np.ndarray, List[float]]:
    """Execute a single run of Particle Swarm Optimization (PSO).

    Parameters
    ----------
    variant : str
        Algorithm variant: 'linear_star', 'const_star', or 'const_ring'.
    objective : ObjectiveFunc
        Objective function to minimize.
    bounds : Bounds
        Search domain bounds (lower, upper).
    dim : int
        Search space dimensionality.
    max_fes : int
        Maximum number of function evaluations (stopping criterion).
    rng : np.random.Generator
        Random number generator.
    population_size : int, default=50
        Swarm size (NP, reasonable range 20-100).
    w_start : float, default=0.8
        Starting inertia weight for linearly decreasing variant.
    w_end : float, default=0.3
        Ending inertia weight for linearly decreasing variant.
    w_const : float, default=0.7
        Inertia weight for constant inertia variants.
    c1_linear : float, default=2.0
        Cognitive acceleration coefficient for linear variant.
    c2_linear : float, default=2.0
        Social acceleration coefficient for linear variant.
    c1_const : float, default=1.49618
        Cognitive acceleration coefficient for constant variants.
    c2_const : float, default=1.49618
        Social acceleration coefficient for constant variants.
    v_max_ratio : float, default=0.2
        Ratio of search range defining v_max = v_max_ratio * (hi - lo).
    neighborhood_size : int, default=3
        Neighborhood size for ring topology.

    Returns
    -------
    best_fitness : float
        Best objective function value found.
    best_solution : np.ndarray
        Best parameter vector found.
    trace : List[float]
        Best-so-far fitness after each objective evaluation.
    """
    if variant not in VARIANT_NAMES:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from {list(VARIANT_NAMES.keys())}"
        )

    lo, hi = bounds
    range_span = float(hi - lo)
    v_max = v_max_ratio * range_span
    is_linear_w = False
    topology = None
    c1 = c2 = w = 0.0

    # Determine algorithm configuration
    if variant == "linear_star":
        is_linear_w = True
        c1 = c1_linear
        c2 = c2_linear
        topology = "star"
    elif variant == "const_star":
        is_linear_w = False
        w = w_const
        c1 = c1_const
        c2 = c2_const
        topology = "star"
    elif variant == "const_ring":
        is_linear_w = False
        w = w_const
        c1 = c1_const
        c2 = c2_const
        topology = "ring"

    # Initialize swarm positions and velocities
    positions, velocities = init_swarm(population_size, dim, bounds, v_max, rng)

    pbest_pos = positions.copy()
    pbest_fit = np.full(population_size, np.inf, dtype=np.float64)

    trace: List[float] = []
    fes = 0
    best_fitness = np.inf
    best_solution = None

    # Evaluate initial swarm
    for i in range(population_size):
        if fes >= max_fes:
            break
        val = objective(positions[i])
        fes += 1
        pbest_fit[i] = val
        if val < best_fitness:
            best_fitness = val
            best_solution = positions[i].copy()
        trace.append(best_fitness)

    if fes >= max_fes:
        return best_fitness, best_solution, trace

    # Generational PSO loop
    max_iterations = max_fes * 20
    iteration = 0

    while fes < max_fes and iteration < max_iterations:
        iteration += 1

        # Calculate inertia weight if linearly decreasing
        if is_linear_w:
            w = w_start - (w_start - w_end) * min(1.0, fes / max_fes)

        # Determine social attractors
        if topology == "star":
            g_idx = int(np.argmin(pbest_fit))
            attractors = pbest_pos[g_idx]  # Broadcast across all particles
        else:
            attractors = get_ring_best(pbest_pos, pbest_fit, neighborhood_size)

        # Generate stochastic acceleration multipliers
        r1 = rng.random(size=(population_size, dim))
        r2 = rng.random(size=(population_size, dim))

        # Velocity update
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_pos - positions)
            + c2 * r2 * (attractors - positions)
        )

        # Velocity clamping to [-v_max, v_max]
        velocities = np.clip(velocities, -v_max, v_max)

        # Position update (particles are allowed to fly outside bounds without clamping)
        positions = positions + velocities

        # Evaluation & Personal/Global best updates
        for i in range(population_size):
            if fes >= max_fes:
                break

            # Only evaluate fitness if particle is within search boundaries
            if np.all((positions[i] >= lo) & (positions[i] <= hi)):
                val = objective(positions[i])
                fes += 1

                if val < pbest_fit[i]:
                    pbest_fit[i] = val
                    pbest_pos[i] = positions[i].copy()

                    if val < best_fitness:
                        best_fitness = val
                        best_solution = positions[i].copy()

                trace.append(best_fitness)
            else:
                # Outside bounds: do not evaluate fitness, do not consume FES
                pass

    return best_fitness, best_solution, trace
