"""
autoresearchABM — fixed simulation harness.
Contains:
  - Fixed constants (grid size, steps, seeds, time budget)
  - The SIR ABM engine
  - The evaluation metric: mean peak infection prevalence (lower = flatter curve)
  - run_experiment() — the single entry point called by train.py

DO NOT MODIFY THIS FILE.
The agent modifies train.py only.
"""

import time
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fixed constants — do not change
# ---------------------------------------------------------------------------

GRID_SIZE   = 100        # NxN grid
N_SEEDS     = 20         # random seeds for evaluation averaging
TIME_BUDGET = 60         # wall-clock seconds per experiment (1 minute)
MAX_STEPS   = 1_000      # max simulation steps per seed

# ---------------------------------------------------------------------------
# SIR ABM engine
# ---------------------------------------------------------------------------

# Agent states
S = 0  # Susceptible
I = 1  # Infected
R = 2  # Recovered

@dataclass
class SIRConfig:
    """
    Parameters the agent is free to tune in train.py.

    Fields
    ------
    beta : float
        Probability of transmission per infected neighbour per step.
        Range (0, 1]. Higher = faster spread.
    gamma : float
        Probability an infected agent recovers each step.
        Range (0, 1]. Higher = shorter infectious period.
    initial_infected : float
        Fraction of agents seeded as infected at t=0.
        Range (0, 1].
    neighborhood : str
        'moore'    — 8 neighbours
        'neumann'  — 4 neighbours
        'extended' — 24 neighbours (5x5 minus centre)
    reinfection : bool
        If False: classic SIR (R agents are permanently immune).
        If True:  SIRS model (R agents return to S after recovery,
                  with probability delta per step).
    delta : float
        Rate at which recovered agents lose immunity (only used if
        reinfection=True). Range (0, 1].
    rewire_prob : float
        Probability of rewiring each neighbour link to a random cell,
        creating a small-world network effect. 0.0 = pure grid.
        Range [0, 1].
    """
    beta             : float = 0.3
    gamma            : float = 0.1
    initial_infected : float = 0.01
    neighborhood     : str   = "moore"
    reinfection      : bool  = False
    delta            : float = 0.01
    rewire_prob      : float = 0.0


def _neighbour_offsets(neighborhood: str) -> np.ndarray:
    if neighborhood == "moore":
        offs = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    elif neighborhood == "neumann":
        offs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif neighborhood == "extended":
        offs = [(dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0)]
    else:
        raise ValueError(f"Unknown neighborhood: {neighborhood!r}. Use 'moore', 'neumann', or 'extended'.")
    return np.array(offs, dtype=np.int32)


def _build_neighbour_map(N, offsets, rewire_prob, rng):
    """
    Pre-compute neighbour indices for every cell.
    If rewire_prob > 0, each link is independently rewired to a random cell
    (small-world effect). Returns array of shape (N*N, n_neighbours).
    """
    n_cells = N * N
    n_neighbours = len(offsets)
    neighbours = np.empty((n_cells, n_neighbours), dtype=np.int32)

    for cell in range(n_cells):
        r, c = divmod(cell, N)
        for j, (dr, dc) in enumerate(offsets):
            nr = (r + dr) % N
            nc = (c + dc) % N
            neighbours[cell, j] = nr * N + nc

    if rewire_prob > 0.0:
        mask = rng.random(neighbours.shape) < rewire_prob
        random_targets = rng.integers(0, n_cells, size=neighbours.shape)
        neighbours = np.where(mask, random_targets, neighbours)

    return neighbours


def run_single(cfg: SIRConfig, rng: np.random.Generator) -> dict:
    """
    Run one SIR simulation.
    Returns dict with peak_prevalence and final_attack_rate.
    """
    N = GRID_SIZE
    n_cells = N * N
    offsets = _neighbour_offsets(cfg.neighborhood)
    neighbours = _build_neighbour_map(N, offsets, cfg.rewire_prob, rng)

    # Initialise: all susceptible, then seed infections
    grid = np.full(n_cells, S, dtype=np.int8)
    n_initial = max(1, int(cfg.initial_infected * n_cells))
    seed_cells = rng.choice(n_cells, size=n_initial, replace=False)
    grid[seed_cells] = I

    ever_infected = np.zeros(n_cells, dtype=bool)
    ever_infected[seed_cells] = True

    peak_prevalence = n_initial / n_cells

    for _ in range(MAX_STEPS):
        infected_cells = np.where(grid == I)[0]
        if len(infected_cells) == 0:
            break

        new_grid = grid.copy()

        # Transmission: for each infected cell, expose neighbours
        for cell in infected_cells:
            nbrs = neighbours[cell]
            susceptible_nbrs = nbrs[grid[nbrs] == S]
            if len(susceptible_nbrs) == 0:
                continue
            transmit = rng.random(len(susceptible_nbrs)) < cfg.beta
            newly_infected = susceptible_nbrs[transmit]
            new_grid[newly_infected] = I
            ever_infected[newly_infected] = True

        # Recovery
        recover_mask = (grid == I) & (rng.random(n_cells) < cfg.gamma)
        new_grid[recover_mask] = R

        # Reinfection (SIRS)
        if cfg.reinfection:
            reinfect_mask = (grid == R) & (rng.random(n_cells) < cfg.delta)
            new_grid[reinfect_mask] = S

        grid = new_grid

        prevalence = (grid == I).sum() / n_cells
        if prevalence > peak_prevalence:
            peak_prevalence = prevalence

    final_attack_rate = ever_infected.sum() / n_cells

    return {
        "peak_prevalence":   peak_prevalence,
        "final_attack_rate": final_attack_rate,
    }


# ---------------------------------------------------------------------------
# Public evaluation entry point — called by train.py
# ---------------------------------------------------------------------------

def run_experiment(cfg: SIRConfig) -> dict:
    """
    Run the full evaluation: N_SEEDS independent runs, TIME_BUDGET wall-clock cap.

    Returns a dict with:
        val_pi       : float — mean peak infection prevalence (primary metric, lower = flatter curve)
        val_pi_std   : float — std of peak prevalence across seeds
        val_ar       : float — mean final attack rate across seeds
        val_ar_std   : float — std of final attack rate across seeds
        seeds_run    : int   — seeds completed before time ran out
        elapsed_s    : float — wall-clock seconds used
    """
    t0 = time.time()
    peak_results = []
    ar_results   = []

    for seed in range(N_SEEDS):
        if time.time() - t0 >= TIME_BUDGET:
            break
        rng = np.random.default_rng(seed)
        result = run_single(cfg, rng)
        peak_results.append(result["peak_prevalence"])
        ar_results.append(result["final_attack_rate"])

    elapsed = time.time() - t0
    peak = np.array(peak_results)
    ar   = np.array(ar_results)

    return {
        "val_pi":     float(peak.mean()) if len(peak) else float("nan"),
        "val_pi_std": float(peak.std())  if len(peak) else float("nan"),
        "val_ar":     float(ar.mean())   if len(ar)   else float("nan"),
        "val_ar_std": float(ar.std())    if len(ar)   else float("nan"),
        "seeds_run":  len(peak_results),
        "elapsed_s":  elapsed,
    }