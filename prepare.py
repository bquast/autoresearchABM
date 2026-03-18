"""
autoresearchABM — fixed simulation harness.
Equivalent of prepare.py in the LLM autoresearch setup.

Contains:
  - Fixed constants (grid size, steps, seeds, time budget)
  - The Schelling ABM engine
  - The evaluation metric: mean interface length at equilibrium (lower = more segregated)
  - run_experiment() — the single entry point called by train.py

DO NOT MODIFY THIS FILE.
The agent modifies train.py only.
"""

import math
import time
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Fixed constants — do not change
# ---------------------------------------------------------------------------

GRID_SIZE     = 100          # NxN grid
N_SEEDS       = 20           # random seeds for evaluation averaging
TIME_BUDGET   = 60           # wall-clock seconds per experiment (1 minute)
MAX_STEPS     = 5_000        # max simulation steps per seed
EVAL_STEPS    = 500          # steps over which to average the metric at end
EMPTY_FRAC    = 0.10         # fraction of cells left empty (fixed)

# ---------------------------------------------------------------------------
# Schelling ABM engine
# ---------------------------------------------------------------------------

@dataclass
class SchellingConfig:
    """
    Parameters the agent is free to tune in train.py.

    Fields
    ------
    tolerance : float
        Fraction of same-type neighbours an agent requires to be satisfied.
        Range [0, 1]. Classic value: 0.3–0.5.
    density : float
        Fraction of non-empty cells occupied (rest are empty, capped by EMPTY_FRAC).
        Range (0, 1 - EMPTY_FRAC].
    n_groups : int
        Number of distinct agent types. Must be >= 2.
    neighborhood : str
        'moore'   — 8 neighbours (default)
        'neumann' — 4 neighbours
        'extended'— 24 neighbours (5x5 minus centre)
    move_rule : str
        'random'  — dissatisfied agents jump to a random empty cell
        'nearest' — dissatisfied agents move to the nearest satisfying empty cell
    update_order : str
        'random'  — agents chosen uniformly at random each step
        'shuffled'— full random permutation each step
    """
    tolerance     : float = 0.375
    density       : float = 0.90
    n_groups      : int   = 2
    neighborhood  : str   = "moore"
    move_rule     : str   = "random"
    update_order  : str   = "random"


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


def interface_length(grid: np.ndarray) -> float:
    """
    Count edges between cells of different non-zero types (Moore neighbourhood).
    Normalised by total number of occupied cells so it's grid-size-independent.
    Higher = more mixed; lower = more segregated.
    This is the metric we report. Agents try to find configurations that
    minimise or maximise it depending on the experiment goal.
    """
    occupied = (grid > 0)
    n_occupied = occupied.sum()
    if n_occupied == 0:
        return 0.0
    count = 0
    for dr, dc in [(-1, 0), (0, -1)]:  # only count each edge once
        rolled = np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
        both_occupied = occupied & (np.roll(np.roll(occupied, dr, axis=0), dc, axis=1))
        count += (both_occupied & (grid != rolled)).sum()
    return count / n_occupied


def run_single(cfg: SchellingConfig, rng: np.random.Generator) -> float:
    """
    Run one Schelling simulation to equilibrium (or MAX_STEPS).
    Returns mean interface_length over the last EVAL_STEPS steps.
    """
    N = GRID_SIZE
    offsets = _neighbour_offsets(cfg.neighborhood)

    # Initialise grid: 0 = empty, 1..n_groups = agent types
    n_cells = N * N
    n_occupied = int(cfg.density * n_cells * (1 - EMPTY_FRAC) + 0.5)
    n_occupied = min(n_occupied, int((1 - EMPTY_FRAC) * n_cells))
    n_per_group = n_occupied // cfg.n_groups

    labels = np.zeros(n_cells, dtype=np.int8)
    for g in range(cfg.n_groups):
        labels[g * n_per_group:(g + 1) * n_per_group] = g + 1
    rng.shuffle(labels)
    grid = labels.reshape(N, N)

    metric_window = []

    for step in range(MAX_STEPS):
        # Identify occupied cells
        occupied_rc = np.argwhere(grid > 0)
        empty_rc    = np.argwhere(grid == 0)
        if len(empty_rc) == 0:
            break

        if cfg.update_order == "shuffled":
            rng.shuffle(occupied_rc)
        else:  # random: sample with replacement (fast approx)
            idx = rng.integers(0, len(occupied_rc), size=len(occupied_rc))
            occupied_rc = occupied_rc[idx]

        moved = False
        for r, c in occupied_rc:
            agent_type = grid[r, c]
            if agent_type == 0:
                continue

            # Count same-type neighbours
            n_rows = (r + offsets[:, 0]) % N
            n_cols = (c + offsets[:, 1]) % N
            neighbour_vals = grid[n_rows, n_cols]
            n_neighbours = len(offsets)
            n_same = (neighbour_vals == agent_type).sum()
            satisfied = (n_same / n_neighbours) >= cfg.tolerance

            if not satisfied:
                if len(empty_rc) == 0:
                    continue
                if cfg.move_rule == "nearest":
                    # Move to nearest empty cell (Euclidean)
                    dists = np.abs(empty_rc[:, 0] - r) + np.abs(empty_rc[:, 1] - c)
                    target_idx = np.argmin(dists)
                else:
                    target_idx = rng.integers(0, len(empty_rc))

                tr, tc = empty_rc[target_idx]
                grid[tr, tc] = agent_type
                grid[r, c] = 0
                empty_rc[target_idx] = [r, c]
                moved = True

        # Collect metric in final window
        if step >= MAX_STEPS - EVAL_STEPS:
            metric_window.append(interface_length(grid))

        # Early stopping: no moves in this step
        if not moved and step > 10:
            remaining = MAX_STEPS - step - 1
            last = interface_length(grid)
            metric_window.extend([last] * min(remaining, EVAL_STEPS))
            break

    if not metric_window:
        metric_window = [interface_length(grid)]

    return float(np.mean(metric_window))


# ---------------------------------------------------------------------------
# Public evaluation entry point — called by train.py
# ---------------------------------------------------------------------------

def run_experiment(cfg: SchellingConfig) -> dict:
    """
    Run the full evaluation: N_SEEDS independent runs, TIME_BUDGET wall-clock cap.

    Returns a dict with:
        val_il       : float  — mean interface length (the primary metric, lower = more segregated)
        val_il_std   : float  — std across seeds
        seeds_run    : int    — how many seeds completed before time ran out
        elapsed_s    : float  — wall-clock seconds used
        steps_budget : int    — MAX_STEPS (fixed reference)
    """
    t0 = time.time()
    results = []

    for seed in range(N_SEEDS):
        elapsed = time.time() - t0
        if elapsed >= TIME_BUDGET:
            break
        rng = np.random.default_rng(seed)
        il = run_single(cfg, rng)
        results.append(il)

    elapsed = time.time() - t0
    vals = np.array(results)

    return {
        "val_il":       float(vals.mean()) if len(vals) else float("nan"),
        "val_il_std":   float(vals.std())  if len(vals) else float("nan"),
        "seeds_run":    len(vals),
        "elapsed_s":    elapsed,
        "steps_budget": MAX_STEPS,
    }