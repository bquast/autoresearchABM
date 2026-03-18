"""
autoresearchABM — agent-editable experiment file.
Equivalent of train.py in the LLM autoresearch setup.

Usage: python train.py

The agent modifies ONLY this file. prepare.py is read-only.
Goal: minimise val_il (interface length at equilibrium).
Lower val_il = more segregated steady state.
"""

import time
from prepare import SchellingConfig, run_experiment, TIME_BUDGET, GRID_SIZE, N_SEEDS

# ---------------------------------------------------------------------------
# Hyperparameters — edit these freely
# ---------------------------------------------------------------------------

# Agent behaviour
TOLERANCE     = 0.375    # fraction of same-type neighbours required [0, 1]
DENSITY       = 0.90     # fraction of grid occupied (rest empty)
N_GROUPS      = 2        # number of distinct agent types (>= 2)

# Spatial structure
NEIGHBORHOOD  = "moore"  # 'moore' (8), 'neumann' (4), 'extended' (24)

# Dynamics
MOVE_RULE     = "random"    # 'random' or 'nearest'
UPDATE_ORDER  = "random"    # 'random' or 'shuffled'

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

t_start = time.time()

cfg = SchellingConfig(
    tolerance    = TOLERANCE,
    density      = DENSITY,
    n_groups     = N_GROUPS,
    neighborhood = NEIGHBORHOOD,
    move_rule    = MOVE_RULE,
    update_order = UPDATE_ORDER,
)

print(f"Grid: {GRID_SIZE}x{GRID_SIZE}  |  Seeds: {N_SEEDS}  |  Budget: {TIME_BUDGET}s")
print(f"Config: tolerance={TOLERANCE}  density={DENSITY}  n_groups={N_GROUPS}")
print(f"        neighborhood={NEIGHBORHOOD}  move_rule={MOVE_RULE}  update_order={UPDATE_ORDER}")
print()

results = run_experiment(cfg)

t_total = time.time() - t_start

# ---------------------------------------------------------------------------
# Summary — grep-friendly, mirrors LLM autoresearch output format
# ---------------------------------------------------------------------------

print("---")
print(f"val_il:           {results['val_il']:.6f}")
print(f"val_il_std:       {results['val_il_std']:.6f}")
print(f"seeds_run:        {results['seeds_run']}")
print(f"elapsed_s:        {results['elapsed_s']:.1f}")
print(f"total_seconds:    {t_total:.1f}")
print(f"grid_size:        {GRID_SIZE}")
print(f"tolerance:        {TOLERANCE}")
print(f"density:          {DENSITY}")
print(f"n_groups:         {N_GROUPS}")
print(f"neighborhood:     {NEIGHBORHOOD}")
print(f"move_rule:        {MOVE_RULE}")
print(f"update_order:     {UPDATE_ORDER}")