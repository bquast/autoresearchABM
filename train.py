"""
autoresearchABM — agent-editable experiment file.
Equivalent of train.py in the LLM autoresearch setup.
Usage: python train.py

The agent modifies ONLY this file. prepare.py is read-only.
Goal: minimise val_pi (peak infection prevalence).
Lower val_pi = flatter epidemic curve.

NOTE — switching the primary metric:
  The experiment also reports val_ar (final attack rate = total fraction ever
  infected). To optimise for that instead, change the final print block to
  report val_ar as the primary metric, and update results.tsv accordingly.
  Both are computed by run_experiment() at no extra cost.
"""

import time
from prepare import SIRConfig, run_experiment, TIME_BUDGET, GRID_SIZE, N_SEEDS

# ---------------------------------------------------------------------------
# Hyperparameters — edit these freely
# ---------------------------------------------------------------------------

# Transmission dynamics
BETA              = 0.3    # transmission probability per infected neighbour [0, 1]
GAMMA             = 0.1    # recovery probability per step [0, 1]
INITIAL_INFECTED  = 0.01   # fraction of grid seeded as infected at t=0

# Spatial structure
NEIGHBORHOOD      = "moore"   # 'moore' (8), 'neumann' (4), 'extended' (24)
REWIRE_PROB       = 0.0       # small-world rewiring probability [0, 1]; 0 = pure grid

# Immunity
REINFECTION       = False  # False = SIR (permanent immunity); True = SIRS
DELTA             = 0.01   # rate of immunity loss (only used if REINFECTION=True)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

t_start = time.time()

cfg = SIRConfig(
    beta             = BETA,
    gamma            = GAMMA,
    initial_infected = INITIAL_INFECTED,
    neighborhood     = NEIGHBORHOOD,
    rewire_prob      = REWIRE_PROB,
    reinfection      = REINFECTION,
    delta            = DELTA,
)

print(f"Grid: {GRID_SIZE}x{GRID_SIZE}  |  Seeds: {N_SEEDS}  |  Budget: {TIME_BUDGET}s")
print(f"Config: beta={BETA}  gamma={GAMMA}  initial_infected={INITIAL_INFECTED}")
print(f"        neighborhood={NEIGHBORHOOD}  rewire_prob={REWIRE_PROB}")
print(f"        reinfection={REINFECTION}  delta={DELTA}")
print()

results = run_experiment(cfg)

t_total = time.time() - t_start

# ---------------------------------------------------------------------------
# Summary — grep-friendly, mirrors LLM autoresearch output format
# ---------------------------------------------------------------------------

print("---")
print(f"val_pi:           {results['val_pi']:.6f}")
print(f"val_pi_std:       {results['val_pi_std']:.6f}")
print(f"val_ar:           {results['val_ar']:.6f}")
print(f"val_ar_std:       {results['val_ar_std']:.6f}")
print(f"seeds_run:        {results['seeds_run']}")
print(f"elapsed_s:        {results['elapsed_s']:.1f}")
print(f"total_seconds:    {t_total:.1f}")
print(f"grid_size:        {GRID_SIZE}")
print(f"beta:             {BETA}")
print(f"gamma:            {GAMMA}")
print(f"initial_infected: {INITIAL_INFECTED}")
print(f"neighborhood:     {NEIGHBORHOOD}")
print(f"rewire_prob:      {REWIRE_PROB}")
print(f"reinfection:      {REINFECTION}")
print(f"delta:            {DELTA}")