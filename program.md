# autoresearchABM

Autonomous hyperparameter search over a spatial SIR epidemic model.
Mirrors the structure of karpathy/autoresearch but for agent-based modelling.

## The model

Each experiment runs a **spatial SIR simulation**: agents on a 2D grid transition
between Susceptible, Infected, and Recovered states. Infected agents spread the
disease to susceptible neighbours with probability `beta`, and recover with
probability `gamma`. Spatial structure matters — unlike the classic ODE model,
local clustering, network topology, and seeding all affect outcomes.

**The metric is `val_pi` (peak infection prevalence): lower is better.**

Peak prevalence is the highest fraction of the grid simultaneously infected —
"flatten the curve" as an optimisation target. It is averaged across `N_SEEDS = 20`
independent seeds. Lower = flatter epidemic curve = better.

The run also reports `val_ar` (final attack rate: total fraction ever infected).
See the note in `train.py` for how to switch to that as the primary metric instead.

**Your goal: find the parameter configuration that minimises `val_pi`.**

## Files

- `prepare.py` — fixed harness: grid, states, metric, time budget. **Do not modify.**
- `train.py` — the only file you edit. Contains `SIRConfig` parameters.
- `results.tsv` — experiment log (untracked by git, you maintain this).
- `program.md` — these instructions.

## Setup

To start a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files**: Read `prepare.py` (understand the engine and metric)
   and `train.py` (the baseline config you will iterate on).
4. **Verify dependencies**: `python -c "import numpy"`. If missing: `pip install numpy`.
5. **Initialise results.tsv**: Create it with just the header row.
6. **Confirm and go**.

Once you get confirmation, kick off the experimentation loop immediately.

## Experimentation

Each experiment runs for a **fixed wall-clock budget of 60 seconds** across
`N_SEEDS = 20` independent random seeds. Launch as:

```
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` freely: change any of `BETA`, `GAMMA`, `INITIAL_INFECTED`,
  `NEIGHBORHOOD`, `REWIRE_PROB`, `REINFECTION`, `DELTA`, or any combination.
- You may add new optional fields to `SIRConfig` in `prepare.py` and implement
  corresponding logic in `run_single` — but only if you do not change existing
  behaviour, do not change the metric, and do not change the fixed constants
  (`GRID_SIZE`, `N_SEEDS`, `TIME_BUDGET`, `MAX_STEPS`).

**What you CANNOT do:**
- Change `GRID_SIZE`, `N_SEEDS`, `TIME_BUDGET`, or `MAX_STEPS` in `prepare.py`.
- Change the `run_experiment` signature or the `val_pi` / `val_ar` computation.
- Install new packages.

## Output format

```
---
val_pi:           0.312400
val_pi_std:       0.021300
val_ar:           0.748200
val_ar_std:       0.031100
seeds_run:        20
elapsed_s:        42.1
total_seconds:    42.2
grid_size:        100
beta:             0.3
gamma:            0.1
initial_infected: 0.01
neighborhood:     moore
rewire_prob:      0.0
reinfection:      False
delta:            0.01
```

Extract the key metric:

```
grep "^val_pi:" run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated — no commas in descriptions).
Do NOT git-commit `results.tsv`.

```
commit	val_pi	seeds_run	status	description
```

Example:

```
commit	val_pi	seeds_run	status	description
a1b2c3d	0.312400	20	keep	baseline
b2c3d4e	0.287100	20	keep	gamma 0.1→0.2 (faster recovery)
c3d4e5f	0.318000	20	discard	neumann neighborhood
d4e5f6g	0.000000	0	crash	beta=-0.1 (invalid)
```

## The experiment loop

LOOP FOREVER:

1. Check current git state (branch, last commit).
2. Edit `train.py` with your experimental idea.
3. `git add train.py && git commit -m "<short description>"`
4. Run: `python train.py > run.log 2>&1`
5. Extract result: `grep "^val_pi:\|^seeds_run:" run.log`
6. If output is empty → crashed. Check: `tail -n 40 run.log`. Fix or skip.
7. Log to `results.tsv`.
8. If `val_pi` improved (lower): keep the commit. New baseline.
9. If equal or worse: `git reset --hard HEAD~1`.

**Timeouts**: each run should complete in ~60s. Kill and treat as failure if >120s.

**Crashes**: fix trivial issues (bad values, typos) and re-run. Skip fundamentally
broken ideas.

**Simplicity criterion**: a small improvement that adds complex code is not worth it.
A simplification with equal results is always worth keeping.

**NEVER STOP**: do not pause to ask the human whether to continue. You are fully
autonomous. Run until manually interrupted.

## Parameter space reference

| Parameter         | Type   | Range / Options                        | Notes                                      |
|-------------------|--------|----------------------------------------|--------------------------------------------|
| `BETA`            | float  | 0.0 – 1.0                              | Core transmission rate                     |
| `GAMMA`           | float  | 0.0 – 1.0                              | Recovery rate; R0 ≈ beta/gamma * neighbours|
| `INITIAL_INFECTED`| float  | 0.001 – 0.1                            | Seed size; affects early dynamics          |
| `NEIGHBORHOOD`    | str    | `moore`, `neumann`, `extended`         | Larger = faster spread                     |
| `REWIRE_PROB`     | float  | 0.0 – 1.0                              | Small-world shortcuts; accelerates spread  |
| `REINFECTION`     | bool   | True / False                           | Switches SIR → SIRS                        |
| `DELTA`           | float  | 0.0 – 1.0                              | Immunity waning rate (SIRS only)           |

Key relationship: **R0 ≈ beta * n_neighbours / gamma**. The epidemic threshold is
R0 = 1. Below it, the disease dies out; above it, it spreads. The interesting
region for minimising peak prevalence is just above R0 = 1.

## Hypotheses to explore (starter ideas)

- What gamma minimises peak prevalence while keeping the epidemic from dying out?
- Does `neumann` (4 neighbours) produce a flatter curve than `moore` (8)?
- Does a small `rewire_prob` sharply increase or decrease peak prevalence?
- Does a smaller `initial_infected` seed meaningfully flatten the curve?
- At what beta/gamma ratio does the peak transition from near-zero to large?
- Does SIRS (`reinfection=True`) produce higher or lower peak than SIR at same beta/gamma?