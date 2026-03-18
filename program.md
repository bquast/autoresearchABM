# autoresearchABM

Autonomous hyperparameter search over a Schelling segregation ABM.
Mirrors the structure of karpathy/autoresearch but for agent-based modelling.

## The model

Each experiment runs a **Schelling segregation** simulation: agents on a 2D grid
move when they are dissatisfied with the fraction of same-type neighbours around them.
The system converges to a steady state that is more or less segregated depending
on the parameters.

**The metric is `val_il` (validation interface length): lower is more segregated.**

Interface length counts edges between cells of different agent types, normalised
by the number of occupied cells. It is evaluated as the mean over the final
`EVAL_STEPS` steps, averaged across `N_SEEDS` independent seeds. Lower is more
segregated (more self-organised clustering); higher is more mixed.

**Your goal: find the parameter configuration that minimises `val_il`.**

## Files

- `prepare.py` — fixed harness: grid, seeds, metric, time budget. **Do not modify.**
- `train.py` — the only file you edit. Contains `SchellingConfig` parameters.
- `results.tsv` — experiment log (untracked by git, you maintain this).
- `program.md` — these instructions (read-only for the agent).

## Setup

To start a new run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from main.
3. **Read the in-scope files**: Read `prepare.py` (understand the engine and metric)
   and `train.py` (the baseline config you will iterate on).
4. **Verify dependencies**: Check that `numpy` is available (`python -c "import numpy"`).
   If not, tell the human to install it (`pip install numpy`).
5. **Initialise results.tsv**: Create it with the header row only. The baseline
   will be recorded after the first run.
6. **Confirm and go**.

Once you get confirmation, kick off the experimentation loop immediately.

## Experimentation

Each experiment runs for a **fixed wall-clock budget of 60 seconds** across
`N_SEEDS = 20` independent random seeds. You launch it as:

```
python train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — this is the only file you edit.
  Change any of: `TOLERANCE`, `DENSITY`, `N_GROUPS`, `NEIGHBORHOOD`,
  `MOVE_RULE`, `UPDATE_ORDER`, or any combination thereof.
- You may also extend `SchellingConfig` by adding new fields to it in `prepare.py`
  and implementing the corresponding logic in `run_single` — **but only if you
  also update `prepare.py` to handle them cleanly**. This is the one exception to
  the read-only rule: you may add new optional parameters with defaults, but you
  must not change existing behaviour, change the metric, or change the fixed
  constants (`GRID_SIZE`, `N_SEEDS`, `TIME_BUDGET`, `MAX_STEPS`, `EVAL_STEPS`,
  `EMPTY_FRAC`).

**What you CANNOT do:**
- Change `GRID_SIZE`, `N_SEEDS`, `TIME_BUDGET`, `MAX_STEPS`, `EVAL_STEPS`, or
  `EMPTY_FRAC` in `prepare.py`.
- Change the `interface_length` function or `run_experiment` signature.
- Install new packages.

## Output format

After the script finishes it prints a summary like this:

```
---
val_il:           0.412300
val_il_std:       0.008100
seeds_run:        20
elapsed_s:        58.3
total_seconds:    58.4
grid_size:        100
tolerance:        0.375
density:          0.90
n_groups:         2
neighborhood:     moore
move_rule:        random
update_order:     random
```

Extract the key metric:

```
grep "^val_il:" run.log
```

## Logging results

Log every experiment to `results.tsv` (tab-separated — no commas in descriptions).
Do NOT git-commit `results.tsv` — leave it untracked.

Columns:

```
commit	val_il	seeds_run	status	description
```

1. Short git commit hash (7 chars)
2. `val_il` achieved (e.g. `0.412300`) — use `0.000000` for crashes
3. Seeds completed (e.g. `20`) — use `0` for crashes
4. Status: `keep`, `discard`, or `crash`
5. Short description of what this experiment tried

Example:

```
commit	val_il	seeds_run	status	description
a1b2c3d	0.412300	20	keep	baseline
b2c3d4e	0.389100	20	keep	tolerance 0.375→0.5
c3d4e5f	0.415000	20	discard	neumann neighborhood
d4e5f6g	0.000000	0	crash	n_groups=-1 (invalid)
```

## The experiment loop

LOOP FOREVER:

1. Check current git state (branch, last commit).
2. Edit `train.py` with your experimental idea.
3. `git add train.py && git commit -m "<short description>"`
4. Run: `python train.py > run.log 2>&1`
5. Extract result: `grep "^val_il:\|^seeds_run:" run.log`
6. If output is empty → run crashed. Check: `tail -n 40 run.log`. Attempt a fix.
   If still failing after 2–3 attempts, log as `crash`, revert, move on.
7. Log the result to `results.tsv`.
8. If `val_il` improved (lower than current best):
   - Keep the commit. This is now the new baseline.
9. If `val_il` is equal or worse:
   - `git reset --hard HEAD~1` to discard.

**Timeouts**: Each run should complete in ~60 seconds. If it exceeds 120 seconds,
kill it (`Ctrl-C`) and treat as a failure.

**Crashes**: Fix trivial bugs (typos, bad values) and re-run. If the idea is
fundamentally broken, skip it and move on.

**Simplicity criterion**: All else equal, simpler is better. A 0.001 improvement
that adds 30 lines of convoluted logic is not worth it. A 0.001 improvement from
deleting a parameter is worth it. A wash that simplifies the config? Keep it.

**NEVER STOP**: Once the loop begins, do not pause to ask the human whether to
continue. Do not ask "should I keep going?" or "is this a good stopping point?".
You are fully autonomous. Run until the human interrupts you.

## Parameter space reference

| Parameter      | Type   | Range / Options                          | Notes                               |
|----------------|--------|------------------------------------------|-------------------------------------|
| `TOLERANCE`    | float  | 0.0 – 1.0                                | Higher → agents harder to satisfy  |
| `DENSITY`      | float  | 0.1 – 0.90                               | Fraction of grid occupied           |
| `N_GROUPS`     | int    | 2 – 6                                    | More groups → less segregation      |
| `NEIGHBORHOOD` | str    | `moore`, `neumann`, `extended`           | Size of local neighbourhood         |
| `MOVE_RULE`    | str    | `random`, `nearest`                      | How dissatisfied agents relocate    |
| `UPDATE_ORDER` | str    | `random`, `shuffled`                     | Order agents are processed          |

The parameter interactions are genuinely non-linear. For example:
- High `TOLERANCE` with `nearest` move rule can produce very tight clusters fast.
- `N_GROUPS > 2` with `extended` neighbourhood tends to resist segregation.
- `DENSITY` near 0.9 leaves little room to move, slowing convergence.

Think carefully about combinations, not just individual knobs.
When you find a direction that helps, explore it more deeply before moving on.

## Hypotheses to explore (starter ideas)

- Does `nearest` move rule produce lower `val_il` than `random`?
- What is the tolerance threshold where segregation transitions sharply?
- Does `extended` neighborhood help or hurt segregation at high density?
- Is there an optimal density that maximises segregation for a given tolerance?
- How does `N_GROUPS = 3` vs `2` affect the equilibrium metric?
- Does `shuffled` update order converge to a different equilibrium than `random`?