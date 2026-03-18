# autoresearchABM

Autonomous hyperparameter search over a spatial SIR epidemic model.
Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The idea: give an AI agent a small but real epidemic simulation and let it experiment
autonomously. It modifies the parameters, runs the simulation, checks if the result
improved, keeps or discards, and repeats. You come back to a log of experiments and
a clearer picture of what drives epidemic dynamics in the model.

The model is a **spatial SIR simulation**: agents on a 100x100 grid transition between
Susceptible, Infected, and Recovered states. The metric is **val_pi** (peak infection
prevalence) — lower is better, literally "flatten the curve". It is averaged over 20
fixed random seeds for stability and comparability across runs.

## How it works

Three files matter:

- **`prepare.py`** — fixed constants, the SIR ABM engine, and the evaluation harness. Not modified by the agent.
- **`train.py`** — the single file the agent edits. All tunable parameters live here: transmission rate, recovery rate, neighbourhood type, network rewiring, immunity waning, and more.
- **`program.md`** — instructions for the agent. Point your agent here and let it go.

Each experiment runs for a **fixed 60-second time budget** across 20 independent random
seeds, so results are directly comparable regardless of parameter changes.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Run a single experiment (~60s)
uv run train.py
```

## Running the agent

Spin up your agent of choice in this repo and prompt it:

```
Have a look at program.md and let's kick off a new experiment — do the setup first.
```

The agent establishes a baseline, then loops autonomously: modify `train.py`,
run the simulation, keep or discard, repeat.

## Project structure

```
prepare.py      — fixed constants, SIR engine, evaluation harness (do not modify)
train.py        — parameters the agent modifies
program.md      — agent instructions
analysis.ipynb  — notebook to visualise results from results.tsv
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`.
- **Fixed time budget.** 60 seconds across 20 seeds. Directly comparable across all parameter changes.
- **No GPU required.** Pure NumPy. Runs on any laptop.
- **Meaningful metric.** Peak prevalence has a direct real-world interpretation — it is the quantity public health policy tries to minimise.
- **Secondary metric included.** `val_ar` (final attack rate) is also computed at no extra cost. See the note in `train.py` for how to switch to it as the primary metric.

## License

MIT