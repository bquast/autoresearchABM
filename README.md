# autoresearchABM

Autonomous hyperparameter search over agent-based models. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

The idea: give an AI agent a small but real ABM simulation setup and let it experiment autonomously. It modifies the parameters, runs the simulation, checks if the result improved, keeps or discards, and repeats. You come back to a log of experiments and (hopefully) a deeper understanding of the parameter space.

The model is a **Schelling segregation** simulation: agents on a 2D grid move when they are dissatisfied with the fraction of same-type neighbours around them. The system converges to a steady state that is more or less segregated depending on the parameters. The metric is **val_il** (validation interface length) — lower is more segregated. It is averaged over 20 fixed random seeds for stability.

## How it works

The repo has three files that matter:

- **`prepare.py`** — fixed constants, the Schelling ABM engine, and the evaluation harness. Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains all tunable parameters: tolerance, density, number of groups, neighbourhood type, move rule, update order.
- **`program.md`** — instructions for the agent. Point your agent here and let it go.

Training runs for a **fixed 60-second time budget** across 20 independent random seeds, so experiments are directly comparable regardless of what parameters change.

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

The agent will establish a baseline, then loop autonomously: modify `train.py`, run the simulation, keep or discard, repeat.

## Project structure

```
prepare.py      — fixed constants, ABM engine, evaluation harness (do not modify)
train.py        — parameters the agent modifies
program.md      — agent instructions
analysis.ipynb  — notebook to visualise results from results.tsv
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. Diffs are small and reviewable.
- **Fixed time budget.** Each experiment always runs for 60 seconds across 20 seeds. Results are directly comparable regardless of parameter changes.
- **No GPU required.** Runs on any laptop. Pure NumPy.
- **Meaningful metric.** Interface length has a clear interpretation — it measures the degree of spatial segregation at equilibrium, independent of grid size.

## License

MIT