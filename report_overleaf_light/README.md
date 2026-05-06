# Overleaf Light Report

This folder is the lightweight Overleaf version of the report.

Upload the whole folder to Overleaf and compile `main.tex` with `pdfLaTeX`.

## Why This Version Is Light

The original report figures are kept in the main submission package, but this folder uses compressed JPG figures in `figures_compact/` to avoid Overleaf free-plan compile timeouts.

## Report Narrative

The report follows this logic:

1. Gymnasium standard 8x8 is used as an important standard benchmark.
2. Gymnasium generated random maps are used as a diagnostic trial.
3. Many low-p generated random maps have near-zero oracle success under `slippery=True`.
4. Controlled stochastic maps are therefore used as the main algorithm-comparison benchmark.
5. Optimistic Q-learning is strongest on the controlled 8x8 benchmark.

## Included Files

```text
main.tex
references.bib
figures_compact/
```

The report currently references 12 compact figures. All figure paths have been checked.

## Optuna Figure Note

The Optuna figure excludes Double Q-learning. Double Q appeared in an early short-budget screening table with an anomalously low value, but it was not used in the final tuned comparison. The final Optuna figure uses the tuned validation scores for:

- Optimistic Q-learning
- Dyna-Q
- Q-learning
- Expected SARSA

