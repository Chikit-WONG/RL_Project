# Code Guide

This folder contains the runnable code used for the final stochastic FrozenLake project.

## Structure

```text
code/
|-- core/
|   |-- enhanced_frozenlake.py
|-- experiments/
|   |-- enhanced_frozenlake.py
|   |-- credibility_experiments.py
|   |-- run_standard_4x4_focused.py
|   |-- run_random8_learning_curves.py
|   |-- run_gymnasium8_experiments.py
|   |-- designed_random_map_experiments.py
|-- analysis/
|   |-- make_long_budget_report_assets.py
|   |-- make_stochastic_report_assets.py
|   |-- plot_standard_4x4_slippery.py
|-- requirements.txt
```

`core/enhanced_frozenlake.py` is the canonical implementation file. A copy is also placed in `experiments/` so the submitted experiment scripts can run directly with simple local imports.

## Install

```bash
python -m pip install -r requirements.txt
```

## Main Controlled 8x8 Run

```bash
cd experiments
python run_random8_learning_curves.py --suite random8 --output-dir random8_learning_curves_long_v1 --episodes 10000 --eval-interval 500 --eval-episodes 50 --oracle-eval-episodes 300 --maps-per-prob 5 --seeds 1 --max-steps 200 --planning-steps 15
```

## Gymnasium Diagnostic Run

```bash
cd experiments
python run_gymnasium8_experiments.py --output-dir gymnasium8_random_maps_v1 --probs 0.8,0.7,0.6,0.5 --maps-per-prob 3 --episodes 10000 --eval-interval 500 --eval-episodes 100 --oracle-eval-episodes 1000 --max-steps 200 --planning-steps 15
```

