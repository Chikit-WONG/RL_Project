# Focused Stochastic FrozenLake Code

This directory contains the runnable code for the current stochastic FrozenLake experiments.

## Scope

- All main experiments use `slippery=True`.
- Standard 4x4 uses the Gym-style map `SFFF/FHFH/FFFH/HFFG`.
- Designed random 4x4 maps use exactly 4 holes and are filtered by reachability and oracle performance.
- Random 8x8 maps use `p=0.80,0.85,0.90,0.95`, with 5 maps per setting.
- The latest random 8x8 learning curves use 10000 training episodes.

## Files

- `enhanced_frozenlake.py`: shared environment and tabular RL algorithms.
- `map_env_utils.py`: explicit map environment, random map generation, and evaluation helpers.
- `designed_random_map_experiments.py`: controlled random-map generation and designed 4x4 experiments.
- `run_standard_4x4_focused.py`: standard 4x4 slippery experiment.
- `plot_standard_4x4_slippery.py`: plotting for the standard 4x4 experiment.
- `run_random8_learning_curves.py`: long-budget per-map learning curves for `designed4` and `random8`.
- `make_long_budget_report_assets.py`: report figure/table generation from long-budget CSV files.
- `requirements.txt`: dependencies.

## Install

```bash
python -m pip install -r requirements.txt
```

## Standard 4x4 Slippery

```bash
python run_standard_4x4_focused.py --output-dir standard_4x4_slippery_focused_v2 --episodes 2000 --eval-interval 50 --eval-episodes 500 --seeds 5 --planning-steps 5
python plot_standard_4x4_slippery.py --input standard_4x4_slippery_focused_v2/standard_4x4_runs.csv --output-dir standard_4x4_slippery_focused_v2/figures
```

## Designed Random 4x4 Long-Budget Curves

```bash
python run_random8_learning_curves.py --suite designed4 --designed4-manifest ../../results_csv/designed4_learning_curve_map_manifest.csv --output-dir designed4_learning_curves_long_v1 --episodes 2000 --eval-interval 50 --eval-episodes 100 --oracle-eval-episodes 300 --seeds 1 --max-steps 100 --planning-steps 10
```

## Random 8x8 Slippery Long-Budget Curves

```bash
python run_random8_learning_curves.py --suite random8 --output-dir random8_learning_curves_long_v1 --episodes 10000 --eval-interval 500 --eval-episodes 50 --oracle-eval-episodes 300 --maps-per-prob 5 --seeds 1 --max-steps 200 --planning-steps 15
```

The random 8x8 command runs:

- 20 random 8x8 maps
- 7 tabular algorithms
- 10000 episodes per curve
- evaluation every 500 episodes

The script supports resume. If a run is interrupted, rerunning the same command skips curves that already reached the final episode.

## Generate Long-Budget Figures And Tables

```bash
python make_long_budget_report_assets.py
```

## Gymnasium 8x8 Harder Benchmark

```bash
python run_gymnasium8_experiments.py --output-dir gymnasium8_random_maps_v1 --probs 0.8,0.7,0.6,0.5 --maps-per-prob 3 --episodes 10000 --eval-interval 500 --eval-episodes 100 --oracle-eval-episodes 1000 --max-steps 200 --planning-steps 15
```

This uses Gymnasium's standard 8x8 map and `generate_random_map`. It is harder and more credible than the earlier custom carved-path random maps.
