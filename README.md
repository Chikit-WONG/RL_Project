# FrozenLake Stochastic RL Experiments

This folder is prepared for uploading to a GitHub branch of the project repository:

```text
https://github.com/Chikit-WONG/RL_Project
```

It contains the current main experimental deliverables for stochastic FrozenLake (`slippery=True`): runnable code, selected figures, result CSV files, and Chinese documentation for result interpretation.

## What Is Included

```text
.
|-- code/
|   |-- focused_random_map/
|-- figures/
|   |-- main/
|       |-- per_map_designed4/
|       |-- per_map_random8/
|-- results_csv/
|-- docs/
|-- README.md
|-- GITHUB_BRANCH_WORKFLOW.md
|-- .gitignore
```

The formal experimental scope is:

- Standard 4x4 slippery FrozenLake.
- Designed random 4x4 slippery maps with exactly 4 holes.
- Random 8x8 slippery maps with `p = 0.80, 0.85, 0.90, 0.95`, 5 maps per setting.
- Gymnasium 8x8 benchmark: standard 8x8 plus Gymnasium `generate_random_map` at `p = 0.8, 0.7, 0.6, 0.5`, 3 maps per setting. This is the harder random-map benchmark and should be used for conservative claims.
- Seven tabular RL algorithms: Q-learning, SARSA, Expected SARSA, SARSA(lambda), Q(lambda), Optimistic Q-learning, and Dyna-Q.
- Value Iteration as an oracle reference, not as a model-free baseline.
- Long-budget learning curves and ablation figures.

## Install Dependencies

Use a normal Python environment. Conda is fine, but no machine-specific path is required.

```bash
cd code/focused_random_map
python -m pip install -r requirements.txt
```

## Reproduce Main Runs

Standard 4x4 slippery:

```bash
cd code/focused_random_map
python run_standard_4x4_focused.py --output-dir standard_4x4_slippery_focused_v2 --episodes 2000 --eval-interval 50 --eval-episodes 500 --seeds 5 --planning-steps 5
python plot_standard_4x4_slippery.py --input standard_4x4_slippery_focused_v2/standard_4x4_runs.csv --output-dir standard_4x4_slippery_focused_v2/figures
```

Designed random 4x4 long-budget curves:

```bash
cd code/focused_random_map
python run_random8_learning_curves.py --suite designed4 --designed4-manifest ../../results_csv/designed4_learning_curve_map_manifest.csv --output-dir designed4_learning_curves_long_v1 --episodes 2000 --eval-interval 50 --eval-episodes 100 --oracle-eval-episodes 300 --seeds 1 --max-steps 100 --planning-steps 10
```

Random 8x8 slippery long-budget curves:

```bash
cd code/focused_random_map
python run_random8_learning_curves.py --suite random8 --output-dir random8_learning_curves_long_v1 --episodes 10000 --eval-interval 500 --eval-episodes 50 --oracle-eval-episodes 300 --maps-per-prob 5 --seeds 1 --max-steps 200 --planning-steps 15
```

Gymnasium 8x8 harder benchmark:

```bash
cd code/focused_random_map
python run_gymnasium8_experiments.py --output-dir gymnasium8_random_maps_v1 --probs 0.8,0.7,0.6,0.5 --maps-per-prob 3 --episodes 10000 --eval-interval 500 --eval-episodes 100 --oracle-eval-episodes 1000 --max-steps 200 --planning-steps 15
```

Regenerate report assets after the runs:

```bash
cd code/focused_random_map
python make_long_budget_report_assets.py
```

## Key Outputs

Recommended figures:

- `figures/main/stochastic_success_heatmap.png`
- `figures/main/random8_learning_curves_all_maps_panel.png`
- `figures/main/random8_final_success_heatmap_by_map.png`
- `figures/main/random8_oracle_gap_heatmap_by_map.png`
- `figures/main/random8_auc_heatmap_by_map.png`
- `figures/main/gymnasium8/gym8_final_success_heatmap.png`
- `figures/main/gymnasium8/gym8_oracle_gap_heatmap.png`
- `figures/main/gymnasium8/gym8_mean_learning_curve.png`
- `figures/main/exploration_ablation_fixed_8x8.png`
- `figures/main/reward_propagation_ablation_fixed_8x8.png`

Recommended CSV tables:

- `results_csv/stochastic_core_summary_long_budget.csv`
- `results_csv/random8_long_budget_algorithm_ranking.csv`
- `results_csv/random8_long_budget_final_success.csv`
- `results_csv/designed4_long_budget_algorithm_ranking.csv`
- `results_csv/standard_4x4_slippery_rerun_aggregate.csv`
- `results_csv/gymnasium8/gymnasium8_algorithm_ranking.csv`
- `results_csv/gymnasium8/gymnasium8_map_summary.csv`

## Documentation

Start from:

```text
docs/README.md
```

The docs include Chinese result summaries, figure-reading guidance, algorithm QA notes, ablation explanations, and a file-structure guide.

For the latest harder-map correction, see:

```text
docs/gymnasium8_benchmark_update.md
```

## GitHub Upload

If you do not own the original repository, do not push directly to its `main` branch. Use a fork or a new branch and submit a pull request. See:

```text
GITHUB_BRANCH_WORKFLOW.md
```
