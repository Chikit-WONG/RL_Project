# Gymnasium 8x8 Benchmark Update

This update adds a stricter stochastic 8x8 benchmark based on Gymnasium.

## Why This Update Was Added

The earlier custom random-map generator used high frozen-tile probabilities and carved safe paths. Many generated maps were therefore relatively easy, and several Value Iteration oracle scores were close to 1.0. That result was useful as a first generalization test, but it should not be treated as a hard FrozenLake benchmark.

The new experiment uses:

- Gymnasium standard 8x8 FrozenLake.
- Gymnasium `generate_random_map(size=8, p=...)`.
- `slippery=True`.
- `p = 0.8, 0.7, 0.6, 0.5`.
- 3 random maps per p value.
- 7 tabular algorithms.
- 10000 training episodes per curve.

## Main Result

Across the 13 maps, the average final success rates are:

| Rank | Algorithm | Mean final success |
|---:|---|---:|
| 1 | Optimistic Q | 0.235 |
| 2 | SARSA(lambda) | 0.143 |
| 3 | Expected SARSA | 0.135 |
| 4 | Q-learning | 0.134 |
| 5 | Q(lambda) | 0.128 |
| 6 | SARSA | 0.113 |
| 7 | Dyna-Q | 0.072 |

The Gymnasium random maps are much harder than the earlier custom maps. For example, several `p=0.5` and `p=0.6` maps have oracle success close to zero, meaning even a model-based oracle has very low empirical success under the slippery transition dynamics.

## Important Interpretation

The earlier custom random-map results should be described as easy-to-medium controlled random-map tests. The Gymnasium results are a more credible harder benchmark.

Recommended conclusion:

> Optimistic Q-learning remains the strongest learned tabular method, but the Gymnasium random maps show that harder stochastic 8x8 FrozenLake is far from solved under the current budget. Therefore, oracle gap and per-map difficulty must be reported alongside mean success.

## New Files

Figures:

- `figures/main/gymnasium8/gym8_final_success_heatmap.png`
- `figures/main/gymnasium8/gym8_oracle_gap_heatmap.png`
- `figures/main/gymnasium8/gym8_mean_learning_curve.png`
- `figures/main/gymnasium8/gym8_final_success_points_by_group.png`
- `figures/main/gymnasium8/per_map/`

CSV:

- `results_csv/gymnasium8/gymnasium8_algorithm_ranking.csv`
- `results_csv/gymnasium8/gymnasium8_final_success.csv`
- `results_csv/gymnasium8/gymnasium8_learning_curves.csv`
- `results_csv/gymnasium8/gymnasium8_map_manifest.csv`
- `results_csv/gymnasium8/gymnasium8_map_summary.csv`

Code:

- `code/focused_random_map/run_gymnasium8_experiments.py`
