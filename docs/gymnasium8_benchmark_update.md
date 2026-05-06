# Gymnasium Standard 8x8 and Random-Map Diagnostic Trial

This update records two related results: the Gymnasium standard 8x8 map, which is an important standard benchmark, and the Gymnasium random-map trial, which motivated the final controlled-map benchmark design.

## Why This Trial Was Added

The Gymnasium standard 8x8 map is not a failure case. It is a useful sanity check and standard comparison because it is fixed, reproducible, stochastic under `slippery=True`, and neither trivial nor near-impossible. In our run, its oracle success is about 0.861 and several tabular algorithms achieve strong learned policies.

The initial idea was to use Gymnasium's standard map tools directly, because `generate_random_map` guarantees that a path exists from start to goal. In slippery FrozenLake, however, deterministic reachability is not enough: a connected map can still be almost impossible when each move can slip into a neighboring direction.

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

The standard Gymnasium 8x8 map should be used as a standard benchmark. The generated low-p random maps should not be used as the main algorithm-ranking benchmark. They are diagnostic evidence for why controlled maps were needed.

Recommended conclusion:

> Direct Gymnasium random maps exposed many near-zero oracle cases, so the final project uses controlled stochastic maps for fair algorithm comparison. The Gymnasium results remain useful as a stress test and as evidence that map generation must be treated as part of the experimental design.

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
