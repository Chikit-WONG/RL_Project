# Stochastic FrozenLake RL Project Submission

This package is organized around the final report narrative:

1. We use the Gymnasium standard 8x8 map as an important standard benchmark.
2. We then tried Gymnasium-generated random maps.
3. Many low-p Gymnasium random maps had near-zero oracle success under `slippery=True`, so they were not suitable for fair algorithm ranking.
4. We therefore used controlled stochastic maps as the main comparison benchmark.
5. We compared seven tabular RL algorithms, added Optuna tuning, and used Value Iteration as an oracle reference.

## Folder Structure

```text
RL_Project_Submission_FINAL_20260507/
|-- code/
|   |-- core/
|   |-- experiments/
|   |-- analysis/
|   |-- requirements.txt
|-- figures/
|   |-- 01_report_used_compact/
|   |-- 02_main_controlled_results/
|   |-- 03_diagnostic_gymnasium8/
|   |-- 04_supplemental_per_map/
|   |-- 05_all_figures_archive/
|-- results_csv/
|   |-- 01_main_controlled/
|   |-- 02_diagnostic_gymnasium8/
|-- docs/
|-- report_overleaf_light/
```

## Main Benchmark

The main algorithm comparison uses controlled stochastic maps:

- Standard 4x4 slippery FrozenLake.
- Designed 4x4 maps with exactly four holes.
- Gymnasium standard 8x8 slippery FrozenLake as a standard reference.
- Fixed 8x8 slippery FrozenLake.
- Controlled random 8x8 maps with `p = 0.80, 0.85, 0.90, 0.95`.

All reported main experiments use `slippery=True`.

## Algorithms

The main comparison includes:

- Q-learning
- SARSA
- Expected SARSA
- SARSA(lambda)
- Q(lambda)
- Optimistic Q-learning
- Dyna-Q

Value Iteration is used as an oracle reference, not as a model-free baseline.

## Recommended Reading Order

1. `report_overleaf_light/main.tex`
2. `docs/中文文件结构说明.md`
3. `docs/图表阅读指南.md`
4. `docs/中文结果整理.md`
5. `docs/algorithm_qa_notes.md`

## Reproduction

Install dependencies:

```bash
cd code
python -m pip install -r requirements.txt
```

Run the controlled 8x8 long-budget experiment:

```bash
cd code/experiments
python run_random8_learning_curves.py --suite random8 --output-dir random8_learning_curves_long_v1 --episodes 10000 --eval-interval 500 --eval-episodes 50 --oracle-eval-episodes 300 --maps-per-prob 5 --seeds 1 --max-steps 200 --planning-steps 15
```

Run the Gymnasium diagnostic trial:

```bash
cd code/experiments
python run_gymnasium8_experiments.py --output-dir gymnasium8_random_maps_v1 --probs 0.8,0.7,0.6,0.5 --maps-per-prob 3 --episodes 10000 --eval-interval 500 --eval-episodes 100 --oracle-eval-episodes 1000 --max-steps 200 --planning-steps 15
```

The submitted CSV and figures are already included, so rerunning is optional.

## Figure Policy

`figures/02_main_controlled_results/` and `figures/03_diagnostic_gymnasium8/` are the recommended figures for the report and presentation.

`figures/05_all_figures_archive/` keeps all current valid figures from the clean working package, including grouped learning curves and individual Gymnasium per-map plots, so no previous useful figure is lost.
