# RL Project: FrozenLake Q-Learning

This project implements tabular Q-Learning for the course Track A topic `Game Playing with Q-Learning`. The code trains and evaluates an agent on 4x4 FrozenLake, saves figures and CSV summaries, and is designed to feed directly into the later project report.

## Environment

- Recommended conda environment: `test`
- Python version verified: `3.10.18`
- CPU-only workflow by default
- Login-node execution is the default because this is a lightweight tabular RL task

## Dependencies

The current `test` environment already has `numpy` and `matplotlib`. `gymnasium` is optional because the script can fall back to a native FrozenLake implementation.

```bash
conda activate test
pip install numpy matplotlib
pip install gymnasium  # optional
```

If `gymnasium` is unavailable, `q_learning_game.py` will automatically use the built-in `native` backend.

## Project Files

- `q_learning_game.py`: main training, evaluation, plotting, and export script
- `plan/plan_zh.md`: standalone Chinese implementation plan
- `plan/plan_v1_claude.md`: earlier discussion draft
- `results/figures/`: generated plots
- `results/tables/`: generated CSV summaries
- `results/logs/`: per-run train/eval logs

## How To Run

Run the full enhanced experiment suite:

```bash
conda activate test
cd /hpc2hdd/home/ckwong627/workdir/Class/AIAA3053-Reinforcement_Learning_Principles_and_Methods/Project/RL_Project
python q_learning_game.py --exp all --multi-seed 5
```

Run a smaller smoke test:

```bash
python q_learning_game.py --exp all --episodes 1000 --multi-seed 2
```

Force the native backend:

```bash
python q_learning_game.py --backend native --exp all
```

## CLI Options

```text
--backend {auto,gymnasium,native}
--exp {all,exp1,exp2,exp3}
--episodes N
--seed N
--multi-seed N
--output-dir PATH
```

Default behavior:

- `backend=auto`
- `exp=all`
- `episodes=10000`
- `seed=0`
- `multi-seed=5`
- `output-dir=results`

## Experiments

- `Exp1`: deterministic FrozenLake baseline
- `Exp2`: stochastic FrozenLake baseline
- `Exp3`: alpha ablation with `alpha in {0.01, 0.1, 0.5}`
- Enhanced display mode also runs multi-seed aggregation and exports mean/std comparisons

## Expected Outputs

Key figures:

- `results/figures/exp1_learning_curve.png`
- `results/figures/exp2_learning_curve.png`
- `results/figures/exp2_eval_curve.png`
- `results/figures/exp2_epsilon_curve.png`
- `results/figures/exp2_qtable_heatmap.png`
- `results/figures/exp2_policy_arrows.png`
- `results/figures/exp3_alpha_comparison.png`
- `results/figures/baseline_multi_seed_eval.png`
- `results/figures/alpha_multi_seed_eval.png`
- `results/figures/alpha_final_success_bar.png`
- `results/figures/alpha_final_success_boxplot.png`

Key tables:

- `results/tables/run_summary.csv`
- `results/tables/summary_table.csv`

Logs:

- `results/logs/*_train.csv`
- `results/logs/*_eval.csv`

## Notes

- No GPU is required for this project.
- The script prefers Gymnasium for parity with standard FrozenLake but stays reproducible without it.
- For cluster execution, an optional CPU-only debug-partition script is available at `run_debug_cpu.slurm`.
- The report, report PDF, and Overleaf `.zip` are intentionally left for the next stage after final experiment outputs are stable.
