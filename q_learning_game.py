"""FrozenLake Q-Learning project - main entry point.

Imports all components from the project modules and exposes the CLI.
Run this file directly to execute the full experiment suite.

Module layout
-------------
config.py          - constants, Config, ExperimentSpec
environment.py     - NativeFrozenLakeEnv, GymFrozenLakeAdapter, make_env,
                     generate_random_map
agent.py           - QTable, DoubleQTable, policy classes, train_once,
                     evaluate_policy
utils.py           - I/O helpers, smoothing, summary statistics
plotting.py        - all matplotlib plot functions
q_learning_game.py - CLI (parse_args) and main()
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np

from config import Config, ExperimentSpec
from agent import train_once
from environment import generate_random_map
from utils import (
    ensure_output_dirs,
    export_run_logs,
    write_csv,
    filter_specs,
    build_run_summary,
    build_aggregate_summary,
)
from plotting import (
    plot_learning_curve,
    plot_eval_curve,
    plot_epsilon_curve,
    plot_qtable_heatmap,
    plot_policy_arrows,
    plot_alpha_comparison,
    plot_mean_eval_bands,
    plot_alpha_final_bar,
    plot_alpha_final_boxplot,
    plot_map_heatmap_grid,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run FrozenLake Q-Learning experiments.")
    parser.add_argument("--backend", choices=["auto", "gymnasium", "native"], default="auto")
    parser.add_argument(
        "--exp",
        choices=["all", "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8"],
        default="all",
    )
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--n-maps",
        type=int,
        default=10,
        help="Number of random maps in the Exp6 map pool.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        choices=[5, 6],
        help="If set, restrict Exp5/6 to maps of this side length.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Spec builder
# ---------------------------------------------------------------------------

def _build_all_specs(
    config: Config,
    n_maps: int,
    size_filter: Optional[int],
    maps_5x5: List[List[str]],
    maps_6x6: List[List[str]],
) -> List[ExperimentSpec]:
    """Construct the full list of experiment specifications for Exp1-8."""

    specs: List[ExperimentSpec] = []

    # ------------------------------------------------------------------
    # Exp1-3  (original baseline + alpha ablation)
    # ------------------------------------------------------------------
    specs += [
        ExperimentSpec("exp1", "Exp1 Deterministic Baseline", False, config.alpha, "baseline"),
        ExperimentSpec("exp2", "Exp2 Stochastic Baseline", True, config.alpha, "baseline"),
        ExperimentSpec("exp3_alpha_0.01", "Exp3 Alpha 0.01", True, 0.01, "alpha"),
        ExperimentSpec("exp3_alpha_0.10", "Exp3 Alpha 0.10", True, 0.10, "alpha"),
        ExperimentSpec("exp3_alpha_0.50", "Exp3 Alpha 0.50", True, 0.50, "alpha"),
    ]

    # ------------------------------------------------------------------
    # Exp4  — exploration policy comparison (4x4 stochastic)
    # ------------------------------------------------------------------
    specs += [
        ExperimentSpec("exp4_random",    "Exp4 Random Policy",    True, config.alpha, "policy", policy_type="random"),
        ExperimentSpec("exp4_epsilon",   "Exp4 Epsilon-Greedy",   True, config.alpha, "policy", policy_type="epsilon_greedy"),
        ExperimentSpec("exp4_ucb",       "Exp4 UCB",              True, config.alpha, "policy", policy_type="ucb"),
        ExperimentSpec("exp4_boltzmann", "Exp4 Boltzmann",        True, config.alpha, "policy", policy_type="boltzmann"),
    ]

    # ------------------------------------------------------------------
    # Exp5  — map scale (fixed 5x5 and 6x6, map_id=0)
    # ------------------------------------------------------------------
    scale_configs = []
    if size_filter is None or size_filter == 5:
        scale_configs.append(("5x5", maps_5x5[0]))
    if size_filter is None or size_filter == 6:
        scale_configs.append(("6x6", maps_6x6[0]))
    for size_str, first_map in scale_configs:
        layout = tuple(first_map)
        specs += [
            ExperimentSpec(f"exp5_{size_str}_det", f"Exp5 {size_str} Deterministic",
                           False, config.alpha, "map_scale", map_layout=layout, map_id=0),
            ExperimentSpec(f"exp5_{size_str}_sto", f"Exp5 {size_str} Stochastic",
                           True,  config.alpha, "map_scale", map_layout=layout, map_id=0),
        ]

    # ------------------------------------------------------------------
    # Exp6  — random map pool (each map trained with seed=0 only)
    # ------------------------------------------------------------------
    pool_configs = []
    if size_filter is None or size_filter == 5:
        pool_configs.append(("5x5", maps_5x5))
    if size_filter is None or size_filter == 6:
        pool_configs.append(("6x6", maps_6x6))
    for size_str, pool in pool_configs:
        for i, m in enumerate(pool[:n_maps]):
            specs.append(
                ExperimentSpec(
                    f"exp6_{size_str}_map{i}",
                    f"Exp6 {size_str} Map {i}",
                    True, config.alpha, "map_pool",
                    map_layout=tuple(m), map_id=i,
                )
            )

    # ------------------------------------------------------------------
    # Exp7  — Standard vs Double Q-Learning (4x4 and 6x6 stochastic)
    # ------------------------------------------------------------------
    layout_6x6 = tuple(maps_6x6[0])
    specs += [
        ExperimentSpec("exp7_4x4_std",    "Exp7 4x4 Standard Q",  True, config.alpha, "algo"),
        ExperimentSpec("exp7_4x4_double", "Exp7 4x4 Double Q",    True, config.alpha, "algo",
                       algo="double_q_learning"),
        ExperimentSpec("exp7_6x6_std",    "Exp7 6x6 Standard Q",  True, config.alpha, "algo",
                       map_layout=layout_6x6),
        ExperimentSpec("exp7_6x6_double", "Exp7 6x6 Double Q",    True, config.alpha, "algo",
                       algo="double_q_learning", map_layout=layout_6x6),
    ]

    # ------------------------------------------------------------------
    # Exp8  — reward shaping (4x4 and 6x6 stochastic)
    # ------------------------------------------------------------------
    shaping_variants = [
        ("base",  dict()),
        ("step",  dict(step_penalty=True)),
        ("pot",   dict(reward_shaping=True)),
        ("both",  dict(reward_shaping=True, step_penalty=True)),
    ]
    for size_str, layout_kw in [("4x4", {}), ("6x6", {"map_layout": layout_6x6})]:
        for suffix, extra in shaping_variants:
            specs.append(
                ExperimentSpec(
                    f"exp8_{size_str}_{suffix}",
                    f"Exp8 {size_str} {suffix.capitalize()}",
                    True, config.alpha, "shaping",
                    **layout_kw, **extra,
                )
            )

    return specs


def main() -> None:
    """Run the full experiment suite and export all required artifacts."""

    args = parse_args()
    config = Config(
        n_episodes=args.episodes,
        multi_seed=args.multi_seed,
        default_seed=args.seed,
        output_dir=args.output_dir,
    )
    directories = ensure_output_dirs(config.output_dir)

    # ------------------------------------------------------------------
    # Generate fixed benchmark map pools (BFS-validated, deterministic)
    # ------------------------------------------------------------------
    maps_5x5: List[List[str]] = [generate_random_map(5, hole_prob=0.25, seed=i) for i in range(10)]
    maps_6x6: List[List[str]] = [generate_random_map(6, hole_prob=0.25, seed=i) for i in range(10)]

    all_specs = _build_all_specs(
        config=config,
        n_maps=args.n_maps,
        size_filter=args.size,
        maps_5x5=maps_5x5,
        maps_6x6=maps_6x6,
    )
    selected_specs = filter_specs(all_specs, args.exp)
    seed_values = list(range(config.multi_seed))
    if config.default_seed not in seed_values:
        seed_values.append(config.default_seed)
        seed_values = sorted(seed_values)

    results_by_condition: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for spec in selected_specs:
        # Exp6 map-pool: each map is its own variation — run with seed 0 only
        seeds_for_spec = [0] if spec.family == "map_pool" else seed_values
        for seed in seeds_for_spec:
            result = train_once(spec=spec, config=config, backend=args.backend, seed=seed)
            results_by_condition[spec.name].append(result)
            export_run_logs(result, directories["logs"])

    run_summary_rows: List[Dict[str, object]] = []
    for runs in results_by_condition.values():
        run_summary_rows.extend(build_run_summary(run, config) for run in runs)
    aggregate_summary_rows = build_aggregate_summary(results_by_condition, config)

    write_csv(directories["tables"] / "run_summary.csv", run_summary_rows)
    write_csv(directories["tables"] / "summary_table.csv", aggregate_summary_rows)

    selected_seed = config.default_seed

    # ==================================================================
    # Exp1 plots
    # ==================================================================
    if "exp1" in results_by_condition:
        exp1_run = next(r for r in results_by_condition["exp1"] if r["seed"] == selected_seed)
        plot_learning_curve(
            rewards=cast(np.ndarray, exp1_run["rewards"]),
            smoothed_rewards=cast(np.ndarray, exp1_run["smoothed_rewards"]),
            path=directories["figures"] / "exp1_learning_curve.png",
            title="Exp1 Learning Curve (Deterministic FrozenLake)",
        )

    # ==================================================================
    # Exp2 plots
    # ==================================================================
    if "exp2" in results_by_condition:
        exp2_run = next(r for r in results_by_condition["exp2"] if r["seed"] == selected_seed)
        plot_learning_curve(
            rewards=cast(np.ndarray, exp2_run["rewards"]),
            smoothed_rewards=cast(np.ndarray, exp2_run["smoothed_rewards"]),
            path=directories["figures"] / "exp2_learning_curve.png",
            title="Exp2 Learning Curve (Stochastic FrozenLake)",
        )
        plot_eval_curve(
            eval_rows=cast(List[Dict[str, object]], exp2_run["eval_rows"]),
            path=directories["figures"] / "exp2_eval_curve.png",
            title="Exp2 Evaluation Success Rate",
        )
        plot_epsilon_curve(
            epsilons=cast(np.ndarray, exp2_run["epsilons"]),
            path=directories["figures"] / "exp2_epsilon_curve.png",
            title="Exp2 Epsilon Decay",
        )
        plot_qtable_heatmap(
            q_values=cast(np.ndarray, exp2_run["q_table"]),
            path=directories["figures"] / "exp2_qtable_heatmap.png",
            title="Exp2 Max-Q Heatmap",
        )
        plot_policy_arrows(
            q_values=cast(np.ndarray, exp2_run["q_table"]),
            path=directories["figures"] / "exp2_policy_arrows.png",
            title="Exp2 Greedy Policy",
        )

    # ==================================================================
    # Exp3 plots (alpha ablation)
    # ==================================================================
    alpha_condition_names = [n for n in results_by_condition if n.startswith("exp3_alpha_")]
    if alpha_condition_names:
        alpha_seed_runs = [
            next(r for r in results_by_condition[n] if r["seed"] == selected_seed)
            for n in sorted(alpha_condition_names)
        ]
        plot_alpha_comparison(
            results=alpha_seed_runs,
            path=directories["figures"] / "exp3_alpha_comparison.png",
            title="Exp3 Alpha Ablation (Single Seed)",
        )

    if all(lbl in results_by_condition for lbl in ("exp1", "exp2")):
        plot_mean_eval_bands(
            grouped_results={
                "Exp1 deterministic": results_by_condition["exp1"],
                "Exp2 stochastic": results_by_condition["exp2"],
            },
            path=directories["figures"] / "baseline_multi_seed_eval.png",
            title="Baseline Evaluation Curves Across Seeds",
        )

    if alpha_condition_names:
        alpha_groups: Dict[str, List[Dict[str, object]]] = {
            f"alpha={cast(ExperimentSpec, results_by_condition[n][0]['spec']).alpha:g}": results_by_condition[n]
            for n in sorted(alpha_condition_names)
        }
        plot_mean_eval_bands(
            grouped_results=alpha_groups,
            path=directories["figures"] / "alpha_multi_seed_eval.png",
            title="Alpha Evaluation Curves Across Seeds",
        )
        plot_alpha_final_bar(
            grouped_results=alpha_groups,
            path=directories["figures"] / "alpha_final_success_bar.png",
            title="Final Success Rate by Alpha",
        )
        plot_alpha_final_boxplot(
            grouped_results=alpha_groups,
            path=directories["figures"] / "alpha_final_success_boxplot.png",
            title="Final Success Rate Distribution by Alpha",
        )

    # ==================================================================
    # Exp4 — exploration policy comparison
    # ==================================================================
    exp4_names = [n for n in results_by_condition if n.startswith("exp4_")]
    if exp4_names:
        exp4_groups: Dict[str, List[Dict[str, object]]] = {
            cast(ExperimentSpec, results_by_condition[n][0]["spec"]).policy_type: results_by_condition[n]
            for n in sorted(exp4_names)
        }
        plot_mean_eval_bands(
            grouped_results=exp4_groups,
            path=directories["figures"] / "exp4_policy_comparison.png",
            title="Exp4 Policy Comparison (4x4 Stochastic)",
        )
        plot_alpha_final_bar(
            grouped_results=exp4_groups,
            path=directories["figures"] / "exp4_policy_final_bar.png",
            title="Exp4 Final Success Rate by Policy",
        )

    # ==================================================================
    # Exp5 — map scale
    # ==================================================================
    for size_str in ("5x5", "6x6"):
        det_key = f"exp5_{size_str}_det"
        sto_key = f"exp5_{size_str}_sto"
        if det_key in results_by_condition and sto_key in results_by_condition:
            plot_mean_eval_bands(
                grouped_results={
                    f"{size_str} deterministic": results_by_condition[det_key],
                    f"{size_str} stochastic":    results_by_condition[sto_key],
                },
                path=directories["figures"] / f"exp5_{size_str}_eval_bands.png",
                title=f"Exp5 {size_str} Deterministic vs Stochastic",
            )
        # Q-table visualisation for stochastic variant (single seed)
        if sto_key in results_by_condition:
            sto_run = next(
                (r for r in results_by_condition[sto_key] if r["seed"] == selected_seed),
                results_by_condition[sto_key][0],
            )
            layout = list(cast(ExperimentSpec, sto_run["spec"]).map_layout or [])
            plot_qtable_heatmap(
                q_values=cast(np.ndarray, sto_run["q_table"]),
                path=directories["figures"] / f"exp5_{size_str}_qtable_heatmap.png",
                title=f"Exp5 {size_str} Max-Q Heatmap",
                map_layout=layout,
            )
            plot_policy_arrows(
                q_values=cast(np.ndarray, sto_run["q_table"]),
                path=directories["figures"] / f"exp5_{size_str}_policy_arrows.png",
                title=f"Exp5 {size_str} Greedy Policy",
                map_layout=layout,
            )

    # ==================================================================
    # Exp6 — random map pool
    # ==================================================================
    for size_str in ("5x5", "6x6"):
        exp6_keys = sorted(n for n in results_by_condition if n.startswith(f"exp6_{size_str}_"))
        if not exp6_keys:
            continue
        # Bar chart: final success rate per map
        exp6_groups: Dict[str, List[Dict[str, object]]] = {
            f"map{cast(ExperimentSpec, results_by_condition[k][0]['spec']).map_id}": results_by_condition[k]
            for k in exp6_keys
        }
        plot_alpha_final_bar(
            grouped_results=exp6_groups,
            path=directories["figures"] / f"exp6_{size_str}_map_variance_bar.png",
            title=f"Exp6 {size_str} Final Success Rate per Map",
        )
        # Map layout visualisation
        pool_layouts = [
            list(cast(ExperimentSpec, results_by_condition[k][0]["spec"]).map_layout or [])
            for k in exp6_keys
        ]
        plot_map_heatmap_grid(
            map_layouts=pool_layouts,
            path=directories["figures"] / f"exp6_{size_str}_map_grid.png",
            title=f"Exp6 {size_str} Random Map Pool",
        )

    # ==================================================================
    # Exp7 — algorithm comparison
    # ==================================================================
    for size_str in ("4x4", "6x6"):
        std_key    = f"exp7_{size_str}_std"
        double_key = f"exp7_{size_str}_double"
        if std_key in results_by_condition and double_key in results_by_condition:
            plot_mean_eval_bands(
                grouped_results={
                    "Standard Q-Learning": results_by_condition[std_key],
                    "Double Q-Learning":   results_by_condition[double_key],
                },
                path=directories["figures"] / f"exp7_{size_str}_algo_comparison.png",
                title=f"Exp7 {size_str} Standard vs Double Q-Learning",
            )

    # ==================================================================
    # Exp8 — reward shaping
    # ==================================================================
    shaping_labels = {
        "base": "Sparse reward",
        "step": "Step penalty",
        "pot":  "Potential shaping",
        "both": "Step + Potential",
    }
    for size_str in ("4x4", "6x6"):
        exp8_keys = {
            suffix: f"exp8_{size_str}_{suffix}"
            for suffix in ("base", "step", "pot", "both")
        }
        present = {lbl: k for suffix, k in exp8_keys.items()
                   if (lbl := shaping_labels[suffix]) and k in results_by_condition}
        if len(present) >= 2:
            plot_mean_eval_bands(
                grouped_results={lbl: results_by_condition[k] for lbl, k in present.items()},
                path=directories["figures"] / f"exp8_{size_str}_shaping_comparison.png",
                title=f"Exp8 {size_str} Reward Shaping Comparison",
            )

    print(f"Completed experiments: {', '.join(results_by_condition.keys())}")
    print(f"Backend requested: {args.backend}")
    print(f"Results saved to: {directories['root'].resolve()}")


if __name__ == "__main__":
    main()
