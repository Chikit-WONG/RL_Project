"""Credibility and generalization experiments for FrozenLake results.

This script implements the five requested follow-up fixes:

1. Larger evaluation episodes for key results.
2. More seeds for the 8x8 stochastic benchmark.
3. Random-map generalization for 4x4 and 8x8.
4. Oracle-filtered 16x16 map study.
5. Stronger planning/exploration methods: Dyna-Q+ and Prioritized Sweeping.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import enhanced_frozenlake as base


class CustomMapFrozenLakeEnv(base.NativeFrozenLakeEnv):
    """Native FrozenLake over an explicit map layout."""

    def __init__(self, map_layout: Sequence[str], slippery: bool, seed: int) -> None:
        self.map_layout = list(map_layout)
        self.nrow = len(self.map_layout)
        self.ncol = len(self.map_layout[0])
        self.n_states = self.nrow * self.ncol
        self.n_actions = 4
        self.slippery = slippery
        self.rng = np.random.default_rng(seed)
        self.start_state = self._find("S")
        self.goal_state = self._find("G")
        self.terminal_states = {
            self._to_state(r, c)
            for r, row in enumerate(self.map_layout)
            for c, cell in enumerate(row)
            if cell in {"H", "G"}
        }
        self.state = self.start_state


def make_random_map(size: int, frozen_prob: float, seed: int) -> List[str]:
    """Generate a random map with a carved safe path."""

    rng = np.random.default_rng(seed)
    grid = np.where(rng.random((size, size)) < frozen_prob, "F", "H")
    r = c = 0
    grid[r, c] = "S"
    while r < size - 1 or c < size - 1:
        if r == size - 1:
            c += 1
        elif c == size - 1:
            r += 1
        elif rng.random() < 0.5:
            r += 1
        else:
            c += 1
        grid[r, c] = "F"
    grid[-1, -1] = "G"
    return ["".join(row) for row in grid]


def train_agent_on_env(
    algorithm: str,
    env,
    cfg: base.ExperimentConfig,
    seed: int,
):
    if algorithm in base.PLANNING_AGENTS:
        return base.PLANNING_AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed, env)
    agent = base.AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed)
    for _ in range(cfg.episodes):
        agent.train_episode(env)
    return agent


def evaluate_on_env(agent, env_factory, cfg: base.ExperimentConfig, seed: int) -> float:
    successes = 0
    env = env_factory(seed)
    rng = np.random.default_rng(seed)
    for ep in range(cfg.eval_episodes):
        state = env.reset(seed + ep)
        for _ in range(cfg.max_steps):
            if hasattr(agent, "act"):
                action = agent.act(state, greedy=True)
            else:
                values = agent.q[state]
                best = np.flatnonzero(np.isclose(values, np.max(values)))
                action = int(rng.choice(best))
            state, reward, done, _ = env.step(action)
            if done:
                successes += int(reward > 0)
                break
    return successes / cfg.eval_episodes


def aggregate(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["suite", "algorithm", "env"], as_index=False)
        .agg(
            mean_success=("success", "mean"),
            std_success=("success", "std"),
            n=("success", "count"),
            mean_oracle=("oracle_success", "mean"),
        )
        .sort_values(["suite", "env", "mean_success"], ascending=[True, True, False])
    )


def run_high_precision_8x8(args, out: Path) -> pd.DataFrame:
    rows = []
    algorithms = [item.strip() for item in args.hp_algorithms.split(",") if item.strip()]
    cfg = base.ExperimentConfig(
        episodes=args.hp_episodes,
        eval_episodes=args.hp_eval_episodes,
        max_steps=300,
        eval_interval=args.hp_episodes,
    )
    cfg.planning_steps = 20
    cfg.ps_max_queue = 1_000
    cfg.dyna_plus_kappa = 0.001
    cfg.optimistic_init = 1.0

    for algorithm in algorithms:
        for seed in range(args.hp_seeds):
            train_env = base.NativeFrozenLakeEnv(8, True, seed, 0.85)
            agent = train_agent_on_env(algorithm, train_env, cfg, seed)
            score = base.evaluate(agent, base.EnvSpec(8, True), seed + 500_000, cfg)
            rows.append(
                {
                    "suite": "high_precision_8x8_stoch",
                    "algorithm": algorithm,
                    "env": "8x8_stoch",
                    "map_id": "standard",
                    "seed": seed,
                    "success": score,
                    "oracle_success": np.nan,
                }
            )
            pd.DataFrame(rows).to_csv(out / "high_precision_8x8_runs_partial.csv", index=False)
            print("high_precision", algorithm, seed, score)
    df = pd.DataFrame(rows)
    df.to_csv(out / "high_precision_8x8_runs.csv", index=False)
    return df


def run_random_map_generalization(args, out: Path) -> pd.DataFrame:
    rows = []
    algorithms = [item.strip() for item in args.gen_algorithms.split(",") if item.strip()]
    cfg = base.ExperimentConfig(
        episodes=args.gen_episodes,
        eval_episodes=args.gen_eval_episodes,
        max_steps=200,
        eval_interval=args.gen_episodes,
    )
    cfg.planning_steps = 15
    cfg.dyna_plus_kappa = 0.001
    cfg.optimistic_init = 1.0

    for size in [4, 8]:
        for frozen_prob in [0.80, 0.85, 0.90, 0.95]:
            for map_id in range(args.gen_maps):
                map_seed = 10_000 + size * 1000 + int(frozen_prob * 100) * 10 + map_id
                layout = make_random_map(size, frozen_prob, map_seed)
                oracle_cfg = base.ExperimentConfig(eval_episodes=args.oracle_eval_episodes, max_steps=200)
                oracle_env = CustomMapFrozenLakeEnv(layout, True, map_seed)
                oracle = base.ValueIterationAgent(oracle_env.n_states, oracle_env.n_actions, oracle_cfg, map_seed, oracle_env)
                oracle_score = evaluate_on_env(
                    oracle,
                    lambda s, layout=layout: CustomMapFrozenLakeEnv(layout, True, s),
                    oracle_cfg,
                    map_seed + 900_000,
                )
                for algorithm in algorithms:
                    for seed in range(args.gen_seeds):
                        train_env = CustomMapFrozenLakeEnv(layout, True, seed)
                        agent = train_agent_on_env(algorithm, train_env, cfg, seed)
                        score = evaluate_on_env(
                            agent,
                            lambda s, layout=layout: CustomMapFrozenLakeEnv(layout, True, s),
                            cfg,
                            seed + 600_000,
                        )
                        rows.append(
                            {
                                "suite": "random_map_generalization",
                                "algorithm": algorithm,
                                "env": f"{size}x{size}_stoch_p{frozen_prob:.2f}",
                                "map_id": map_id,
                                "seed": seed,
                                "success": score,
                                "oracle_success": oracle_score,
                            }
                        )
                        pd.DataFrame(rows).to_csv(out / "random_map_generalization_runs_partial.csv", index=False)
                        print("generalization", size, frozen_prob, map_id, algorithm, seed, score, "oracle", oracle_score)
    df = pd.DataFrame(rows)
    df.to_csv(out / "random_map_generalization_runs.csv", index=False)
    return df


def run_oracle_filtered_16x16(args, out: Path) -> pd.DataFrame:
    rows = []
    selected_maps = []
    algorithms = [item.strip() for item in args.of_algorithms.split(",") if item.strip()]
    cfg = base.ExperimentConfig(
        episodes=args.of_episodes,
        eval_episodes=args.of_eval_episodes,
        max_steps=500,
        eval_interval=args.of_episodes,
    )
    cfg.planning_steps = 25
    cfg.dyna_plus_kappa = 0.001
    cfg.optimistic_init = 1.0

    frozen_probs = [float(item.strip()) for item in args.of_probs.split(",") if item.strip()]
    for frozen_prob in frozen_probs:
        for map_id in range(args.of_candidate_maps):
            map_seed = 50_000 + int(frozen_prob * 100) * 100 + map_id
            layout = make_random_map(16, frozen_prob, map_seed)
            oracle_cfg = base.ExperimentConfig(eval_episodes=args.oracle_eval_episodes, max_steps=500)
            oracle_env = CustomMapFrozenLakeEnv(layout, True, map_seed)
            oracle = base.ValueIterationAgent(oracle_env.n_states, oracle_env.n_actions, oracle_cfg, map_seed, oracle_env)
            oracle_score = evaluate_on_env(
                oracle,
                lambda s, layout=layout: CustomMapFrozenLakeEnv(layout, True, s),
                oracle_cfg,
                map_seed + 700_000,
            )
            selected_maps.append(
                {
                    "frozen_prob": frozen_prob,
                    "map_id": map_id,
                    "map_seed": map_seed,
                    "oracle_success": oracle_score,
                    "layout": "|".join(layout),
                }
            )
            if oracle_score < args.of_min_oracle:
                continue
            for algorithm in algorithms:
                for seed in range(args.of_seeds):
                    train_env = CustomMapFrozenLakeEnv(layout, True, seed)
                    agent = train_agent_on_env(algorithm, train_env, cfg, seed)
                    score = evaluate_on_env(
                        agent,
                        lambda s, layout=layout: CustomMapFrozenLakeEnv(layout, True, s),
                        cfg,
                        seed + 800_000,
                    )
                    rows.append(
                        {
                            "suite": "oracle_filtered_16x16",
                            "algorithm": algorithm,
                            "env": f"16x16_stoch_p{frozen_prob:.2f}",
                            "map_id": map_id,
                            "seed": seed,
                            "success": score,
                            "oracle_success": oracle_score,
                        }
                    )
                    pd.DataFrame(rows).to_csv(out / "oracle_filtered_16x16_runs_partial.csv", index=False)
                    print("oracle_filtered", frozen_prob, map_id, algorithm, seed, score, "oracle", oracle_score)
    pd.DataFrame(selected_maps).to_csv(out / "oracle_filtered_16x16_candidates.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "oracle_filtered_16x16_runs.csv", index=False)
    return df


def plot_outputs(all_rows: pd.DataFrame, agg: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=agg, x="suite", y="mean_success", hue="algorithm", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Credibility and Generalization Improvements")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(fig_dir / "credibility_summary_bar.png", dpi=220)
    plt.close(fig)

    gen = all_rows[all_rows["suite"] == "random_map_generalization"].copy()
    if not gen.empty:
        gen["frozen_prob"] = gen["env"].str.extract(r"p(\d+\.\d+)").astype(float)
        gen["map_size"] = gen["env"].str.extract(r"(\d+)x").astype(int)
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.lineplot(data=gen, x="frozen_prob", y="success", hue="algorithm", style="map_size", marker="o", errorbar="sd", ax=ax)
        ax.set_ylim(0, 1.05)
        ax.set_title("Random Map Generalization")
        fig.tight_layout()
        fig.savefig(fig_dir / "random_map_generalization_line.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=gen, x="oracle_success", y="success", hue="algorithm", style="map_size", s=100, ax=ax)
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.set_title("Oracle Success vs Learned Success")
        fig.tight_layout()
        fig.savefig(fig_dir / "oracle_vs_learned_scatter.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("credibility_results_v1"))
    parser.add_argument("--hp-seeds", type=int, default=5)
    parser.add_argument("--hp-episodes", type=int, default=3000)
    parser.add_argument("--hp-eval-episodes", type=int, default=1000)
    parser.add_argument("--hp-algorithms", default="q_learning,optimistic_q,dyna_q,dyna_q_plus,prioritized_sweeping")
    parser.add_argument("--gen-maps", type=int, default=3)
    parser.add_argument("--gen-seeds", type=int, default=2)
    parser.add_argument("--gen-episodes", type=int, default=1200)
    parser.add_argument("--gen-eval-episodes", type=int, default=300)
    parser.add_argument("--gen-algorithms", default="q_learning,optimistic_q,dyna_q_plus,prioritized_sweeping")
    parser.add_argument("--of-candidate-maps", type=int, default=4)
    parser.add_argument("--of-min-oracle", type=float, default=0.72)
    parser.add_argument("--of-seeds", type=int, default=2)
    parser.add_argument("--of-episodes", type=int, default=2000)
    parser.add_argument("--of-eval-episodes", type=int, default=300)
    parser.add_argument("--of-algorithms", default="optimistic_q,dyna_q_plus,prioritized_sweeping")
    parser.add_argument("--of-probs", default="0.90,0.92,0.95")
    parser.add_argument("--oracle-eval-episodes", type=int, default=300)
    parser.add_argument(
        "--suites",
        default="high_precision,generalization,oracle16",
        help="Comma-separated subset: high_precision,generalization,oracle16",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    requested = {item.strip() for item in args.suites.split(",") if item.strip()}
    frames = []
    if "high_precision" in requested:
        frames.append(run_high_precision_8x8(args, out))
    if "generalization" in requested:
        frames.append(run_random_map_generalization(args, out))
    if "oracle16" in requested:
        frames.append(run_oracle_filtered_16x16(args, out))
    if not frames:
        raise ValueError("No suites selected.")
    all_rows = pd.concat([df for df in frames if not df.empty], ignore_index=True)
    agg = aggregate(all_rows.to_dict("records"))
    all_rows.to_csv(out / "all_credibility_runs.csv", index=False)
    agg.to_csv(out / "aggregate_credibility_summary.csv", index=False)
    plot_outputs(all_rows, agg, out)
    (out / "README.md").write_text(
        "\n".join(
            [
                "# Credibility and Generalization Results",
                "",
                "This experiment implements the five requested follow-up fixes:",
                "",
                "1. Larger evaluation episodes for key 8x8 stochastic results.",
                "2. More seeds for 8x8 stochastic top methods.",
                "3. Random map generalization on 4x4 and 8x8.",
                "4. Oracle-filtered 16x16 maps.",
                "5. Dyna-Q+ and Prioritized Sweeping baselines.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved credibility experiments to {out.resolve()}")


if __name__ == "__main__":
    main()
