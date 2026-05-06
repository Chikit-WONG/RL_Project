"""Gymnasium-based slippery 8x8 FrozenLake benchmark.

Uses Gymnasium's predefined 8x8 map and generate_random_map for random maps.
The learning code reuses the project's tabular agents while the map source is
strictly Gymnasium.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

import enhanced_frozenlake as base


DISPLAY = {
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
    "sarsa_lambda": "SARSA(lambda)",
    "q_lambda": "Q(lambda)",
    "optimistic_q": "Optimistic Q",
    "dyna_q": "Dyna-Q",
}

PALETTE = {
    "Q-learning": "#4C78A8",
    "SARSA": "#F58518",
    "Expected SARSA": "#54A24B",
    "SARSA(lambda)": "#B279A2",
    "Q(lambda)": "#E45756",
    "Optimistic Q": "#72B7B2",
    "Dyna-Q": "#FF9DA6",
    "Value Iteration oracle": "#111111",
}


class ExplicitMapFrozenLake(base.NativeFrozenLakeEnv):
    """Project-native environment using an explicit Gymnasium map layout."""

    def __init__(self, layout: Sequence[str], slippery: bool, seed: int) -> None:
        self.map_layout = list(layout)
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


def parse_csv(text: str, cast=str) -> list:
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def gym_standard_8x8() -> list[str]:
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    desc = env.unwrapped.desc.astype(str).tolist()
    env.close()
    return ["".join(row) for row in desc]


def evaluate(agent, layout: Sequence[str], cfg: base.ExperimentConfig, seed: int) -> float:
    successes = 0
    env = ExplicitMapFrozenLake(layout, True, seed)
    rng = np.random.default_rng(seed)
    for episode in range(cfg.eval_episodes):
        state = env.reset(seed + episode)
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


def value_iteration_oracle(layout: Sequence[str], cfg: base.ExperimentConfig, seed: int) -> tuple[object, float]:
    env = ExplicitMapFrozenLake(layout, True, seed)
    oracle = base.ValueIterationAgent(env.n_states, env.n_actions, cfg, seed, env)
    score = evaluate(oracle, layout, cfg, seed + 900_000)
    return oracle, score


def run_one_curve(
    algorithm: str,
    layout: Sequence[str],
    map_label: str,
    group: str,
    map_id: int,
    cfg: base.ExperimentConfig,
    seed: int,
) -> list[dict[str, object]]:
    env = ExplicitMapFrozenLake(layout, True, seed)
    agent = base.AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed)
    rows = []
    eval_seed_base = 800_000 + map_id * 10_000 + seed
    for episode in range(0, cfg.episodes + 1):
        if episode == 0 or episode % cfg.eval_interval == 0 or episode == cfg.episodes:
            score = evaluate(agent, layout, cfg, eval_seed_base + episode)
            rows.append(
                {
                    "group": group,
                    "map_label": map_label,
                    "map_id": map_id,
                    "algorithm": algorithm,
                    "display_algorithm": DISPLAY.get(algorithm, algorithm),
                    "seed": seed,
                    "episode": episode,
                    "success_rate": score,
                }
            )
        if episode < cfg.episodes:
            agent.train_episode(env)
    return rows


def save_map_curve(curves: pd.DataFrame, oracle: float, layout: Sequence[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for name, data in curves.groupby("display_algorithm", sort=False):
        data = data.sort_values("episode")
        ax.plot(
            data["episode"],
            data["success_rate"],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=name,
            color=PALETTE.get(name),
        )
    ax.axhline(oracle, color="black", linestyle="--", linewidth=2.0, label="Value Iteration oracle")
    ax.set_title(f"Gymnasium 8x8 Slippery {curves['map_label'].iloc[0]}")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    ax.text(
        1.02,
        0.02,
        "\n".join(layout),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        family="monospace",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def annotate_heatmap(ax, data: pd.DataFrame, fmt: str = ".2f") -> None:
    for y, idx in enumerate(data.index):
        for x, col in enumerate(data.columns):
            value = data.loc[idx, col]
            if pd.isna(value):
                continue
            ax.text(x + 0.5, y + 0.5, format(value, fmt), ha="center", va="center", color="white", fontsize=8)


def save_summary_figures(curves: pd.DataFrame, final_df: pd.DataFrame, map_df: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pivot = final_df.pivot(index="display_algorithm", columns="map_label", values="final_success")
    fig, ax = plt.subplots(figsize=(max(10, pivot.shape[1] * 0.8), 5.8))
    sns.heatmap(pivot, cmap="viridis", vmin=0, vmax=1, cbar_kws={"label": "Final success"}, ax=ax)
    annotate_heatmap(ax, pivot)
    ax.set_title("Gymnasium 8x8 Slippery Final Success")
    ax.set_xlabel("Map")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(fig_dir / "gym8_final_success_heatmap.png", dpi=220)
    plt.close(fig)

    oracle_lookup = map_df.set_index("map_label")["oracle_success"]
    gap = pivot.copy()
    for col in gap.columns:
        gap[col] = oracle_lookup[col] - gap[col]
    fig, ax = plt.subplots(figsize=(max(10, gap.shape[1] * 0.8), 5.8))
    sns.heatmap(gap, cmap="magma_r", vmin=0, vmax=1, cbar_kws={"label": "Oracle - learned success"}, ax=ax)
    annotate_heatmap(ax, gap)
    ax.set_title("Gymnasium 8x8 Slippery Oracle Gap")
    ax.set_xlabel("Map")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(fig_dir / "gym8_oracle_gap_heatmap.png", dpi=220)
    plt.close(fig)

    mean_curve = curves.groupby(["group", "display_algorithm", "episode"], as_index=False)["success_rate"].mean()
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    for name, data in mean_curve.groupby("display_algorithm", sort=False):
        data = data.sort_values("episode")
        ax.plot(data["episode"], data["success_rate"], marker="o", linewidth=1.8, markersize=3.5, label=name, color=PALETTE.get(name))
    ax.set_title("Gymnasium Random 8x8 Mean Learning Curves")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean greedy success")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gym8_mean_learning_curve.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.stripplot(data=final_df, x="group", y="final_success", hue="display_algorithm", dodge=True, ax=ax)
    ax.set_title("Gymnasium 8x8 Final Success by Map Group")
    ax.set_xlabel("")
    ax.set_ylabel("Final success")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gym8_final_success_points_by_group.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("gymnasium8_random_maps_v1"))
    parser.add_argument("--probs", default="0.8,0.7,0.6,0.5")
    parser.add_argument("--maps-per-prob", type=int, default=3)
    parser.add_argument("--algorithms", default="q_learning,sarsa,expected_sarsa,sarsa_lambda,q_lambda,optimistic_q,dyna_q")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--oracle-eval-episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--planning-steps", type=int, default=15)
    parser.add_argument("--seed-base", type=int, default=31_000)
    args = parser.parse_args()

    out = args.output_dir
    table_dir = out / "tables"
    per_map_dir = out / "figures" / "per_map"
    table_dir.mkdir(parents=True, exist_ok=True)
    per_map_dir.mkdir(parents=True, exist_ok=True)

    cfg = base.ExperimentConfig(
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )
    cfg.planning_steps = args.planning_steps

    oracle_cfg = base.ExperimentConfig(
        episodes=0,
        eval_interval=1,
        eval_episodes=args.oracle_eval_episodes,
        max_steps=args.max_steps,
    )

    maps: list[dict[str, object]] = []
    standard = gym_standard_8x8()
    _, standard_oracle = value_iteration_oracle(standard, oracle_cfg, args.seed_base)
    maps.append(
        {
            "group": "standard8",
            "map_label": "standard8",
            "map_id": 0,
            "map_seed": args.seed_base,
            "p": np.nan,
            "layout": standard,
            "holes": sum(row.count("H") for row in standard),
            "oracle_success": standard_oracle,
        }
    )

    map_counter = 1
    for prob in parse_csv(args.probs, float):
        for map_id in range(args.maps_per_prob):
            seed = args.seed_base + int(prob * 1000) * 10 + map_id
            layout = generate_random_map(size=8, p=prob, seed=seed)
            _, oracle = value_iteration_oracle(layout, oracle_cfg, seed)
            maps.append(
                {
                    "group": f"p{prob:.2f}",
                    "map_label": f"p{prob:.2f}_m{map_id}",
                    "map_id": map_counter,
                    "map_seed": seed,
                    "p": prob,
                    "layout": layout,
                    "holes": sum(row.count("H") for row in layout),
                    "oracle_success": oracle,
                }
            )
            map_counter += 1

    manifest = pd.DataFrame(
        [
            {**m, "layout": "/".join(m["layout"])}
            for m in maps
        ]
    )
    manifest.to_csv(table_dir / "gymnasium8_map_manifest.csv", index=False)

    algorithms = parse_csv(args.algorithms)
    partial_path = table_dir / "gymnasium8_learning_curves_partial.csv"
    rows: list[dict[str, object]] = []
    completed: set[tuple[str, str, int]] = set()
    if partial_path.exists():
        existing = pd.read_csv(partial_path)
        if not existing.empty:
            rows = existing.to_dict("records")
            done = existing[existing["episode"] >= args.episodes]
            completed = set(zip(done["map_label"], done["algorithm"], done["seed"]))
            print(f"resume: loaded {len(existing)} rows and {len(completed)} completed curves")

    for m in maps:
        layout = m["layout"]
        for algorithm in algorithms:
            key = (str(m["map_label"]), algorithm, 0)
            if key in completed:
                print("skip", key)
                continue
            print("run", m["map_label"], algorithm, "holes", m["holes"], "oracle", m["oracle_success"])
            curve_rows = run_one_curve(
                algorithm,
                layout,
                str(m["map_label"]),
                str(m["group"]),
                int(m["map_id"]),
                cfg,
                0,
            )
            rows.extend(curve_rows)
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    curves = pd.DataFrame(rows)
    curves.to_csv(table_dir / "gymnasium8_learning_curves.csv", index=False)

    final_df = curves.sort_values("episode").groupby(["group", "map_label", "algorithm", "display_algorithm"], as_index=False).tail(1)
    final_df = final_df.rename(columns={"success_rate": "final_success"})
    final_df.to_csv(table_dir / "gymnasium8_final_success.csv", index=False)

    ranking = (
        final_df.groupby(["algorithm", "display_algorithm"], as_index=False)
        .agg(mean_final_success=("final_success", "mean"), std_final_success=("final_success", "std"), n_maps=("map_label", "nunique"))
        .sort_values("mean_final_success", ascending=False)
    )
    ranking.to_csv(table_dir / "gymnasium8_algorithm_ranking.csv", index=False)

    map_summary = (
        final_df.groupby(["group", "map_label"], as_index=False)
        .agg(mean_final_success=("final_success", "mean"), best_final_success=("final_success", "max"), worst_final_success=("final_success", "min"))
    )
    map_df = pd.DataFrame([{**m, "layout": "/".join(m["layout"])} for m in maps])
    map_summary = map_summary.merge(map_df[["group", "map_label", "p", "holes", "oracle_success", "layout"]], on=["group", "map_label"], how="left")
    map_summary.to_csv(table_dir / "gymnasium8_map_summary.csv", index=False)

    for m in maps:
        map_curves = curves[curves["map_label"] == m["map_label"]]
        save_map_curve(map_curves, float(m["oracle_success"]), m["layout"], per_map_dir / f"{m['map_label']}_learning_curve.png")

    save_summary_figures(curves, final_df, map_df, out)
    print("saved", out.resolve())


if __name__ == "__main__":
    main()
