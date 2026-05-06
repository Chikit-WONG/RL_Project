"""Run per-map learning curves for slippery random FrozenLake maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import enhanced_frozenlake as base
from map_env_utils import CustomMapFrozenLakeEnv, evaluate_on_env, make_random_map, parse_layout


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


def parse_csv(text: str, cast=str) -> list:
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def make_cfg(args: argparse.Namespace) -> base.ExperimentConfig:
    cfg = base.ExperimentConfig(
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
    )
    cfg.planning_steps = args.planning_steps
    cfg.optimistic_init = args.optimistic_init
    cfg.epsilon_decay = args.epsilon_decay
    cfg.epsilon_min = args.epsilon_min
    return cfg


def train_agent(algorithm: str, layout: Sequence[str], seed: int, cfg: base.ExperimentConfig):
    env = CustomMapFrozenLakeEnv(layout, True, seed)
    agent = base.AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed)
    return env, agent


def run_one_curve(
    algorithm: str,
    layout: Sequence[str],
    map_label: str,
    frozen_prob: float,
    map_id: int,
    seed: int,
    cfg: base.ExperimentConfig,
) -> list[dict[str, object]]:
    train_env, agent = train_agent(algorithm, layout, seed, cfg)
    rows = []
    eval_seed_base = 700_000 + int(frozen_prob * 1000) * 100 + map_id * 10 + seed

    for episode in range(0, cfg.episodes + 1):
        if episode == 0 or episode % cfg.eval_interval == 0 or episode == cfg.episodes:
            score = evaluate_on_env(
                agent,
                lambda s, layout=layout: CustomMapFrozenLakeEnv(layout, True, s),
                cfg,
                eval_seed_base + episode,
            )
            rows.append(
                {
                    "algorithm": algorithm,
                    "display_algorithm": DISPLAY.get(algorithm, algorithm),
                    "frozen_prob": frozen_prob,
                    "map_id": map_id,
                    "map_label": map_label,
                    "seed": seed,
                    "episode": episode,
                    "success_rate": score,
                }
            )
        if episode < cfg.episodes:
            agent.train_episode(train_env)
    return rows


def load_designed4_maps(path: Path) -> list[dict[str, object]]:
    manifest = pd.read_csv(path)
    maps = []
    for _, row in manifest.iterrows():
        maps.append(
            {
                "frozen_prob": np.nan,
                "map_id": int(row["map_id"]),
                "map_label": f"designed4_m{int(row['map_id'])}",
                "map_seed": int(row["map_seed"]),
                "oracle_success": float(row["oracle_success"]),
                "layout": parse_layout(str(row["layout"])),
                "holes": int(row["holes"]),
            }
        )
    return maps


def build_random8_maps(args: argparse.Namespace, oracle_cfg: base.ExperimentConfig) -> list[dict[str, object]]:
    maps = []
    for prob in parse_csv(args.probs, float):
        for map_id in range(args.maps_per_prob):
            map_seed = 10_000 + 8 * 1000 + int(prob * 100) * 10 + map_id
            layout = make_random_map(8, prob, map_seed)
            map_label = f"p{prob:.2f}_m{map_id}"
            oracle_env = CustomMapFrozenLakeEnv(layout, True, map_seed)
            oracle = base.ValueIterationAgent(oracle_env.n_states, oracle_env.n_actions, oracle_cfg, map_seed, oracle_env)
            oracle_score = evaluate_on_env(
                oracle,
                lambda s, layout=layout: CustomMapFrozenLakeEnv(layout, True, s),
                oracle_cfg,
                map_seed + 900_000,
            )
            maps.append(
                {
                    "frozen_prob": prob,
                    "map_id": map_id,
                    "map_label": map_label,
                    "map_seed": map_seed,
                    "oracle_success": oracle_score,
                    "layout": layout,
                    "holes": sum(row.count("H") for row in layout),
                }
            )
    return maps


def save_map_plot(curves: pd.DataFrame, oracle_score: float, layout: Sequence[str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for name, data in curves.groupby("display_algorithm", sort=False):
        data = data.sort_values("episode")
        ax.plot(
            data["episode"],
            data["success_rate"],
            marker="o",
            linewidth=1.8,
            markersize=3.8,
            label=name,
            color=PALETTE.get(name),
        )
    ax.axhline(oracle_score, color=PALETTE["Value Iteration oracle"], linestyle="--", linewidth=2.0, label="Value Iteration oracle")
    title = f"{curves['map_size'].iloc[0]} Slippery Map {curves['map_label'].iloc[0]} Learning Curves"
    ax.set_title(title)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    map_text = "\n".join(layout)
    ax.text(
        1.02,
        0.02,
        map_text,
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


def save_panel_plot(curves: pd.DataFrame, oracles: pd.DataFrame, out: Path) -> None:
    sort_cols = ["map_id"] if oracles["frozen_prob"].isna().all() else ["frozen_prob", "map_id"]
    labels = list(oracles.sort_values(sort_cols)["map_label"])
    n = len(labels)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.6 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).ravel()
    for ax, label in zip(axes, labels):
        data_map = curves[curves["map_label"] == label]
        for name, data in data_map.groupby("display_algorithm", sort=False):
            data = data.sort_values("episode")
            ax.plot(data["episode"], data["success_rate"], linewidth=1.1, color=PALETTE.get(name), label=name)
        oracle = float(oracles.loc[oracles["map_label"] == label, "oracle_success"].iloc[0])
        ax.axhline(oracle, color="#111111", linestyle="--", linewidth=1.2)
        ax.set_title(label, fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(-0.02, 1.05)
    for ax in axes[len(labels):]:
        ax.axis("off")
    for ax in axes[-n_cols:]:
        ax.set_xlabel("Episode")
    for ax in axes[::n_cols]:
        ax.set_ylabel("Success")
    handles, labels_legend = axes[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.2))
    labels_legend.append("Value Iteration oracle")
    fig.legend(handles, labels_legend, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, frameon=False)
    fig.suptitle(f"{curves['map_size'].iloc[0]} Slippery Maps: Per-Map Learning Curves", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.92, 0.985))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def save_probability_panels(curves: pd.DataFrame, oracles: pd.DataFrame, out_dir: Path) -> None:
    if oracles["frozen_prob"].isna().all():
        return
    for prob in sorted(curves["frozen_prob"].unique()):
        labels = list(oracles[oracles["frozen_prob"] == prob].sort_values("map_id")["map_label"])
        fig, axes = plt.subplots(1, len(labels), figsize=(20.0, 4.4), sharex=True, sharey=True)
        if len(labels) == 1:
            axes = [axes]
        for ax, label in zip(axes, labels):
            data_map = curves[curves["map_label"] == label]
            for name, data in data_map.groupby("display_algorithm", sort=False):
                data = data.sort_values("episode")
                ax.plot(data["episode"], data["success_rate"], marker="o", markersize=2.2, linewidth=1.2, color=PALETTE.get(name), label=name)
            oracle = float(oracles.loc[oracles["map_label"] == label, "oracle_success"].iloc[0])
            ax.axhline(oracle, color="#111111", linestyle="--", linewidth=1.2)
            ax.set_title(label, fontsize=9)
            ax.grid(True, alpha=0.2)
            ax.set_ylim(-0.02, 1.05)
            ax.set_xlabel("Episode")
        axes[0].set_ylabel("Success")
        handles, labels_legend = axes[0].get_legend_handles_labels()
        handles.append(plt.Line2D([0], [0], color="#111111", linestyle="--", linewidth=1.2))
        labels_legend.append("Value Iteration oracle")
        fig.legend(handles, labels_legend, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, frameon=False)
        fig.suptitle(f"8x8 Slippery Random Maps, p={prob:.2f}: Learning Curves by Map", y=1.02)
        fig.tight_layout(rect=(0, 0, 0.89, 0.96))
        fig.savefig(out_dir / f"random8_learning_curves_p{prob:.2f}.png", dpi=220)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("random_learning_curves_v1"))
    parser.add_argument("--suite", choices=["designed4", "random8"], default="random8")
    parser.add_argument("--algorithms", default="q_learning,sarsa,expected_sarsa,sarsa_lambda,q_lambda,optimistic_q,dyna_q")
    parser.add_argument("--probs", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--maps-per-prob", type=int, default=5)
    parser.add_argument("--designed4-manifest", type=Path, default=Path("designed_map_results_v1/tables/designed_4x4_map_manifest.csv"))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--oracle-eval-episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--planning-steps", type=int, default=15)
    parser.add_argument("--optimistic-init", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    args = parser.parse_args()

    out = args.output_dir
    fig_dir = out / "figures"
    per_map_dir = fig_dir / "per_map"
    table_dir = out / "tables"
    for path in [fig_dir, per_map_dir, table_dir]:
        path.mkdir(parents=True, exist_ok=True)

    algorithms = parse_csv(args.algorithms)
    cfg = make_cfg(args)
    oracle_cfg = base.ExperimentConfig(eval_episodes=args.oracle_eval_episodes, max_steps=args.max_steps)

    partial_path = table_dir / f"{args.suite}_learning_curves_partial.csv"
    if partial_path.exists():
        existing = pd.read_csv(partial_path)
        rows: list[dict[str, object]] = existing.to_dict("records")
        completed = set(
            tuple(row)
            for row in existing[existing["episode"] >= args.episodes][["map_label", "algorithm", "seed"]].drop_duplicates().to_numpy()
        )
        print(f"resume: loaded {len(existing)} rows and {len(completed)} completed curves from {partial_path}")
    else:
        rows = []
        completed = set()
    manifest: list[dict[str, object]] = []

    map_specs = load_designed4_maps(args.designed4_manifest) if args.suite == "designed4" else build_random8_maps(args, oracle_cfg)

    for spec in map_specs:
        prob = spec["frozen_prob"]
        map_id = int(spec["map_id"])
        map_label = str(spec["map_label"])
        layout = list(spec["layout"])
        oracle_score = float(spec["oracle_success"])
        manifest.append(
            {
                "suite": args.suite,
                "map_size": f"{len(layout)}x{len(layout)}",
                "frozen_prob": prob,
                "map_id": map_id,
                "map_label": map_label,
                "map_seed": int(spec["map_seed"]),
                "oracle_success": oracle_score,
                "layout": "/".join(layout),
                "holes": int(spec["holes"]),
            }
        )
        for algorithm in algorithms:
            for seed in range(args.seeds):
                key = (map_label, algorithm, seed)
                if key in completed:
                    print("skip completed", args.suite, map_label, algorithm, seed)
                    continue
                curve_rows = run_one_curve(algorithm, layout, map_label, float(prob) if not pd.isna(prob) else -1.0, map_id, seed, cfg)
                for row in curve_rows:
                    row["suite"] = args.suite
                    row["map_size"] = f"{len(layout)}x{len(layout)}"
                    row["oracle_success"] = oracle_score
                    row["holes"] = int(spec["holes"])
                rows.extend(curve_rows)
                pd.DataFrame(rows).to_csv(partial_path, index=False)
                print("curve", args.suite, map_label, algorithm, seed, curve_rows[-1]["success_rate"], "oracle", oracle_score)

        map_curves = pd.DataFrame([row for row in rows if row["map_label"] == map_label])
        if not map_curves.empty:
            save_map_plot(map_curves, oracle_score, layout, per_map_dir / f"{map_label}_learning_curve.png")

    curves = pd.DataFrame(rows)
    oracles = pd.DataFrame(manifest)
    curves.to_csv(table_dir / f"{args.suite}_learning_curves.csv", index=False)
    oracles.to_csv(table_dir / f"{args.suite}_learning_curve_map_manifest.csv", index=False)

    save_panel_plot(curves, oracles, fig_dir / f"{args.suite}_learning_curves_all_maps_panel.png")
    save_probability_panels(curves, oracles, fig_dir)

    final = curves.sort_values("episode").groupby(["map_label", "frozen_prob", "map_id", "algorithm", "display_algorithm", "seed"], as_index=False).tail(1)
    final.to_csv(table_dir / f"{args.suite}_learning_curve_final_success.csv", index=False)


if __name__ == "__main__":
    main()
