"""Fast focused rerun for the standard 4x4 slippery FrozenLake setting."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import enhanced_frozenlake as base


ALGORITHMS = [
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "sarsa_lambda",
    "q_lambda",
    "optimistic_q",
    "dyna_q",
    "value_iteration",
    "policy_iteration",
]

DISPLAY = {
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
    "sarsa_lambda": "SARSA(lambda)",
    "q_lambda": "Q(lambda)",
    "optimistic_q": "Optimistic Q",
    "dyna_q": "Dyna-Q",
    "value_iteration": "Value Iteration",
    "policy_iteration": "Policy Iteration",
}


def evaluate_agent(agent, cfg: base.ExperimentConfig, seed: int) -> float:
    env = base.NativeFrozenLakeEnv(4, True, seed, 0.85)
    successes = 0
    for ep in range(cfg.eval_episodes):
        state = env.reset(seed + ep)
        for _ in range(cfg.max_steps):
            action = agent.act(state, greedy=True)
            state, reward, done, _ = env.step(action)
            if done:
                successes += int(reward > 0)
                break
    return successes / cfg.eval_episodes


def run(args: argparse.Namespace) -> None:
    out = args.output_dir
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    cfg = base.ExperimentConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        output_dir=out,
    )
    cfg.planning_steps = args.planning_steps
    cfg.optimistic_init = args.optimistic_init

    eval_rows = []
    final_rows = []
    algorithms = [item.strip() for item in args.algorithms.split(",") if item.strip()]
    for algorithm in algorithms:
        for seed in range(args.seeds):
            if algorithm in base.PLANNING_AGENTS:
                env = base.NativeFrozenLakeEnv(4, True, seed, 0.85)
                agent = base.PLANNING_AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed, env)
                score = evaluate_agent(agent, cfg, seed + 100_000)
                eval_rows.append(
                    {
                        "episode": 0,
                        "algorithm": algorithm,
                        "display_algorithm": DISPLAY.get(algorithm, algorithm),
                        "env": "standard_4x4_stoch",
                        "seed": seed,
                        "success_rate": score,
                    }
                )
                final_rows.append(
                    {
                        "algorithm": algorithm,
                        "display_algorithm": DISPLAY.get(algorithm, algorithm),
                        "env": "standard_4x4_stoch",
                        "seed": seed,
                        "final_success": score,
                    }
                )
                continue

            env = base.NativeFrozenLakeEnv(4, True, seed, 0.85)
            agent = base.AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed)
            final_score = 0.0
            for episode in range(1, args.episodes + 1):
                agent.train_episode(env)
                if episode % args.eval_interval != 0 and episode != args.episodes:
                    continue
                final_score = evaluate_agent(agent, cfg, seed + 100_000 + episode)
                eval_rows.append(
                    {
                        "episode": episode,
                        "algorithm": algorithm,
                        "display_algorithm": DISPLAY.get(algorithm, algorithm),
                        "env": "standard_4x4_stoch",
                        "seed": seed,
                        "success_rate": final_score,
                    }
                )
            final_rows.append(
                {
                    "algorithm": algorithm,
                    "display_algorithm": DISPLAY.get(algorithm, algorithm),
                    "env": "standard_4x4_stoch",
                    "seed": seed,
                    "final_success": final_score,
                }
            )
            print("standard4x4", algorithm, seed, final_score)

    eval_df = pd.DataFrame(eval_rows)
    final_df = pd.DataFrame(final_rows)
    aggregate = (
        final_df.groupby(["algorithm", "display_algorithm", "env"], as_index=False)
        .agg(
            mean_final_success=("final_success", "mean"),
            std_final_success=("final_success", "std"),
            n=("final_success", "count"),
        )
        .sort_values("mean_final_success", ascending=False)
    )
    eval_df.to_csv(out / "tables" / "standard_4x4_eval_curves.csv", index=False)
    final_df.to_csv(out / "tables" / "standard_4x4_run_summary.csv", index=False)
    aggregate.to_csv(out / "tables" / "standard_4x4_aggregate_summary.csv", index=False)

    learn_only = eval_df[~eval_df["algorithm"].isin(base.PLANNING_AGENTS)]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.lineplot(
        data=learn_only,
        x="episode",
        y="success_rate",
        hue="display_algorithm",
        errorbar="sd",
        ax=ax,
    )
    oracle = aggregate[aggregate["algorithm"].isin(["value_iteration", "policy_iteration"])]["mean_final_success"]
    if not oracle.empty:
        ax.axhline(float(oracle.max()), color="black", linestyle="--", linewidth=2.0, label="Planning oracle")
    ax.set_title("Standard 4x4 FrozenLake-v1, Slippery Only")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "figures" / "standard_4x4_slippery_only_success.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("standard_4x4_slippery_focused_v1"))
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=150)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--planning-steps", type=int, default=5)
    parser.add_argument("--optimistic-init", type=float, default=1.0)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
