"""Create a focused plot for the fixed standard 4x4 slippery benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ALGORITHMS = [
    "q_learning",
    "sarsa",
    "expected_sarsa",
    "sarsa_lambda",
    "q_lambda",
    "optimistic_q",
    "dyna_q",
]

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
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(args.curves)
    curves = curves[(curves["env"] == "4x4_stoch") & (curves["algorithm"].isin(ALGORITHMS))].copy()
    curves["display_algorithm"] = curves["algorithm"].map(DISPLAY)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    sns.lineplot(
        data=curves,
        x="episode",
        y="success_rate",
        hue="display_algorithm",
        errorbar="sd",
        hue_order=[DISPLAY[a] for a in ALGORITHMS],
        palette=PALETTE,
        ax=ax,
    )
    ax.set_title("Standard 4x4 FrozenLake, Slippery: Learning Curves")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "standard_4x4_slippery_learning_curves.png", dpi=220)
    plt.close(fig)

    agg = pd.read_csv(args.aggregate)
    agg = agg[(agg["env"] == "4x4_stoch") & (agg["algorithm"].isin(ALGORITHMS))].copy()
    agg["display_algorithm"] = agg["algorithm"].map(DISPLAY)
    agg = agg.sort_values("mean_final_success", ascending=False)
    agg[
        [
            "display_algorithm",
            "algorithm",
            "mean_final_success",
            "std_final_success",
            "mean_best_success",
            "mean_eval_auc",
        ]
    ].to_csv(args.output_dir / "standard_4x4_slippery_algorithm_ranking.csv", index=False)


if __name__ == "__main__":
    main()
