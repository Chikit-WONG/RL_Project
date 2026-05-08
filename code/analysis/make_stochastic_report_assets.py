"""Build stochastic-only report assets and ablation figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "ablation_study_v1"
FIG = OUT / "figures"
TAB = OUT / "tables"

DISPLAY = {
    "q_learning": "Q-learning",
    "sarsa": "SARSA",
    "expected_sarsa": "Expected SARSA",
    "n_step_sarsa": "n-step SARSA",
    "sarsa_lambda": "SARSA(lambda)",
    "q_lambda": "Q(lambda)",
    "dyna_q": "Dyna-Q",
    "optimistic_q": "Optimistic Q",
    "ucb_q": "UCB-Q",
    "softmax_q": "Softmax Q",
    "value_iteration": "Value Iteration",
    "policy_iteration": "Policy Iteration",
}

CORE = ["q_learning", "sarsa", "expected_sarsa", "sarsa_lambda", "q_lambda", "optimistic_q", "dyna_q"]
EXPLORATION = ["q_learning", "optimistic_q", "ucb_q", "softmax_q"]
PROPAGATION = ["q_learning", "sarsa", "expected_sarsa", "n_step_sarsa", "sarsa_lambda", "q_lambda", "dyna_q"]

PALETTE = {
    "Q-learning": "#4C78A8",
    "SARSA": "#F58518",
    "Expected SARSA": "#54A24B",
    "SARSA(lambda)": "#B279A2",
    "Q(lambda)": "#E45756",
    "Optimistic Q": "#72B7B2",
    "Dyna-Q": "#FF9DA6",
}


def ensure_dirs() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)


def load_fixed() -> pd.DataFrame:
    path = ROOT.parent / "RL_Project_Enhanced" / "enhanced_results_final_v1" / "tables" / "aggregate_summary_final.csv"
    if not path.exists():
        path = Path("E:/000_hmw/RL/proj2/RL_Project_Enhanced/enhanced_results_final_v1/tables/aggregate_summary_final.csv")
    df = pd.read_csv(path)
    df["display_algorithm"] = df["algorithm"].map(DISPLAY).fillna(df["algorithm"])
    return df


def load_standard4() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "standard_4x4_slippery_focused_v2" / "tables" / "standard_4x4_aggregate_summary.csv")
    out = df.rename(columns={"mean_final_success": "mean_success", "std_final_success": "std_success"}).copy()
    out["setting"] = "Standard 4x4"
    out["algorithm"] = out["algorithm"].astype(str)
    out["display_algorithm"] = out["display_algorithm"].fillna(out["algorithm"].map(DISPLAY))
    return out[["setting", "algorithm", "display_algorithm", "mean_success", "std_success", "n"]]


def load_designed4() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "designed_map_results_v1" / "tables" / "designed_4x4_aggregate_summary.csv")
    out = df.copy()
    out["setting"] = "Designed 4x4"
    return out[["setting", "algorithm", "display_algorithm", "mean_success", "std_success", "n", "mean_oracle"]]


def load_random8() -> pd.DataFrame:
    runs = pd.read_csv(ROOT / "focused_generalization_v2" / "all_credibility_runs.csv")
    data = runs[(runs["env"].str.startswith("8x8")) & (runs["algorithm"].isin(CORE))].copy()
    data["display_algorithm"] = data["algorithm"].map(DISPLAY)
    agg = (
        data.groupby(["algorithm", "display_algorithm"], as_index=False)
        .agg(
            mean_success=("success", "mean"),
            std_success=("success", "std"),
            n=("success", "count"),
            mean_oracle=("oracle_success", "mean"),
        )
        .sort_values("mean_success", ascending=False)
    )
    agg["setting"] = "Random 8x8"
    return agg[["setting", "algorithm", "display_algorithm", "mean_success", "std_success", "n", "mean_oracle"]]


def fixed_setting(df: pd.DataFrame, env: str, setting: str, algorithms: list[str]) -> pd.DataFrame:
    data = df[(df["env"] == env) & (df["algorithm"].isin(algorithms))].copy()
    data["setting"] = setting
    data = data.rename(columns={"mean_final_success": "mean_success", "std_final_success": "std_success"})
    data["n"] = data["n_seeds"]
    return data[["setting", "algorithm", "display_algorithm", "mean_success", "std_success", "n"]]


def save_bar(data: pd.DataFrame, path: Path, title: str, x: str = "display_algorithm", hue: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(max(8.0, len(data[x].unique()) * 0.8), 5.2))
    sns.barplot(data=data, x=x, y="mean_success", hue=hue, ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Mean greedy success rate")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_annotated_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """Draw heatmaps with explicit per-cell text to avoid missing annotations."""
    values = matrix.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    fig_w = max(8.6, 1.7 * n_cols + 3.2)
    fig_h = max(5.0, 0.62 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(list(matrix.columns), rotation=0, ha="center")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(list(matrix.index))
    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = vmin + 0.55 * (vmax - vmin)
    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]
            if np.isnan(val):
                label = "NA"
                color = "#222222"
            else:
                label = f"{val:.2f}"
                color = "white" if val < threshold else "#222222"
            ax.text(j, i, label, ha="center", va="center", fontsize=11, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def load_random8_runs() -> pd.DataFrame:
    random_runs = pd.read_csv(ROOT / "focused_generalization_v2" / "all_credibility_runs.csv")
    data = random_runs[(random_runs["env"].str.startswith("8x8")) & (random_runs["algorithm"].isin(CORE))].copy()
    data["display_algorithm"] = data["algorithm"].map(DISPLAY)
    data["frozen_prob"] = data["env"].str.extract(r"p(\d+\.\d+)").astype(float)
    data["map_key"] = data["env"].str.replace("8x8_stoch_", "p", regex=False) + "_m" + data["map_id"].astype(str)
    data["oracle_gap"] = data["oracle_success"] - data["success"]
    return data


def save_random8_per_map_lines(random8_runs: pd.DataFrame) -> None:
    summary = (
        random8_runs.groupby(["frozen_prob", "map_id", "map_key", "display_algorithm"], as_index=False)
        .agg(success=("success", "mean"), oracle_success=("oracle_success", "mean"))
        .sort_values(["frozen_prob", "map_id", "display_algorithm"])
    )
    map_order = (
        summary[["frozen_prob", "map_id", "map_key", "oracle_success"]]
        .drop_duplicates()
        .sort_values(["frozen_prob", "map_id"])
        .reset_index(drop=True)
    )
    map_order["x"] = np.arange(len(map_order))
    summary = summary.merge(map_order[["map_key", "x"]], on="map_key", how="left")
    summary.to_csv(TAB / "random8_per_map_algorithm_success.csv", index=False)

    fig, ax = plt.subplots(figsize=(14.0, 6.2))
    for name in [DISPLAY[a] for a in CORE]:
        data = summary[summary["display_algorithm"] == name]
        ax.plot(
            data["x"],
            data["success"],
            marker="o",
            linewidth=1.7,
            markersize=4.2,
            label=name,
            color=PALETTE.get(name),
        )
    ax.plot(
        map_order["x"],
        map_order["oracle_success"],
        color="black",
        linestyle="--",
        marker="D",
        markersize=4.0,
        linewidth=2.0,
        label="Value Iteration oracle",
    )
    for boundary in [4.5, 9.5, 14.5]:
        ax.axvline(boundary, color="#cccccc", linewidth=1.0)
    ax.set_xticks(map_order["x"])
    ax.set_xticklabels(map_order["map_key"], rotation=45, ha="right")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("8x8 Slippery Random Maps: Algorithm Comparison on Each Map")
    ax.set_xlabel("Individual random maps (5 maps per frozen-tile probability)")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "random8_per_map_algorithm_lines.png", dpi=240)
    plt.close(fig)


def save_random8_faceted_map_lines(random8_runs: pd.DataFrame) -> None:
    summary = (
        random8_runs.groupby(["frozen_prob", "map_id", "display_algorithm"], as_index=False)
        .agg(success=("success", "mean"), oracle_success=("oracle_success", "mean"))
        .sort_values(["frozen_prob", "map_id"])
    )
    probs = sorted(summary["frozen_prob"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.0), sharey=True)
    axes = axes.ravel()
    for ax, prob in zip(axes, probs):
        data_p = summary[summary["frozen_prob"] == prob]
        for name in [DISPLAY[a] for a in CORE]:
            data = data_p[data_p["display_algorithm"] == name]
            ax.plot(
                data["map_id"],
                data["success"],
                marker="o",
                linewidth=1.7,
                markersize=4.0,
                label=name,
                color=PALETTE.get(name),
            )
        oracle = data_p[["map_id", "oracle_success"]].drop_duplicates().sort_values("map_id")
        ax.plot(
            oracle["map_id"],
            oracle["oracle_success"],
            color="black",
            linestyle="--",
            marker="D",
            linewidth=1.8,
            markersize=3.8,
            label="Value Iteration oracle",
        )
        ax.set_title(f"p = {prob:.2f}")
        ax.set_xlabel("Map id")
        ax.set_xticks(sorted(data_p["map_id"].unique()))
        ax.set_ylim(-0.02, 1.05)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Success rate")
    axes[2].set_ylabel("Success rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, frameon=False)
    fig.suptitle("8x8 Slippery Random Maps: Five Maps per Difficulty", y=0.995)
    fig.tight_layout(rect=(0, 0, 0.86, 0.97))
    fig.savefig(FIG / "random8_by_probability_per_map_lines.png", dpi=240)
    plt.close(fig)


def save_random8_variation_decomposition(random8_runs: pd.DataFrame) -> None:
    per_map = (
        random8_runs.groupby(["frozen_prob", "map_id", "display_algorithm"], as_index=False)
        .agg(success=("success", "mean"))
    )
    by_p = (
        per_map.groupby(["frozen_prob", "display_algorithm"], as_index=False)
        .agg(mean_success=("success", "mean"), std_across_maps=("success", "std"))
    )
    by_p.to_csv(TAB / "random8_variance_across_maps.csv", index=False)
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    sns.boxplot(
        data=per_map,
        x="frozen_prob",
        y="success",
        hue="display_algorithm",
        order=sorted(per_map["frozen_prob"].unique()),
        hue_order=[DISPLAY[a] for a in CORE],
        palette=PALETTE,
        ax=ax,
        fliersize=2.5,
    )
    ax.set_title("8x8 Slippery Random Maps: Distribution across Individual Maps")
    ax.set_xlabel("Frozen tile probability p")
    ax.set_ylabel("Success rate across 5 maps")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "random8_map_variation_boxplot.png", dpi=240)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    fixed = load_fixed()

    exploration = fixed_setting(fixed, "8x8_stoch", "Fixed 8x8", EXPLORATION)
    exploration.to_csv(TAB / "exploration_ablation_fixed_8x8.csv", index=False)
    save_bar(exploration, FIG / "exploration_ablation_fixed_8x8.png", "Exploration Ablation on 8x8 Slippery")

    propagation = fixed_setting(fixed, "8x8_stoch", "Fixed 8x8", PROPAGATION)
    propagation.to_csv(TAB / "reward_propagation_ablation_fixed_8x8.csv", index=False)
    save_bar(propagation, FIG / "reward_propagation_ablation_fixed_8x8.png", "Reward-Propagation Ablation on 8x8 Slippery")

    standard4 = load_standard4()
    designed4 = load_designed4()
    fixed8 = fixed_setting(fixed, "8x8_stoch", "Fixed 8x8", CORE)
    random8 = load_random8()
    combined = pd.concat([standard4, designed4, fixed8, random8], ignore_index=True)
    combined = combined[combined["algorithm"].isin(CORE)]
    combined.to_csv(TAB / "stochastic_core_summary.csv", index=False)

    save_bar(
        combined,
        FIG / "map_difficulty_ablation.png",
        "Map/Difficulty Ablation under Slippery Dynamics",
        x="setting",
        hue="display_algorithm",
    )

    rank = combined.copy()
    rank["rank"] = rank.groupby("setting")["mean_success"].rank(ascending=False, method="min")
    rank.to_csv(TAB / "stochastic_algorithm_ranks.csv", index=False)

    pivot = combined.pivot_table(index="display_algorithm", columns="setting", values="mean_success", aggfunc="mean")
    ordered_rows = [DISPLAY[a] for a in CORE if DISPLAY[a] in pivot.index]
    ordered_cols = ["Standard 4x4", "Designed 4x4", "Fixed 8x8", "Random 8x8"]
    pivot = pivot.reindex(index=ordered_rows, columns=ordered_cols)
    save_annotated_heatmap(
        pivot,
        FIG / "stochastic_success_heatmap.png",
        "Stochastic FrozenLake Success Heatmap",
        "Mean success",
        "viridis",
    )

    random8_runs = load_random8_runs()
    gap = random8_runs.groupby(["display_algorithm", "env"], as_index=False)["oracle_gap"].mean()
    gap_pivot = gap.pivot(index="display_algorithm", columns="env", values="oracle_gap").reindex(index=ordered_rows)
    gap_pivot = gap_pivot.rename(columns=lambda c: c.replace("8x8_stoch_", ""))
    save_annotated_heatmap(
        gap_pivot,
        FIG / "random8_oracle_gap_heatmap_fixed.png",
        "8x8 Slippery Random-Map Oracle Gap",
        "Oracle - learned success",
        "rocket_r",
    )

    save_random8_per_map_lines(random8_runs)
    save_random8_faceted_map_lines(random8_runs)
    save_random8_variation_decomposition(random8_runs)

    top = combined.sort_values(["setting", "mean_success"], ascending=[True, False])
    top.groupby("setting").head(3).to_csv(TAB / "top3_by_stochastic_setting.csv", index=False)


if __name__ == "__main__":
    main()
