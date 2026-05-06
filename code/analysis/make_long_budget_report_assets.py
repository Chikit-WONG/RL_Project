"""Build report/submission figures from long-budget per-map learning curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "long_budget_report_assets_v1"
FIG = OUT / "figures"
TAB = OUT / "tables"

DISPLAY_ORDER = [
    "Q-learning",
    "SARSA",
    "Expected SARSA",
    "SARSA(lambda)",
    "Q(lambda)",
    "Optimistic Q",
    "Dyna-Q",
]

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
    (FIG / "per_map_random8").mkdir(parents=True, exist_ok=True)
    (FIG / "per_map_designed4").mkdir(parents=True, exist_ok=True)


def load_curves() -> tuple[pd.DataFrame, pd.DataFrame]:
    r8 = pd.read_csv(ROOT / "random8_learning_curves_long_v1" / "tables" / "random8_learning_curves.csv")
    d4 = pd.read_csv(ROOT / "designed4_learning_curves_long_v1" / "tables" / "designed4_learning_curves.csv")
    for df in [r8, d4]:
        df["display_algorithm"] = pd.Categorical(df["display_algorithm"], DISPLAY_ORDER, ordered=True)
        df["oracle_gap"] = df["oracle_success"] - df["success_rate"]
    return r8, d4


def trapezoid_auc(group: pd.DataFrame) -> float:
    g = group.sort_values("episode")
    denom = max(float(g["episode"].max()), 1.0)
    auc_fn = getattr(np, "trapezoid", np.trapz)
    return float(auc_fn(g["success_rate"], g["episode"]) / denom)


def first_reach(group: pd.DataFrame, threshold: float = 0.8) -> float:
    g = group.sort_values("episode")
    reached = g[g["success_rate"] >= threshold]
    if reached.empty:
        return np.nan
    return float(reached["episode"].iloc[0])


def final_table(curves: pd.DataFrame, label: str) -> pd.DataFrame:
    final = curves.sort_values("episode").groupby(["map_label", "algorithm", "display_algorithm", "seed"], as_index=False).tail(1)
    auc = (
        curves.groupby(["map_label", "algorithm", "display_algorithm", "seed"], observed=True)
        .apply(trapezoid_auc, include_groups=False)
        .reset_index(name="eval_auc")
    )
    conv = (
        curves.groupby(["map_label", "algorithm", "display_algorithm", "seed"], observed=True)
        .apply(first_reach, include_groups=False)
        .reset_index(name="first_episode_success_ge_0p8")
    )
    out = final.merge(auc, on=["map_label", "algorithm", "display_algorithm", "seed"], how="left")
    out = out.merge(conv, on=["map_label", "algorithm", "display_algorithm", "seed"], how="left")
    out["suite_label"] = label
    return out


def annotated_heatmap(
    matrix: pd.DataFrame,
    path: Path,
    title: str,
    cbar_label: str,
    cmap: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    fmt: str = ".2f",
    nan_label: str = "--",
) -> None:
    values = matrix.to_numpy(dtype=float)
    n_rows, n_cols = values.shape
    fig, ax = plt.subplots(figsize=(max(8.5, 0.52 * n_cols + 4.2), max(4.8, 0.58 * n_rows + 1.9)))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=45 if n_cols > 8 else 0, ha="right" if n_cols > 8 else "center")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index)
    ax.set_title(title, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    threshold = vmin + 0.55 * (vmax - vmin)
    for i in range(n_rows):
        for j in range(n_cols):
            value = values[i, j]
            text = nan_label if np.isnan(value) else format(value, fmt)
            color = "#222222" if (np.isnan(value) or value >= threshold) else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=9.0, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=260)
    plt.close(fig)


def save_final_heatmaps(final: pd.DataFrame, prefix: str) -> None:
    matrix = final.pivot_table(index="display_algorithm", columns="map_label", values="success_rate", aggfunc="mean", observed=True)
    matrix = matrix.reindex(index=DISPLAY_ORDER)
    annotated_heatmap(
        matrix,
        FIG / f"{prefix}_final_success_heatmap_by_map.png",
        f"{prefix.upper()} Final Success by Map",
        "Final success",
        "viridis",
    )
    gap = final.copy()
    gap["oracle_gap"] = gap["oracle_success"] - gap["success_rate"]
    matrix = gap.pivot_table(index="display_algorithm", columns="map_label", values="oracle_gap", aggfunc="mean", observed=True).reindex(index=DISPLAY_ORDER)
    annotated_heatmap(
        matrix,
        FIG / f"{prefix}_oracle_gap_heatmap_by_map.png",
        f"{prefix.upper()} Oracle Gap by Map",
        "Oracle - learned success",
        "rocket_r",
    )
    auc = final.pivot_table(index="display_algorithm", columns="map_label", values="eval_auc", aggfunc="mean", observed=True).reindex(index=DISPLAY_ORDER)
    annotated_heatmap(
        auc,
        FIG / f"{prefix}_auc_heatmap_by_map.png",
        f"{prefix.upper()} Evaluation AUC by Map",
        "Normalized AUC",
        "mako",
    )
    conv = final.pivot_table(index="display_algorithm", columns="map_label", values="first_episode_success_ge_0p8", aggfunc="mean", observed=True).reindex(index=DISPLAY_ORDER)
    max_ep = 10000 if prefix == "random8" else 2000
    annotated_heatmap(
        conv,
        FIG / f"{prefix}_convergence_episode_heatmap.png",
        f"{prefix.upper()} First Episode with Success >= 0.80",
        "Training episode",
        "magma_r",
        vmin=0,
        vmax=max_ep,
        fmt=".0f",
    )


def save_mean_curve(curves: pd.DataFrame, prefix: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    summary = curves.groupby(["episode", "display_algorithm"], as_index=False, observed=True).agg(
        mean_success=("success_rate", "mean"),
        std_success=("success_rate", "std"),
    )
    for name in DISPLAY_ORDER:
        data = summary[summary["display_algorithm"] == name].sort_values("episode")
        if data.empty:
            continue
        ax.plot(data["episode"], data["mean_success"], label=name, color=PALETTE[name], linewidth=2.0)
        lower = (data["mean_success"] - data["std_success"]).clip(0, 1)
        upper = (data["mean_success"] + data["std_success"]).clip(0, 1)
        ax.fill_between(data["episode"], lower, upper, color=PALETTE[name], alpha=0.10, linewidth=0)
    oracle = curves.groupby("episode", as_index=False)["oracle_success"].mean()
    ax.plot(oracle["episode"], oracle["oracle_success"], color="black", linestyle="--", linewidth=2.2, label="Mean oracle")
    ax.set_title(title)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"{prefix}_mean_learning_curve.png", dpi=240)
    plt.close(fig)


def save_probability_plot(r8_final: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 6.1))
    sns.stripplot(
        data=r8_final,
        x="frozen_prob",
        y="success_rate",
        hue="display_algorithm",
        hue_order=DISPLAY_ORDER,
        dodge=True,
        jitter=0.12,
        size=4.0,
        alpha=0.75,
        palette=PALETTE,
        ax=ax,
    )
    sns.pointplot(
        data=r8_final,
        x="frozen_prob",
        y="success_rate",
        hue="display_algorithm",
        hue_order=DISPLAY_ORDER,
        dodge=0.55,
        errorbar=None,
        markers="_",
        linestyles="",
        palette=PALETTE,
        ax=ax,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[: len(DISPLAY_ORDER)], labels[: len(DISPLAY_ORDER)], loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_title("Random 8x8 Long-Budget Final Success: Individual Map Points")
    ax.set_xlabel("Frozen tile probability p")
    ax.set_ylabel("Final success at 10000 episodes")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "random8_final_success_points_by_probability.png", dpi=240)
    plt.close(fig)


def save_algorithm_ranking(final: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = (
        final.groupby(["algorithm", "display_algorithm"], as_index=False, observed=True)
        .agg(
            mean_final_success=("success_rate", "mean"),
            std_final_success=("success_rate", "std"),
            mean_auc=("eval_auc", "mean"),
            mean_oracle_gap=("oracle_gap", "mean"),
            mean_first_episode_ge_0p8=("first_episode_success_ge_0p8", "mean"),
            n=("success_rate", "count"),
        )
        .sort_values("mean_final_success", ascending=False)
    )
    out.to_csv(TAB / f"{prefix}_long_budget_algorithm_ranking.csv", index=False)
    return out


def save_per_map_summary(final: pd.DataFrame, prefix: str) -> None:
    out = (
        final.groupby(["map_label", "frozen_prob", "map_id", "holes"], as_index=False)
        .agg(
            oracle_success=("oracle_success", "first"),
            mean_final_success=("success_rate", "mean"),
            best_final_success=("success_rate", "max"),
            worst_final_success=("success_rate", "min"),
            best_algorithm=("display_algorithm", lambda s: final.loc[s.index, :].sort_values("success_rate", ascending=False)["display_algorithm"].iloc[0]),
        )
    )
    out.to_csv(TAB / f"{prefix}_long_budget_map_summary.csv", index=False)


def save_combined_stochastic_summary(r8_final: pd.DataFrame, d4_final: pd.DataFrame) -> None:
    standard_path = ROOT / "standard_4x4_slippery_focused_v2" / "tables" / "standard_4x4_aggregate_summary.csv"
    fixed_path = ROOT.parent / "RL_Project_Enhanced" / "enhanced_results_final_v1" / "tables" / "aggregate_summary_final.csv"
    standard = pd.read_csv(standard_path)
    fixed = pd.read_csv(fixed_path)

    display_to_algorithm = {
        "Q-learning": "q_learning",
        "SARSA": "sarsa",
        "Expected SARSA": "expected_sarsa",
        "SARSA(lambda)": "sarsa_lambda",
        "Q(lambda)": "q_lambda",
        "Optimistic Q": "optimistic_q",
        "Dyna-Q": "dyna_q",
    }
    rows = []
    for display, alg in display_to_algorithm.items():
        s = standard[standard["algorithm"] == alg]
        if not s.empty:
            rows.append({"setting": "Standard 4x4", "display_algorithm": display, "mean_success": float(s["mean_final_success"].iloc[0])})
        d = d4_final[d4_final["display_algorithm"].astype(str) == display]
        if not d.empty:
            rows.append({"setting": "Designed 4x4 long", "display_algorithm": display, "mean_success": float(d["success_rate"].mean())})
        f = fixed[(fixed["env"] == "8x8_stoch") & (fixed["algorithm"] == alg)]
        if not f.empty:
            rows.append({"setting": "Fixed 8x8", "display_algorithm": display, "mean_success": float(f["mean_final_success"].iloc[0])})
        r = r8_final[r8_final["display_algorithm"].astype(str) == display]
        if not r.empty:
            rows.append({"setting": "Random 8x8 long", "display_algorithm": display, "mean_success": float(r["success_rate"].mean())})
    summary = pd.DataFrame(rows)
    summary.to_csv(TAB / "stochastic_core_summary_long_budget.csv", index=False)
    matrix = summary.pivot(index="display_algorithm", columns="setting", values="mean_success")
    matrix = matrix.reindex(index=DISPLAY_ORDER, columns=["Standard 4x4", "Designed 4x4 long", "Fixed 8x8", "Random 8x8 long"])
    annotated_heatmap(
        matrix,
        FIG / "stochastic_success_heatmap.png",
        "Stochastic FrozenLake Success Heatmap (Long-Budget Update)",
        "Mean success",
        "viridis",
    )

    r8_by_prob = (
        r8_final.groupby(["frozen_prob", "display_algorithm"], as_index=False, observed=True)
        .agg(mean_success=("success_rate", "mean"), std_success=("success_rate", "std"))
    )
    r8_by_prob.to_csv(TAB / "random8_long_budget_by_probability.csv", index=False)
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    for name in DISPLAY_ORDER:
        data = r8_by_prob[r8_by_prob["display_algorithm"].astype(str) == name].sort_values("frozen_prob")
        if data.empty:
            continue
        ax.plot(data["frozen_prob"], data["mean_success"], marker="o", linewidth=2.0, label=name, color=PALETTE[name])
        lower = (data["mean_success"] - data["std_success"]).clip(0, 1)
        upper = (data["mean_success"] + data["std_success"]).clip(0, 1)
        ax.fill_between(data["frozen_prob"], lower, upper, color=PALETTE[name], alpha=0.10, linewidth=0)
    oracle = r8_final.groupby("frozen_prob", as_index=False)["oracle_success"].mean()
    ax.plot(oracle["frozen_prob"], oracle["oracle_success"], color="black", linestyle="--", marker="D", linewidth=2.0, label="Mean oracle")
    ax.set_title("Random 8x8 Long-Budget Success by Frozen Probability")
    ax.set_xlabel("Frozen tile probability p")
    ax.set_ylabel("Final success at 10000 episodes")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "random8_long_budget_success_vs_frozen_prob.png", dpi=240)
    plt.close(fig)


def copy_existing_panels() -> None:
    sources = [
        ROOT / "random8_learning_curves_long_v1" / "figures" / "random8_learning_curves_all_maps_panel.png",
        ROOT / "random8_learning_curves_long_v1" / "figures" / "random8_learning_curves_p0.80.png",
        ROOT / "random8_learning_curves_long_v1" / "figures" / "random8_learning_curves_p0.85.png",
        ROOT / "random8_learning_curves_long_v1" / "figures" / "random8_learning_curves_p0.90.png",
        ROOT / "random8_learning_curves_long_v1" / "figures" / "random8_learning_curves_p0.95.png",
        ROOT / "designed4_learning_curves_long_v1" / "figures" / "designed4_learning_curves_all_maps_panel.png",
    ]
    for src in sources:
        if src.exists():
            (FIG / src.name).write_bytes(src.read_bytes())
    for src in (ROOT / "random8_learning_curves_long_v1" / "figures" / "per_map").glob("*.png"):
        (FIG / "per_map_random8" / src.name).write_bytes(src.read_bytes())
    for src in (ROOT / "designed4_learning_curves_long_v1" / "figures" / "per_map").glob("*.png"):
        (FIG / "per_map_designed4" / src.name).write_bytes(src.read_bytes())


def main() -> None:
    ensure_dirs()
    r8, d4 = load_curves()
    r8_final = final_table(r8, "Random 8x8 long")
    d4_final = final_table(d4, "Designed 4x4 long")
    for df in [r8_final, d4_final]:
        df["oracle_gap"] = df["oracle_success"] - df["success_rate"]

    r8_final.to_csv(TAB / "random8_long_budget_final_success.csv", index=False)
    d4_final.to_csv(TAB / "designed4_long_budget_final_success.csv", index=False)
    save_algorithm_ranking(r8_final, "random8")
    save_algorithm_ranking(d4_final, "designed4")
    save_per_map_summary(r8_final, "random8")
    save_per_map_summary(d4_final, "designed4")

    save_final_heatmaps(r8_final, "random8")
    save_final_heatmaps(d4_final, "designed4")
    save_mean_curve(r8, "random8", "Random 8x8 Long-Budget Mean Learning Curve")
    save_mean_curve(d4, "designed4", "Designed 4x4 Long-Budget Mean Learning Curve")
    save_probability_plot(r8_final)
    save_combined_stochastic_summary(r8_final, d4_final)
    copy_existing_panels()


if __name__ == "__main__":
    main()
