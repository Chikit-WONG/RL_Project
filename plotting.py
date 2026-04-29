"""All matplotlib plotting functions for training curves, Q-tables, and comparisons."""

from __future__ import annotations

import math as _math
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from config import ACTION_LABELS, MAP_LAYOUT


def plot_learning_curve(
    rewards: np.ndarray,
    smoothed_rewards: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """Plot raw and smoothed episode rewards."""

    episodes = np.arange(1, rewards.size + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.25, color="tab:blue", label="Raw reward")
    ax.plot(episodes, smoothed_rewards, linewidth=2.0, color="tab:red", label="Smoothed reward")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_eval_curve(eval_rows: Sequence[Dict[str, object]], path: Path, title: str) -> None:
    """Plot evaluation success rate against training episodes."""

    eval_episodes = [row["episode"] for row in eval_rows]
    success_rates = [row["success_rate"] for row in eval_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eval_episodes, success_rates, marker="o", linewidth=2.0, color="tab:green")
    ax.set_title(title)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_epsilon_curve(epsilons: np.ndarray, path: Path, title: str) -> None:
    """Plot the epsilon value used over training episodes."""

    episodes = np.arange(1, epsilons.size + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, epsilons, linewidth=2.0, color="tab:orange")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qtable_heatmap(
    q_values: np.ndarray,
    path: Path,
    title: str,
    map_layout: Optional[Sequence[str]] = None,
) -> None:
    """Plot the max Q-value per state on the FrozenLake grid."""

    layout = list(map_layout) if map_layout is not None else MAP_LAYOUT
    nrow = len(layout)
    ncol = len(layout[0])
    value_grid = np.max(q_values, axis=1).reshape(nrow, ncol)
    cell_px = max(1.2, 6.0 / nrow)
    fig, ax = plt.subplots(figsize=(ncol * cell_px, nrow * cell_px))
    image = ax.imshow(value_grid, cmap="RdYlGn")
    fontsize = max(6, 10 - nrow)
    for row in range(nrow):
        for col in range(ncol):
            label = layout[row][col]
            ax.text(
                col,
                row,
                f"{label}\n{value_grid[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=fontsize,
            )
    ax.set_title(title)
    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="max Q(s, a)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_policy_arrows(
    q_values: np.ndarray,
    path: Path,
    title: str,
    map_layout: Optional[Sequence[str]] = None,
) -> None:
    """Plot the greedy policy on the FrozenLake grid using text arrows."""

    layout = list(map_layout) if map_layout is not None else MAP_LAYOUT
    nrow = len(layout)
    ncol = len(layout[0])
    cell_px = max(1.2, 6.0 / max(nrow, ncol))
    fig, ax = plt.subplots(figsize=(ncol * cell_px, nrow * cell_px))
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    fontsize = max(10, 24 - nrow * 2)
    for row in range(nrow):
        for col in range(ncol):
            state = row * ncol + col
            cell = layout[row][col]
            if cell == "H":
                text = "X"
            elif cell == "G":
                text = "*"
            else:
                text = ACTION_LABELS[int(np.argmax(q_values[state]))]
            ax.text(col, row, text, ha="center", va="center", fontsize=fontsize)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_alpha_comparison(results: Sequence[Dict[str, object]], path: Path, title: str) -> None:
    """Plot the single-seed alpha ablation reward curves on one figure."""

    fig, ax = plt.subplots(figsize=(10, 5))
    for result in results:
        spec = result["spec"]
        rewards = result["rewards"]
        smoothed_rewards = result["smoothed_rewards"]
        episodes = np.arange(1, rewards.size + 1)
        ax.plot(episodes, rewards, alpha=0.15, linewidth=0.8)
        ax.plot(episodes, smoothed_rewards, linewidth=2.0, label=f"alpha={spec.alpha}")
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_mean_eval_bands(
    grouped_results: Mapping[str, Sequence[Dict[str, object]]],
    path: Path,
    title: str,
) -> None:
    """Plot mean evaluation curves with one-standard-deviation bands."""

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, runs in grouped_results.items():
        episodes = np.asarray([row["episode"] for row in runs[0]["eval_rows"]], dtype=np.int32)
        curve_matrix = np.asarray(
            [[row["success_rate"] for row in run["eval_rows"]] for run in runs],
            dtype=np.float64,
        )
        mean_curve = np.mean(curve_matrix, axis=0)
        std_curve = np.std(curve_matrix, axis=0)
        ax.plot(episodes, mean_curve, linewidth=2.0, label=label)
        ax.fill_between(
            episodes,
            np.clip(mean_curve - std_curve, 0.0, 1.0),
            np.clip(mean_curve + std_curve, 0.0, 1.0),
            alpha=0.2,
        )
    ax.set_title(title)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Greedy evaluation success rate")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_alpha_final_bar(
    grouped_results: Mapping[str, Sequence[Dict[str, object]]],
    path: Path,
    title: str,
) -> None:
    """Plot final evaluation success rates for each alpha as a bar chart."""

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for label, runs in grouped_results.items():
        finals = [run["eval_rows"][-1]["success_rate"] for run in runs]
        labels.append(label)
        means.append(float(np.mean(finals)))
        stds.append(float(np.std(finals)))

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, means, yerr=stds, capsize=6, color=["#4C78A8", "#F58518", "#54A24B"][: len(labels)])
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Final evaluation success rate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_alpha_final_boxplot(
    grouped_results: Mapping[str, Sequence[Dict[str, object]]],
    path: Path,
    title: str,
) -> None:
    """Plot final evaluation success rates as boxplots with per-seed scatter."""

    labels: List[str] = []
    final_groups: List[List[float]] = []
    for label, runs in grouped_results.items():
        labels.append(label)
        final_groups.append([float(run["eval_rows"][-1]["success_rate"]) for run in runs])

    positions = np.arange(1, len(labels) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    boxplot = ax.boxplot(
        final_groups,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=False,
    )

    colors = ["#4C78A8", "#F58518", "#54A24B"][: len(labels)]
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.8)

    rng = np.random.default_rng(2026)
    for idx, values in enumerate(final_groups, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), idx, dtype=np.float64) + jitter,
            values,
            color="black",
            s=28,
            alpha=0.8,
            zorder=3,
        )

    ax.set_xticks(positions, labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Final evaluation success rate")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_map_heatmap_grid(
    map_layouts: Sequence[Sequence[str]],
    path: Path,
    title: str,
    n_cols: int = 5,
) -> None:
    """Visualise a collection of map layouts as a grid of colour-coded tiles.

    Cell colours: S=green, G=orange, H=dark-brown, F=light-blue.
    """

    _CELL_COLORS = {
        "S": (0.298, 0.686, 0.314),
        "G": (1.000, 0.596, 0.094),
        "H": (0.365, 0.251, 0.216),
        "F": (0.733, 0.871, 0.984),
    }

    n_maps = len(map_layouts)
    n_rows_grid = _math.ceil(n_maps / n_cols)
    fig, axes = plt.subplots(
        n_rows_grid, n_cols, figsize=(n_cols * 2.0, n_rows_grid * 2.0)
    )
    axes_flat = np.asarray(axes).reshape(-1)

    for idx, layout in enumerate(map_layouts):
        ax = axes_flat[idx]
        nrow = len(layout)
        ncol_map = len(layout[0])
        color_grid = np.ones((nrow, ncol_map, 3), dtype=np.float64)
        for r in range(nrow):
            for c in range(ncol_map):
                color_grid[r, c] = _CELL_COLORS.get(layout[r][c], (1.0, 1.0, 1.0))
        ax.imshow(color_grid, interpolation="nearest")
        ax.set_title(f"Map {idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(n_maps, n_rows_grid * n_cols):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
