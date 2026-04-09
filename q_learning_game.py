"""FrozenLake Q-Learning project with plotting, logging, and native fallback.

This script implements tabular Q-Learning for the RL final project. It prefers
Gymnasium's FrozenLake-v1 when available and automatically falls back to a
native implementation when Gymnasium is unavailable in the active environment.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


MAP_LAYOUT = ["SFFF", "FHFH", "FFFH", "HFFG"]
ACTION_LABELS = {0: "^", 1: ">", 2: "v", 3: "<"}
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
GYM_ACTION_MAP = {0: 3, 1: 2, 2: 1, 3: 0}


@dataclass
class Config:
    """Global configuration for training, evaluation, and result export."""

    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    n_episodes: int = 10_000
    max_steps: int = 100
    eval_interval: int = 500
    eval_episodes: int = 100
    smoothing_window: int = 200
    multi_seed: int = 5
    default_seed: int = 0
    output_dir: Path = Path("results")


@dataclass(frozen=True)
class ExperimentSpec:
    """Describes one experimental setting."""

    name: str
    title: str
    is_slippery: bool
    alpha: float
    family: str


class NativeFrozenLakeEnv:
    """A small FrozenLake environment compatible with this project."""

    def __init__(self, is_slippery: bool, seed: Optional[int] = None) -> None:
        self.map_layout = MAP_LAYOUT
        self.nrow = len(self.map_layout)
        self.ncol = len(self.map_layout[0])
        self.n_states = self.nrow * self.ncol
        self.n_actions = 4
        self.is_slippery = is_slippery
        self.start_state = self._find_state("S")
        self.goal_state = self._find_state("G")
        self.terminal_states = {
            self._to_state(row, col)
            for row in range(self.nrow)
            for col in range(self.ncol)
            if self.map_layout[row][col] in {"H", "G"}
        }
        self.rng = np.random.default_rng(seed)
        self.state = self.start_state

    def _find_state(self, symbol: str) -> int:
        """Return the state index for the requested symbol."""

        for row in range(self.nrow):
            for col in range(self.ncol):
                if self.map_layout[row][col] == symbol:
                    return self._to_state(row, col)
        raise ValueError(f"Symbol {symbol!r} not found in the map.")

    def _to_state(self, row: int, col: int) -> int:
        """Convert row/column coordinates to a flattened state index."""

        return row * self.ncol + col

    def _to_pos(self, state: int) -> Tuple[int, int]:
        """Convert a flattened state index to row/column coordinates."""

        return divmod(state, self.ncol)

    def reset(self, seed: Optional[int] = None) -> int:
        """Reset the environment and optionally reseed the RNG."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start_state
        return self.state

    def _move(self, state: int, action: int) -> int:
        """Move one step in the selected direction with boundary handling."""

        row, col = self._to_pos(state)
        delta_row, delta_col = ACTION_DELTAS[action]
        next_row = min(max(row + delta_row, 0), self.nrow - 1)
        next_col = min(max(col + delta_col, 0), self.ncol - 1)
        return self._to_state(next_row, next_col)

    def _sample_transition_action(self, action: int) -> int:
        """Sample the executed action under slippery dynamics."""

        if not self.is_slippery:
            return action
        candidates = [((action - 1) % 4), action, ((action + 1) % 4)]
        return int(self.rng.choice(candidates))

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        """Apply one action and return the next transition tuple."""

        if self.state in self.terminal_states:
            reward = 1.0 if self.state == self.goal_state else 0.0
            return self.state, reward, True, {}

        executed_action = self._sample_transition_action(action)
        next_state = self._move(self.state, executed_action)
        self.state = next_state

        cell = self.map_layout[next_state // self.ncol][next_state % self.ncol]
        done = cell in {"H", "G"}
        reward = 1.0 if cell == "G" else 0.0
        return next_state, reward, done, {"executed_action": ACTION_NAMES[executed_action]}

    def close(self) -> None:
        """Compatibility method for parity with Gymnasium environments."""


class GymFrozenLakeAdapter:
    """Wrap FrozenLake-v1 so the project can use a consistent action layout."""

    def __init__(self, is_slippery: bool, seed: Optional[int] = None) -> None:
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise RuntimeError("Gymnasium is not available.") from exc

        self._env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
        self.n_states = int(self._env.observation_space.n)
        self.n_actions = 4
        self.map_layout = MAP_LAYOUT
        self.goal_state = 15
        self.is_slippery = is_slippery
        self.reset(seed=seed)

    def reset(self, seed: Optional[int] = None) -> int:
        """Reset the wrapped Gymnasium environment."""

        observation, _ = self._env.reset(seed=seed)
        return int(observation)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        """Step through the wrapped environment using canonical actions."""

        gym_action = GYM_ACTION_MAP[action]
        observation, reward, terminated, truncated, info = self._env.step(gym_action)
        done = bool(terminated or truncated)
        return int(observation), float(reward), done, dict(info)

    def close(self) -> None:
        """Close the wrapped environment."""

        self._env.close()


class QTable:
    """Simple Q-table wrapper with tie-aware greedy selection."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.values = np.zeros((n_states, n_actions), dtype=np.float64)

    def max_q(self, state: int) -> float:
        """Return the maximum action value for the provided state."""

        return float(np.max(self.values[state]))

    def best_action(self, state: int, rng: np.random.Generator) -> int:
        """Break action ties randomly among the best-valued actions."""

        state_values = self.values[state]
        best_value = np.max(state_values)
        best_actions = np.flatnonzero(np.isclose(state_values, best_value))
        return int(rng.choice(best_actions))

    def update(self, state: int, action: int, td_target: float, alpha: float) -> float:
        """Apply one Q-Learning update and return the TD error."""

        td_error = td_target - self.values[state, action]
        self.values[state, action] += alpha * td_error
        return float(td_error)


class EpsilonGreedyPolicy:
    """Epsilon-greedy action selection with multiplicative decay."""

    def __init__(
        self,
        epsilon_start: float,
        epsilon_min: float,
        epsilon_decay: float,
        rng: np.random.Generator,
    ) -> None:
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng

    def select_action(self, q_table: QTable, state: int) -> int:
        """Select a random or greedy action according to epsilon."""

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, q_table.values.shape[1]))
        return q_table.best_action(state, self.rng)

    def greedy_action(self, q_table: QTable, state: int) -> int:
        """Select the greedy action without additional exploration."""

        return q_table.best_action(state, self.rng)

    def decay(self) -> None:
        """Decay epsilon after each episode."""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Run FrozenLake Q-Learning experiments.")
    parser.add_argument("--backend", choices=["auto", "gymnasium", "native"], default="auto")
    parser.add_argument("--exp", choices=["all", "exp1", "exp2", "exp3"], default="all")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--multi-seed", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def make_env(
    backend: str,
    is_slippery: bool,
    seed: Optional[int] = None,
) -> Tuple[object, str]:
    """Create the requested environment backend and report the backend used."""

    if backend == "native":
        return NativeFrozenLakeEnv(is_slippery=is_slippery, seed=seed), "native"
    if backend == "gymnasium":
        return GymFrozenLakeAdapter(is_slippery=is_slippery, seed=seed), "gymnasium"
    try:
        return GymFrozenLakeAdapter(is_slippery=is_slippery, seed=seed), "gymnasium"
    except RuntimeError:
        return NativeFrozenLakeEnv(is_slippery=is_slippery, seed=seed), "native"


def q_learning_update(
    q_table: QTable,
    state: int,
    action: int,
    reward: float,
    next_state: int,
    done: bool,
    alpha: float,
    gamma: float,
) -> float:
    """Compute the Q-Learning target and update the Q-table."""

    td_target = reward if done else reward + gamma * q_table.max_q(next_state)
    return q_table.update(state, action, td_target, alpha)


def smooth_curve(values: Sequence[float], window: int) -> np.ndarray:
    """Return a simple moving-average smoothed version of the input sequence."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    if array.size < window:
        return np.full_like(array, np.mean(array), dtype=np.float64)
    weights = np.ones(window, dtype=np.float64) / float(window)
    smoothed = np.convolve(array, weights, mode="valid")
    prefix = np.full(window - 1, smoothed[0], dtype=np.float64)
    return np.concatenate([prefix, smoothed])


def ensure_output_dirs(output_dir: Path) -> Dict[str, Path]:
    """Create and return the standard output directory layout."""

    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    logs_dir = output_dir / "logs"
    for directory in (output_dir, figures_dir, tables_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": output_dir, "figures": figures_dir, "tables": tables_dir, "logs": logs_dir}


def evaluate_policy(
    q_table: QTable,
    backend: str,
    is_slippery: bool,
    max_steps: int,
    eval_episodes: int,
    seed: int,
) -> float:
    """Evaluate the greedy policy and return the success rate."""

    env, _ = make_env(backend=backend, is_slippery=is_slippery, seed=seed)
    rng = np.random.default_rng(seed)
    successes = 0
    try:
        for episode in range(eval_episodes):
            state = env.reset(seed=seed + episode)
            for _ in range(max_steps):
                action = q_table.best_action(state, rng)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                if done:
                    successes += int(reward > 0.0)
                    break
    finally:
        env.close()
    return successes / float(eval_episodes)


def train_once(
    spec: ExperimentSpec,
    config: Config,
    backend: str,
    seed: int,
) -> Dict[str, object]:
    """Run one training job and return the full statistics dictionary."""

    env, backend_used = make_env(backend=backend, is_slippery=spec.is_slippery, seed=seed)
    rng = np.random.default_rng(seed)
    q_table = QTable(n_states=env.n_states, n_actions=env.n_actions)
    policy = EpsilonGreedyPolicy(
        epsilon_start=config.epsilon_start,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        rng=rng,
    )

    rewards: List[float] = []
    epsilons: List[float] = []
    train_rows: List[Dict[str, object]] = []
    eval_rows: List[Dict[str, object]] = []

    try:
        for episode in range(1, config.n_episodes + 1):
            state = env.reset(seed=seed + episode)
            total_reward = 0.0
            steps_taken = 0

            for step in range(1, config.max_steps + 1):
                action = policy.select_action(q_table, state)
                next_state, reward, done, _ = env.step(action)
                q_learning_update(
                    q_table=q_table,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    alpha=spec.alpha,
                    gamma=config.gamma,
                )
                total_reward += reward
                state = next_state
                steps_taken = step
                if done:
                    break

            rewards.append(total_reward)
            epsilons.append(policy.epsilon)
            train_rows.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "steps": steps_taken,
                    "epsilon": policy.epsilon,
                    "alpha": spec.alpha,
                    "is_slippery": spec.is_slippery,
                    "seed": seed,
                    "backend": backend_used,
                }
            )

            if episode % config.eval_interval == 0 or episode == config.n_episodes:
                success_rate = evaluate_policy(
                    q_table=q_table,
                    backend=backend_used,
                    is_slippery=spec.is_slippery,
                    max_steps=config.max_steps,
                    eval_episodes=config.eval_episodes,
                    seed=seed + 100_000 + episode,
                )
                eval_rows.append(
                    {
                        "episode": episode,
                        "success_rate": success_rate,
                        "alpha": spec.alpha,
                        "is_slippery": spec.is_slippery,
                        "seed": seed,
                        "backend": backend_used,
                    }
                )

            policy.decay()
    finally:
        env.close()

    return {
        "spec": spec,
        "backend_used": backend_used,
        "seed": seed,
        "q_table": q_table.values.copy(),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "smoothed_rewards": smooth_curve(rewards, config.smoothing_window),
        "epsilons": np.asarray(epsilons, dtype=np.float64),
        "eval_rows": eval_rows,
        "train_rows": train_rows,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write a list of dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def plot_qtable_heatmap(q_values: np.ndarray, path: Path, title: str) -> None:
    """Plot the max Q-value per state on the 4x4 FrozenLake grid."""

    value_grid = np.max(q_values, axis=1).reshape(4, 4)
    fig, ax = plt.subplots(figsize=(6, 6))
    image = ax.imshow(value_grid, cmap="RdYlGn")
    for row in range(4):
        for col in range(4):
            state = row * 4 + col
            label = MAP_LAYOUT[row][col]
            ax.text(
                col,
                row,
                f"{label}\n{value_grid[row, col]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=10,
            )
    ax.set_title(title)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="max Q(s, a)")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_policy_arrows(q_values: np.ndarray, path: Path, title: str) -> None:
    """Plot the greedy policy on the FrozenLake grid using text arrows."""

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    for row in range(4):
        for col in range(4):
            state = row * 4 + col
            cell = MAP_LAYOUT[row][col]
            if cell == "H":
                text = "X"
            elif cell == "G":
                text = "*"
            else:
                text = ACTION_LABELS[int(np.argmax(q_values[state]))]
            ax.text(col, row, text, ha="center", va="center", fontsize=24)

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
    grouped_results: Dict[str, Sequence[Dict[str, object]]],
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
    grouped_results: Dict[str, Sequence[Dict[str, object]]],
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
    grouped_results: Dict[str, Sequence[Dict[str, object]]],
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


def estimate_convergence_episode(eval_rows: Sequence[Dict[str, object]], threshold: float) -> Optional[int]:
    """Estimate the first evaluation point that reaches the target threshold."""

    for row in eval_rows:
        if float(row["success_rate"]) >= threshold:
            return int(row["episode"])
    return None


def build_run_summary(result: Dict[str, object], config: Config) -> Dict[str, object]:
    """Create a compact summary row for one training run."""

    spec: ExperimentSpec = result["spec"]
    eval_rows = result["eval_rows"]
    rewards = result["rewards"]
    smoothed_rewards = result["smoothed_rewards"]
    target_threshold = 0.95 if spec.name == "exp1" else 0.65
    convergence_episode = estimate_convergence_episode(eval_rows, target_threshold)
    final_success = float(eval_rows[-1]["success_rate"]) if eval_rows else math.nan
    best_success = float(max(row["success_rate"] for row in eval_rows)) if eval_rows else math.nan
    return {
        "condition": spec.name,
        "title": spec.title,
        "family": spec.family,
        "seed": result["seed"],
        "backend": result["backend_used"],
        "alpha": spec.alpha,
        "is_slippery": spec.is_slippery,
        "episodes": config.n_episodes,
        "final_eval_success": final_success,
        "best_eval_success": best_success,
        "final_smoothed_reward": float(smoothed_rewards[-1]) if smoothed_rewards.size else math.nan,
        "mean_reward": float(np.mean(rewards)) if rewards.size else math.nan,
        "convergence_episode": convergence_episode if convergence_episode is not None else "",
    }


def build_aggregate_summary(
    grouped_results: Dict[str, Sequence[Dict[str, object]]],
    config: Config,
) -> List[Dict[str, object]]:
    """Aggregate seed-wise statistics for each condition."""

    rows: List[Dict[str, object]] = []
    for condition, runs in grouped_results.items():
        template_spec: ExperimentSpec = runs[0]["spec"]
        run_summaries = [build_run_summary(run, config) for run in runs]
        final_successes = np.asarray([row["final_eval_success"] for row in run_summaries], dtype=np.float64)
        best_successes = np.asarray([row["best_eval_success"] for row in run_summaries], dtype=np.float64)
        mean_rewards = np.asarray([row["mean_reward"] for row in run_summaries], dtype=np.float64)
        convergence_values = np.asarray(
            [row["convergence_episode"] for row in run_summaries if row["convergence_episode"] != ""],
            dtype=np.float64,
        )
        rows.append(
            {
                "condition": condition,
                "title": template_spec.title,
                "family": template_spec.family,
                "alpha": template_spec.alpha,
                "is_slippery": template_spec.is_slippery,
                "n_seeds": len(runs),
                "mean_final_eval_success": float(np.mean(final_successes)),
                "std_final_eval_success": float(np.std(final_successes)),
                "mean_best_eval_success": float(np.mean(best_successes)),
                "std_best_eval_success": float(np.std(best_successes)),
                "mean_reward": float(np.mean(mean_rewards)),
                "std_reward": float(np.std(mean_rewards)),
                "mean_convergence_episode": float(np.mean(convergence_values))
                if convergence_values.size
                else "",
                "std_convergence_episode": float(np.std(convergence_values))
                if convergence_values.size
                else "",
            }
        )
    return rows


def filter_specs(all_specs: Sequence[ExperimentSpec], exp_choice: str) -> List[ExperimentSpec]:
    """Select the experiments requested by the CLI."""

    if exp_choice == "all":
        return list(all_specs)
    if exp_choice == "exp1":
        return [spec for spec in all_specs if spec.name == "exp1"]
    if exp_choice == "exp2":
        return [spec for spec in all_specs if spec.name == "exp2"]
    if exp_choice == "exp3":
        return [spec for spec in all_specs if spec.family == "alpha"]
    raise ValueError(f"Unsupported experiment selection: {exp_choice}")


def sanitize_name(text: str) -> str:
    """Convert a label to a filesystem-safe name."""

    return text.replace(" ", "_").replace("=", "").replace(".", "_")


def export_run_logs(result: Dict[str, object], logs_dir: Path) -> None:
    """Write per-run training and evaluation logs to disk."""

    spec: ExperimentSpec = result["spec"]
    stem = f"{spec.name}_seed{result['seed']}"
    write_csv(logs_dir / f"{stem}_train.csv", result["train_rows"])
    write_csv(logs_dir / f"{stem}_eval.csv", result["eval_rows"])


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

    all_specs = [
        ExperimentSpec("exp1", "Exp1 Deterministic Baseline", False, config.alpha, "baseline"),
        ExperimentSpec("exp2", "Exp2 Stochastic Baseline", True, config.alpha, "baseline"),
        ExperimentSpec("exp3_alpha_0.01", "Exp3 Alpha 0.01", True, 0.01, "alpha"),
        ExperimentSpec("exp3_alpha_0.10", "Exp3 Alpha 0.10", True, 0.10, "alpha"),
        ExperimentSpec("exp3_alpha_0.50", "Exp3 Alpha 0.50", True, 0.50, "alpha"),
    ]
    selected_specs = filter_specs(all_specs, args.exp)
    seed_values = list(range(config.multi_seed))
    if config.default_seed not in seed_values:
        seed_values.append(config.default_seed)
        seed_values = sorted(seed_values)

    results_by_condition: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for spec in selected_specs:
        for seed in seed_values:
            result = train_once(spec=spec, config=config, backend=args.backend, seed=seed)
            results_by_condition[spec.name].append(result)
            export_run_logs(result, directories["logs"])

    run_summary_rows: List[Dict[str, object]] = []
    for runs in results_by_condition.values():
        run_summary_rows.extend(build_run_summary(run, config) for run in runs)
    aggregate_summary_rows = build_aggregate_summary(results_by_condition, config)

    write_csv(directories["tables"] / "run_summary.csv", run_summary_rows)
    write_csv(directories["tables"] / "summary_table.csv", aggregate_summary_rows)

    # Single-seed figures use the requested default seed when available.
    selected_seed = config.default_seed

    if "exp1" in results_by_condition:
        exp1_seed_run = next(run for run in results_by_condition["exp1"] if run["seed"] == selected_seed)
        plot_learning_curve(
            rewards=exp1_seed_run["rewards"],
            smoothed_rewards=exp1_seed_run["smoothed_rewards"],
            path=directories["figures"] / "exp1_learning_curve.png",
            title="Exp1 Learning Curve (Deterministic FrozenLake)",
        )

    if "exp2" in results_by_condition:
        exp2_seed_run = next(run for run in results_by_condition["exp2"] if run["seed"] == selected_seed)
        plot_learning_curve(
            rewards=exp2_seed_run["rewards"],
            smoothed_rewards=exp2_seed_run["smoothed_rewards"],
            path=directories["figures"] / "exp2_learning_curve.png",
            title="Exp2 Learning Curve (Stochastic FrozenLake)",
        )
        plot_eval_curve(
            eval_rows=exp2_seed_run["eval_rows"],
            path=directories["figures"] / "exp2_eval_curve.png",
            title="Exp2 Evaluation Success Rate",
        )
        plot_epsilon_curve(
            epsilons=exp2_seed_run["epsilons"],
            path=directories["figures"] / "exp2_epsilon_curve.png",
            title="Exp2 Epsilon Decay",
        )
        plot_qtable_heatmap(
            q_values=exp2_seed_run["q_table"],
            path=directories["figures"] / "exp2_qtable_heatmap.png",
            title="Exp2 Max-Q Heatmap",
        )
        plot_policy_arrows(
            q_values=exp2_seed_run["q_table"],
            path=directories["figures"] / "exp2_policy_arrows.png",
            title="Exp2 Greedy Policy",
        )

    alpha_condition_names = [name for name in results_by_condition if name.startswith("exp3_alpha_")]
    if alpha_condition_names:
        alpha_seed_runs = [
            next(run for run in results_by_condition[name] if run["seed"] == selected_seed)
            for name in sorted(alpha_condition_names)
        ]
        plot_alpha_comparison(
            results=alpha_seed_runs,
            path=directories["figures"] / "exp3_alpha_comparison.png",
            title="Exp3 Alpha Ablation (Single Seed)",
        )

    if all(label in results_by_condition for label in ("exp1", "exp2")):
        baseline_groups = {
            "Exp1 deterministic": results_by_condition["exp1"],
            "Exp2 stochastic": results_by_condition["exp2"],
        }
        plot_mean_eval_bands(
            grouped_results=baseline_groups,
            path=directories["figures"] / "baseline_multi_seed_eval.png",
            title="Baseline Evaluation Curves Across Seeds",
        )

    if alpha_condition_names:
        alpha_groups = {
            f"alpha={results_by_condition[name][0]['spec'].alpha:g}": results_by_condition[name]
            for name in sorted(alpha_condition_names)
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

    print(f"Completed experiments: {', '.join(results_by_condition.keys())}")
    print(f"Backend requested: {args.backend}")
    print(f"Results saved to: {directories['root'].resolve()}")


if __name__ == "__main__":
    main()
