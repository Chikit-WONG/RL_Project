"""Shared helpers for explicit-layout slippery FrozenLake experiments."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

import enhanced_frozenlake as base


class CustomMapFrozenLakeEnv(base.NativeFrozenLakeEnv):
    """Native FrozenLake environment over an explicit map layout."""

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
    """Generate a random map with a carved safe path from start to goal."""

    rng = np.random.default_rng(seed)
    grid = np.where(rng.random((size, size)) < frozen_prob, "F", "H")
    row = col = 0
    grid[row, col] = "S"
    while row < size - 1 or col < size - 1:
        if row == size - 1:
            col += 1
        elif col == size - 1:
            row += 1
        elif rng.random() < 0.5:
            row += 1
        else:
            col += 1
        grid[row, col] = "F"
    grid[-1, -1] = "G"
    return ["".join(line) for line in grid]


def parse_layout(text: str) -> list[str]:
    """Read layouts saved with either pipe or slash row separators."""

    sep = "|" if "|" in text else "/"
    return [row for row in text.split(sep) if row]


def evaluate_on_env(agent, env_factory, cfg: base.ExperimentConfig, seed: int) -> float:
    successes = 0
    env = env_factory(seed)
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
