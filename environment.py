"""FrozenLake environment implementations and factory function."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ACTION_DELTAS, ACTION_NAMES, GYM_ACTION_MAP, MAP_LAYOUT


# ---------------------------------------------------------------------------
# Map generation helpers
# ---------------------------------------------------------------------------

def _bfs_reachable(grid: List[List[str]], size: int) -> bool:
    """Return True if the goal (bottom-right) is reachable from the start (top-left)."""

    goal = (size - 1, size - 1)
    visited: set = {(0, 0)}
    queue: deque = deque([(0, 0)])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < size
                and 0 <= nc < size
                and (nr, nc) not in visited
                and grid[nr][nc] != "H"
            ):
                visited.add((nr, nc))
                queue.append((nr, nc))
    return False


def generate_random_map(size: int, hole_prob: float = 0.25, seed: int = 0) -> List[str]:
    """Generate a random FrozenLake map guaranteed to have a path from S to G.

    Parameters
    ----------
    size:
        Side length of the square grid.
    hole_prob:
        Probability that any free cell becomes a hole.
    seed:
        RNG seed; deterministic for the same (size, hole_prob, seed) triplet.

    Returns
    -------
    List[str]
        Row strings in the same format as MAP_LAYOUT (e.g. ``["SFFF", ...``]).
    """

    rng = np.random.default_rng(seed)
    while True:
        grid: List[List[str]] = [["F"] * size for _ in range(size)]
        grid[0][0] = "S"
        grid[size - 1][size - 1] = "G"
        for r in range(size):
            for c in range(size):
                if grid[r][c] == "F" and rng.random() < hole_prob:
                    grid[r][c] = "H"
        if _bfs_reachable(grid, size):
            return ["".join(row) for row in grid]


# ---------------------------------------------------------------------------
# Environment implementations
# ---------------------------------------------------------------------------


class NativeFrozenLakeEnv:
    """A small FrozenLake environment compatible with this project."""

    def __init__(
        self,
        is_slippery: bool,
        seed: Optional[int] = None,
        map_layout: Optional[List[str]] = None,
    ) -> None:
        self.map_layout = map_layout if map_layout is not None else MAP_LAYOUT
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
        self.nrow = 4
        self.ncol = 4
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_env(
    backend: str,
    is_slippery: bool,
    seed: Optional[int] = None,
    map_layout: Optional[List[str]] = None,
) -> Tuple[object, str]:
    """Create the requested environment backend and report the backend used.

    When *map_layout* is provided the native backend is always used, because
    Gymnasium only supports the standard 4x4 and 8x8 layouts.
    """

    if map_layout is not None:
        return NativeFrozenLakeEnv(is_slippery=is_slippery, seed=seed, map_layout=map_layout), "native"
    if backend == "native":
        return NativeFrozenLakeEnv(is_slippery=is_slippery, seed=seed), "native"
    if backend == "gymnasium":
        return GymFrozenLakeAdapter(is_slippery=is_slippery, seed=seed), "gymnasium"
    try:
        return GymFrozenLakeAdapter(is_slippery=is_slippery, seed=seed), "gymnasium"
    except RuntimeError:
        return NativeFrozenLakeEnv(is_slippery=is_slippery, seed=seed), "native"
