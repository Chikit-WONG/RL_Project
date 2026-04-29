"""Global constants, configuration dataclasses, and experiment specifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Map & action constants
# ---------------------------------------------------------------------------

MAP_LAYOUT = ["SFFF", "FHFH", "FFFH", "HFFG"]
ACTION_LABELS = {0: "^", 1: ">", 2: "v", 3: "<"}
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
GYM_ACTION_MAP = {0: 3, 1: 2, 2: 1, 3: 0}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

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
    policy_type: str = "epsilon_greedy"           # random / epsilon_greedy / ucb / boltzmann
    map_layout: Optional[Tuple[str, ...]] = None  # None -> default 4x4 MAP_LAYOUT
    algo: str = "q_learning"                      # q_learning / double_q_learning
    reward_shaping: bool = False
    step_penalty: bool = False
    map_id: int = 0
