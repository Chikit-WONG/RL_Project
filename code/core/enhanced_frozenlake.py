"""Enhanced FrozenLake benchmark for classical and advanced RL methods.

This script keeps the original project intact and adds a larger experiment
matrix: multiple tabular RL algorithms, 4x4/8x8/16x16 maps, Optuna tuning, and
comparison plots/tables for report and presentation use.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import warnings
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


ACTION_LABELS = {0: "^", 1: ">", 2: "v", 3: "<"}
ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
ACTION_DELTAS = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

STANDARD_4X4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
STANDARD_8X8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]


@dataclass
class ExperimentConfig:
    episodes: int = 5_000
    max_steps: int = 200
    eval_interval: int = 250
    eval_episodes: int = 100
    seeds: int = 5
    alpha: float = 0.15
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.996
    lam: float = 0.8
    n_step: int = 4
    output_dir: Path = Path("enhanced_results")


@dataclass(frozen=True)
class EnvSpec:
    map_size: int
    slippery: bool
    frozen_prob: float = 0.85

    @property
    def label(self) -> str:
        mode = "stoch" if self.slippery else "det"
        return f"{self.map_size}x{self.map_size}_{mode}"


class NativeFrozenLakeEnv:
    """Gym-like FrozenLake with arbitrary square maps and slippery dynamics."""

    def __init__(
        self,
        map_size: int,
        slippery: bool,
        seed: int,
        frozen_prob: float = 0.85,
    ) -> None:
        # Keep generated maps fixed across run seeds; seeds should affect
        # exploration and transition sampling, not the task definition itself.
        self.map_layout = make_map(map_size=map_size, seed=2026, frozen_prob=frozen_prob)
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

    def _find(self, symbol: str) -> int:
        for r, row in enumerate(self.map_layout):
            for c, cell in enumerate(row):
                if cell == symbol:
                    return self._to_state(r, c)
        raise ValueError(f"Map has no {symbol!r} cell.")

    def _to_state(self, row: int, col: int) -> int:
        return row * self.ncol + col

    def _to_pos(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.ncol)

    def reset(self, seed: Optional[int] = None) -> int:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.state = self.start_state
        return self.state

    def _move(self, state: int, action: int) -> int:
        row, col = self._to_pos(state)
        dr, dc = ACTION_DELTAS[action]
        nr = min(max(row + dr, 0), self.nrow - 1)
        nc = min(max(col + dc, 0), self.ncol - 1)
        return self._to_state(nr, nc)

    def transition_outcomes(self, state: int, action: int) -> List[Tuple[float, int, float, bool]]:
        actions = [action] if not self.slippery else [((action - 1) % 4), action, ((action + 1) % 4)]
        prob = 1.0 / len(actions)
        outcomes = []
        for executed in actions:
            next_state = self._move(state, executed)
            cell = self.map_layout[next_state // self.ncol][next_state % self.ncol]
            done = cell in {"H", "G"}
            reward = 1.0 if cell == "G" else 0.0
            outcomes.append((prob, next_state, reward, done))
        return outcomes

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        if self.state in self.terminal_states:
            return self.state, float(self.state == self.goal_state), True, {}
        outcomes = self.transition_outcomes(self.state, action)
        idx = int(self.rng.choice(len(outcomes), p=[x[0] for x in outcomes]))
        _, next_state, reward, done = outcomes[idx]
        self.state = next_state
        return next_state, reward, done, {}


def make_map(map_size: int, seed: int, frozen_prob: float) -> List[str]:
    if map_size == 4:
        return STANDARD_4X4
    if map_size == 8:
        return STANDARD_8X8

    rng = np.random.default_rng(seed + 13_337 + map_size)
    grid = np.where(rng.random((map_size, map_size)) < frozen_prob, "F", "H")
    grid[0, 0] = "S"
    grid[-1, -1] = "G"
    # Carve a guaranteed but not always shortest corridor.
    row = 0
    for col in range(map_size):
        grid[row, col] = "F"
    for row in range(map_size):
        grid[row, map_size - 1] = "F"
    grid[0, 0] = "S"
    grid[-1, -1] = "G"
    return ["".join(row) for row in grid]


class BaseAgent:
    name = "base"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.epsilon = cfg.epsilon_start

    def best_action(self, state: int) -> int:
        values = self.q[state]
        best = np.flatnonzero(np.isclose(values, np.max(values)))
        return int(self.rng.choice(best))

    def act(self, state: int, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.best_action(state)

    def decay(self) -> None:
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        state = env.reset()
        total_reward = 0.0
        last_td = 0.0
        for step in range(1, self.cfg.max_steps + 1):
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            last_td = self.update(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state
            if done:
                break
        self.decay()
        return total_reward, step, abs(last_td)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        raise NotImplementedError


class RandomPolicyAgent(BaseAgent):
    name = "random_policy"

    def act(self, state: int, greedy: bool = False) -> int:
        return int(self.rng.integers(self.n_actions))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        return 0.0


class QLearningAgent(BaseAgent):
    name = "q_learning"

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        target = reward if done else reward + self.cfg.gamma * np.max(self.q[next_state])
        td = target - self.q[state, action]
        self.q[state, action] += self.cfg.alpha * td
        return float(td)


class OptimisticQLearningAgent(QLearningAgent):
    name = "optimistic_q"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.q.fill(getattr(cfg, "optimistic_init", 1.0))


class SoftmaxQLearningAgent(QLearningAgent):
    name = "softmax_q"

    def act(self, state: int, greedy: bool = False) -> int:
        if greedy:
            return self.best_action(state)
        temperature = max(0.05, self.epsilon)
        logits = self.q[state] / temperature
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return int(self.rng.choice(self.n_actions, p=probs))


class UCBQLearningAgent(QLearningAgent):
    name = "ucb_q"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.action_counts = np.zeros((n_states, n_actions), dtype=np.float64)
        self.total_steps = 1.0
        self.c = 1.5

    def act(self, state: int, greedy: bool = False) -> int:
        if greedy:
            return self.best_action(state)
        bonus = self.c * np.sqrt(np.log(self.total_steps + 1.0) / (self.action_counts[state] + 1.0))
        return int(np.argmax(self.q[state] + bonus))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        self.action_counts[state, action] += 1.0
        self.total_steps += 1.0
        return super().update(state, action, reward, next_state, done)


class DynaQAgent(QLearningAgent):
    name = "dyna_q"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
        self.planning_steps = getattr(cfg, "planning_steps", 10)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        td = super().update(state, action, reward, next_state, done)
        self.model[(state, action)] = (reward, next_state, done)
        keys = list(self.model.keys())
        for _ in range(min(self.planning_steps, len(keys))):
            s, a = keys[int(self.rng.integers(len(keys)))]
            r, ns, d = self.model[(s, a)]
            super().update(s, a, r, ns, d)
        return td


class DynaQPlusAgent(DynaQAgent):
    name = "dyna_q_plus"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.time = 0
        self.last_seen = np.zeros((n_states, n_actions), dtype=np.float64)
        self.kappa = getattr(cfg, "dyna_plus_kappa", 0.001)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        self.time += 1
        td = QLearningAgent.update(self, state, action, reward, next_state, done)
        self.model[(state, action)] = (reward, next_state, done)
        self.last_seen[state, action] = self.time
        keys = list(self.model.keys())
        for _ in range(min(self.planning_steps, len(keys))):
            s, a = keys[int(self.rng.integers(len(keys)))]
            r, ns, d = self.model[(s, a)]
            bonus = self.kappa * math.sqrt(max(0.0, self.time - self.last_seen[s, a]))
            QLearningAgent.update(self, s, a, r + bonus, ns, d)
        return td


class PrioritizedSweepingAgent(QLearningAgent):
    name = "prioritized_sweeping"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
        self.predecessors: Dict[int, set[Tuple[int, int]]] = defaultdict(set)
        self.planning_steps = getattr(cfg, "planning_steps", 20)
        self.theta = getattr(cfg, "ps_theta", 1e-4)
        self.max_queue = getattr(cfg, "ps_max_queue", 2_000)
        self.queue: List[Tuple[float, int, int]] = []

    def _push_queue(self, priority: float, state: int, action: int) -> None:
        if len(self.queue) >= self.max_queue:
            self.queue = heapq.nsmallest(self.max_queue // 2, self.queue)
            heapq.heapify(self.queue)
        heapq.heappush(self.queue, (-priority, state, action))

    def _priority(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        target = reward if done else reward + self.cfg.gamma * np.max(self.q[next_state])
        return abs(target - self.q[state, action])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        priority = self._priority(state, action, reward, next_state, done)
        td = QLearningAgent.update(self, state, action, reward, next_state, done)
        self.model[(state, action)] = (reward, next_state, done)
        self.predecessors[next_state].add((state, action))
        if priority > self.theta:
            self._push_queue(priority, state, action)

        for _ in range(self.planning_steps):
            if not self.queue:
                break
            _, s, a = heapq.heappop(self.queue)
            r, ns, d = self.model[(s, a)]
            QLearningAgent.update(self, s, a, r, ns, d)
            for ps, pa in self.predecessors.get(s, set()):
                pr, pns, pdone = self.model[(ps, pa)]
                p = self._priority(ps, pa, pr, pns, pdone)
                if p > self.theta:
                    self._push_queue(p, ps, pa)
        return td


class SARSAAgent(BaseAgent):
    name = "sarsa"

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        state = env.reset()
        action = self.act(state)
        total_reward = 0.0
        last_td = 0.0
        for step in range(1, self.cfg.max_steps + 1):
            next_state, reward, done, _ = env.step(action)
            next_action = self.act(next_state) if not done else 0
            target = reward if done else reward + self.cfg.gamma * self.q[next_state, next_action]
            last_td = target - self.q[state, action]
            self.q[state, action] += self.cfg.alpha * last_td
            total_reward += reward
            state, action = next_state, next_action
            if done:
                break
        self.decay()
        return total_reward, step, abs(last_td)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        return 0.0


class ExpectedSARSAAgent(BaseAgent):
    name = "expected_sarsa"

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        if done:
            expected = 0.0
        else:
            best = self.best_action(next_state)
            probs = np.full(self.n_actions, self.epsilon / self.n_actions)
            probs[best] += 1.0 - self.epsilon
            expected = float(np.dot(probs, self.q[next_state]))
        target = reward + self.cfg.gamma * expected
        td = target - self.q[state, action]
        self.q[state, action] += self.cfg.alpha * td
        return float(td)


class DoubleQLearningAgent(BaseAgent):
    name = "double_q"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.q_a = np.zeros_like(self.q)
        self.q_b = np.zeros_like(self.q)

    @property
    def q(self) -> np.ndarray:  # type: ignore[override]
        return (self.q_a + self.q_b) / 2.0

    @q.setter
    def q(self, value: np.ndarray) -> None:  # used by BaseAgent during init
        self.q_a = value.copy()
        self.q_b = value.copy()

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        if self.rng.random() < 0.5:
            best = np.flatnonzero(np.isclose(self.q_a[next_state], np.max(self.q_a[next_state])))
            next_action = int(self.rng.choice(best))
            target = reward if done else reward + self.cfg.gamma * self.q_b[next_state, next_action]
            td = target - self.q_a[state, action]
            self.q_a[state, action] += self.cfg.alpha * td
        else:
            best = np.flatnonzero(np.isclose(self.q_b[next_state], np.max(self.q_b[next_state])))
            next_action = int(self.rng.choice(best))
            target = reward if done else reward + self.cfg.gamma * self.q_a[next_state, next_action]
            td = target - self.q_b[state, action]
            self.q_b[state, action] += self.cfg.alpha * td
        return float(td)


class NStepSARSAAgent(BaseAgent):
    name = "n_step_sarsa"

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        states: List[int] = [env.reset()]
        actions: List[int] = [self.act(states[0])]
        rewards: List[float] = [0.0]
        total_reward = 0.0
        terminal_t = math.inf
        last_td = 0.0
        t = 0
        while True:
            if t < terminal_t:
                next_state, reward, done, _ = env.step(actions[t])
                states.append(next_state)
                rewards.append(reward)
                total_reward += reward
                if done or t + 1 >= self.cfg.max_steps:
                    terminal_t = t + 1
                else:
                    actions.append(self.act(next_state))
            tau = t - self.cfg.n_step + 1
            if tau >= 0:
                upper = tau + self.cfg.n_step if math.isinf(terminal_t) else min(tau + self.cfg.n_step, int(terminal_t))
                g = sum((self.cfg.gamma ** (i - tau - 1)) * rewards[i] for i in range(tau + 1, upper + 1))
                if tau + self.cfg.n_step < terminal_t:
                    g += (self.cfg.gamma ** self.cfg.n_step) * self.q[states[tau + self.cfg.n_step], actions[tau + self.cfg.n_step]]
                last_td = g - self.q[states[tau], actions[tau]]
                self.q[states[tau], actions[tau]] += self.cfg.alpha * last_td
            if tau == terminal_t - 1:
                break
            t += 1
        self.decay()
        return total_reward, min(int(terminal_t), self.cfg.max_steps), abs(last_td)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        return 0.0


class SARSALambdaAgent(SARSAAgent):
    name = "sarsa_lambda"

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        traces = np.zeros_like(self.q)
        state = env.reset()
        action = self.act(state)
        total_reward = 0.0
        last_td = 0.0
        for step in range(1, self.cfg.max_steps + 1):
            next_state, reward, done, _ = env.step(action)
            next_action = self.act(next_state) if not done else 0
            target = reward if done else reward + self.cfg.gamma * self.q[next_state, next_action]
            last_td = target - self.q[state, action]
            traces[state, action] += 1.0
            self.q += self.cfg.alpha * last_td * traces
            traces *= self.cfg.gamma * self.cfg.lam
            total_reward += reward
            state, action = next_state, next_action
            if done:
                break
        self.decay()
        return total_reward, step, abs(last_td)


class QLambdaAgent(QLearningAgent):
    name = "q_lambda"

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        traces = np.zeros_like(self.q)
        state = env.reset()
        total_reward = 0.0
        last_td = 0.0
        for step in range(1, self.cfg.max_steps + 1):
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            greedy_next = self.best_action(next_state)
            target = reward if done else reward + self.cfg.gamma * self.q[next_state, greedy_next]
            last_td = target - self.q[state, action]
            traces[state, action] += 1.0
            self.q += self.cfg.alpha * last_td * traces
            if done or action != greedy_next:
                traces.fill(0.0)
            else:
                traces *= self.cfg.gamma * self.cfg.lam
            total_reward += reward
            state = next_state
            if done:
                break
        self.decay()
        return total_reward, step, abs(last_td)


class MonteCarloControlAgent(BaseAgent):
    name = "mc_control"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.returns_count = np.zeros_like(self.q)

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        episode: List[Tuple[int, int, float]] = []
        state = env.reset()
        total_reward = 0.0
        for step in range(1, self.cfg.max_steps + 1):
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            total_reward += reward
            state = next_state
            if done:
                break
        g = 0.0
        seen = set()
        last_delta = 0.0
        for state, action, reward in reversed(episode):
            g = reward + self.cfg.gamma * g
            if (state, action) in seen:
                continue
            seen.add((state, action))
            self.returns_count[state, action] += 1.0
            old = self.q[state, action]
            self.q[state, action] += (g - old) / self.returns_count[state, action]
            last_delta = self.q[state, action] - old
        self.decay()
        return total_reward, step, abs(last_delta)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        return 0.0


class PPOTabularAgent(BaseAgent):
    """Small tabular PPO baseline using torch when available."""

    name = "ppo_tabular"

    def __init__(self, n_states: int, n_actions: int, cfg: ExperimentConfig, seed: int) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("ppo_tabular requires torch. Install torch or omit this algorithm.") from exc
        self.torch = torch
        torch.manual_seed(seed)
        self.logits = torch.zeros((n_states, n_actions), requires_grad=True)
        self.values = torch.zeros(n_states, requires_grad=True)
        self.optim = torch.optim.Adam([self.logits, self.values], lr=max(cfg.alpha, 1e-4))
        self.clip = 0.2

    def act(self, state: int, greedy: bool = False) -> int:
        with self.torch.no_grad():
            probs = self.torch.softmax(self.logits[state], dim=-1)
            if greedy:
                return int(self.torch.argmax(probs).item())
            return int(self.torch.distributions.Categorical(probs).sample().item())

    @property
    def q(self) -> np.ndarray:  # type: ignore[override]
        with self.torch.no_grad():
            probs = self.torch.softmax(self.logits, dim=-1).cpu().numpy()
        return probs

    @q.setter
    def q(self, value: np.ndarray) -> None:
        self._q_placeholder = value

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        torch = self.torch
        states: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        old_log_probs: List[float] = []
        state = env.reset()
        total_reward = 0.0
        for step in range(1, self.cfg.max_steps + 1):
            probs = torch.softmax(self.logits[state], dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = int(dist.sample().item())
            old_log_probs.append(float(dist.log_prob(torch.tensor(action)).detach().item()))
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            if done:
                break

        returns: Deque[float] = deque()
        g = 0.0
        for reward in reversed(rewards):
            g = reward + self.cfg.gamma * g
            returns.appendleft(g)
        if not states:
            return 0.0, 0, 0.0

        states_t = torch.tensor(states, dtype=torch.long)
        actions_t = torch.tensor(actions, dtype=torch.long)
        returns_t = torch.tensor(list(returns), dtype=torch.float32)
        old_log_t = torch.tensor(old_log_probs, dtype=torch.float32)
        for _ in range(3):
            dist = torch.distributions.Categorical(logits=self.logits[states_t])
            log_probs = dist.log_prob(actions_t)
            value_pred = self.values[states_t]
            adv = returns_t - value_pred.detach()
            ratio = torch.exp(log_probs - old_log_t)
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = 0.5 * (returns_t - value_pred).pow(2).mean()
            entropy_bonus = dist.entropy().mean()
            loss = policy_loss + value_loss - 0.01 * entropy_bonus
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        return total_reward, step, float(loss.detach().abs().item())

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        return 0.0


class REINFORCETabularAgent(PPOTabularAgent):
    name = "reinforce_tabular"

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        torch = self.torch
        states: List[int] = []
        actions: List[int] = []
        rewards: List[float] = []
        state = env.reset()
        total_reward = 0.0
        for step in range(1, self.cfg.max_steps + 1):
            dist = torch.distributions.Categorical(logits=self.logits[state])
            action = int(dist.sample().item())
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            total_reward += reward
            state = next_state
            if done:
                break

        returns: Deque[float] = deque()
        g = 0.0
        for reward in reversed(rewards):
            g = reward + self.cfg.gamma * g
            returns.appendleft(g)
        if not states:
            return 0.0, 0, 0.0

        states_t = torch.tensor(states, dtype=torch.long)
        actions_t = torch.tensor(actions, dtype=torch.long)
        returns_t = torch.tensor(list(returns), dtype=torch.float32)
        if returns_t.numel() > 1 and float(returns_t.std()) > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
        dist = torch.distributions.Categorical(logits=self.logits[states_t])
        loss = -(dist.log_prob(actions_t) * returns_t).mean() - 0.01 * dist.entropy().mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return total_reward, step, float(loss.detach().abs().item())


class ActorCriticTabularAgent(PPOTabularAgent):
    name = "actor_critic_tabular"

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        torch = self.torch
        state = env.reset()
        total_reward = 0.0
        losses = []
        for step in range(1, self.cfg.max_steps + 1):
            dist = torch.distributions.Categorical(logits=self.logits[state])
            action = int(dist.sample().item())
            next_state, reward, done, _ = env.step(action)
            value = self.values[state]
            with torch.no_grad():
                bootstrap = 0.0 if done else float(self.values[next_state].item())
                target = reward + self.cfg.gamma * bootstrap
            target_t = torch.tensor(target, dtype=torch.float32)
            advantage = target_t - value
            policy_loss = -dist.log_prob(torch.tensor(action)) * advantage.detach()
            value_loss = 0.5 * advantage.pow(2)
            entropy_bonus = dist.entropy()
            loss = policy_loss + value_loss - 0.01 * entropy_bonus
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            losses.append(float(loss.detach().abs().item()))
            total_reward += reward
            state = next_state
            if done:
                break
        return total_reward, step, float(np.mean(losses)) if losses else 0.0


class PlanningAgent(BaseAgent):
    """Exact dynamic-programming oracle for known FrozenLake transition models."""

    name = "planning_base"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        cfg: ExperimentConfig,
        seed: int,
        env: NativeFrozenLakeEnv,
    ) -> None:
        super().__init__(n_states, n_actions, cfg, seed)
        self.env = env

    def train_episode(self, env: NativeFrozenLakeEnv) -> Tuple[float, int, float]:
        return 0.0, 0, 0.0

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> float:
        return 0.0


class ValueIterationAgent(PlanningAgent):
    name = "value_iteration"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        cfg: ExperimentConfig,
        seed: int,
        env: NativeFrozenLakeEnv,
    ) -> None:
        super().__init__(n_states, n_actions, cfg, seed, env)
        self.solve()

    def solve(self) -> None:
        values = np.zeros(self.n_states, dtype=np.float64)
        for _ in range(2_000):
            old = values.copy()
            for state in range(self.n_states):
                if state in self.env.terminal_states:
                    continue
                action_values = []
                for action in range(self.n_actions):
                    q = sum(
                        prob * (reward + self.cfg.gamma * old[next_state] * (not done))
                        for prob, next_state, reward, done in self.env.transition_outcomes(state, action)
                    )
                    action_values.append(q)
                values[state] = max(action_values)
            if np.max(np.abs(values - old)) < 1e-10:
                break
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.q[state, action] = sum(
                    prob * (reward + self.cfg.gamma * values[next_state] * (not done))
                    for prob, next_state, reward, done in self.env.transition_outcomes(state, action)
                )


class PolicyIterationAgent(PlanningAgent):
    name = "policy_iteration"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        cfg: ExperimentConfig,
        seed: int,
        env: NativeFrozenLakeEnv,
    ) -> None:
        super().__init__(n_states, n_actions, cfg, seed, env)
        self.solve()

    def solve(self) -> None:
        policy = np.zeros(self.n_states, dtype=np.int64)
        values = np.zeros(self.n_states, dtype=np.float64)
        for _ in range(200):
            for _ in range(1_000):
                old = values.copy()
                for state in range(self.n_states):
                    if state in self.env.terminal_states:
                        continue
                    action = int(policy[state])
                    values[state] = sum(
                        prob * (reward + self.cfg.gamma * old[next_state] * (not done))
                        for prob, next_state, reward, done in self.env.transition_outcomes(state, action)
                    )
                if np.max(np.abs(values - old)) < 1e-10:
                    break
            stable = True
            for state in range(self.n_states):
                if state in self.env.terminal_states:
                    continue
                old_action = policy[state]
                q_values = []
                for action in range(self.n_actions):
                    q_values.append(
                        sum(
                            prob * (reward + self.cfg.gamma * values[next_state] * (not done))
                            for prob, next_state, reward, done in self.env.transition_outcomes(state, action)
                        )
                    )
                policy[state] = int(np.argmax(q_values))
                stable = stable and old_action == policy[state]
            if stable:
                break
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.q[state, action] = sum(
                    prob * (reward + self.cfg.gamma * values[next_state] * (not done))
                    for prob, next_state, reward, done in self.env.transition_outcomes(state, action)
                )


AGENTS = {
    cls.name: cls
    for cls in [
        RandomPolicyAgent,
        QLearningAgent,
        OptimisticQLearningAgent,
        SoftmaxQLearningAgent,
        UCBQLearningAgent,
        DynaQAgent,
        DynaQPlusAgent,
        PrioritizedSweepingAgent,
        SARSAAgent,
        ExpectedSARSAAgent,
        DoubleQLearningAgent,
        NStepSARSAAgent,
        SARSALambdaAgent,
        QLambdaAgent,
        MonteCarloControlAgent,
        PPOTabularAgent,
        REINFORCETabularAgent,
        ActorCriticTabularAgent,
    ]
}

PLANNING_AGENTS = {ValueIterationAgent.name: ValueIterationAgent, PolicyIterationAgent.name: PolicyIterationAgent}
ALL_ALGORITHMS = {**AGENTS, **PLANNING_AGENTS}


def parse_list(text: str, cast=str) -> List:
    return [cast(part.strip()) for part in text.split(",") if part.strip()]


def ensure_dirs(root: Path) -> Dict[str, Path]:
    dirs = {
        "root": root,
        "figures": root / "figures",
        "tables": root / "tables",
        "logs": root / "logs",
        "optuna": root / "optuna",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def evaluate(agent: BaseAgent, spec: EnvSpec, seed: int, cfg: ExperimentConfig) -> float:
    successes = 0
    env = NativeFrozenLakeEnv(spec.map_size, spec.slippery, seed, spec.frozen_prob)
    for ep in range(cfg.eval_episodes):
        state = env.reset(seed + ep)
        for _ in range(cfg.max_steps):
            action = agent.act(state, greedy=True)
            state, reward, done, _ = env.step(action)
            if done:
                successes += int(reward > 0)
                break
    return successes / cfg.eval_episodes


def train_run(algorithm: str, spec: EnvSpec, seed: int, cfg: ExperimentConfig) -> Dict[str, object]:
    env = NativeFrozenLakeEnv(spec.map_size, spec.slippery, seed, spec.frozen_prob)
    if algorithm in PLANNING_AGENTS:
        agent = PLANNING_AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed, env)
    else:
        agent = AGENTS[algorithm](env.n_states, env.n_actions, cfg, seed)
    train_rows = []
    eval_rows = []
    if algorithm in PLANNING_AGENTS:
        success = evaluate(agent, spec, seed + 100_000, cfg)
        eval_rows.append(
            {
                "episode": 0,
                "algorithm": algorithm,
                "env": spec.label,
                "seed": seed,
                "success_rate": success,
            }
        )
        train_rows.append(
            {
                "episode": 0,
                "algorithm": algorithm,
                "env": spec.label,
                "seed": seed,
                "reward": success,
                "steps": 0,
                "epsilon": 0.0,
                "td_abs": 0.0,
            }
        )
        return {
            "algorithm": algorithm,
            "env": spec.label,
            "seed": seed,
            "map_size": spec.map_size,
            "slippery": spec.slippery,
            "q": np.asarray(agent.q, dtype=np.float64),
            "train_rows": train_rows,
            "eval_rows": eval_rows,
        }
    for episode in range(1, cfg.episodes + 1):
        reward, steps, td_abs = agent.train_episode(env)
        train_rows.append(
            {
                "episode": episode,
                "algorithm": algorithm,
                "env": spec.label,
                "seed": seed,
                "reward": reward,
                "steps": steps,
                "epsilon": getattr(agent, "epsilon", math.nan),
                "td_abs": td_abs,
            }
        )
        if episode % cfg.eval_interval == 0 or episode == cfg.episodes:
            eval_rows.append(
                {
                    "episode": episode,
                    "algorithm": algorithm,
                    "env": spec.label,
                    "seed": seed,
                    "success_rate": evaluate(agent, spec, seed + 100_000 + episode, cfg),
                }
            )
    return {
        "algorithm": algorithm,
        "env": spec.label,
        "seed": seed,
        "map_size": spec.map_size,
        "slippery": spec.slippery,
        "q": np.asarray(agent.q, dtype=np.float64),
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }


def summarize_runs(runs: Sequence[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for run in runs:
        eval_rows = run["eval_rows"]
        train_rows = run["train_rows"]
        final_success = float(eval_rows[-1]["success_rate"])
        best_success = float(max(row["success_rate"] for row in eval_rows))
        y_values = [row["success_rate"] for row in eval_rows]
        x_values = [row["episode"] for row in eval_rows]
        auc_fn = getattr(np, "trapezoid", np.trapz)
        auc = float(auc_fn(y_values, x_values))
        rows.append(
            {
                "algorithm": run["algorithm"],
                "env": run["env"],
                "seed": run["seed"],
                "final_success": final_success,
                "best_success": best_success,
                "eval_auc": auc,
                "mean_train_reward": float(np.mean([row["reward"] for row in train_rows])),
                "mean_steps": float(np.mean([row["steps"] for row in train_rows])),
            }
        )
    return pd.DataFrame(rows)


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = summary.groupby(["algorithm", "env"], as_index=False)
    return grouped.agg(
        mean_final_success=("final_success", "mean"),
        std_final_success=("final_success", "std"),
        mean_best_success=("best_success", "mean"),
        mean_eval_auc=("eval_auc", "mean"),
        mean_train_reward=("mean_train_reward", "mean"),
        mean_steps=("mean_steps", "mean"),
        n_seeds=("seed", "count"),
    )


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_eval_curves(all_eval_rows: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=all_eval_rows,
        x="episode",
        y="success_rate",
        hue="algorithm",
        col="env",
        kind="line",
        errorbar="sd",
        col_wrap=2,
        height=4,
        facet_kws={"sharey": True, "sharex": False},
    )
    g.set(ylim=(0.0, 1.05))
    g.set_axis_labels("Training episode", "Greedy success rate")
    g.fig.tight_layout()
    g.fig.savefig(path, dpi=200)
    plt.close(g.fig)


def plot_final_bars(summary: pd.DataFrame, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(10, len(summary["algorithm"].unique()) * 1.2), 6))
    sns.barplot(data=summary, x="env", y="final_success", hue="algorithm", errorbar="sd", ax=ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Final evaluation success rate")
    ax.set_xlabel("Environment")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_heatmap(aggregate: pd.DataFrame, path: Path) -> None:
    pivot = aggregate.pivot(index="algorithm", columns="env", values="mean_final_success")
    fig, ax = plt.subplots(figsize=(max(7, pivot.shape[1] * 1.4), max(5, pivot.shape[0] * 0.45)))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis", vmin=0.0, vmax=1.0, ax=ax)
    ax.set_title("Mean final success rate")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_boxplot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(10, len(summary["env"].unique()) * 1.6), 6))
    sns.boxplot(data=summary, x="env", y="final_success", hue="algorithm", ax=ax)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Final success by seed")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> None:
    cfg = ExperimentConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    dirs = ensure_dirs(cfg.output_dir)
    algorithms = parse_list(args.algorithms)
    map_sizes = parse_list(args.maps, int)
    slippery_values = [False, True] if args.slippery == "both" else [args.slippery == "true"]
    specs = [EnvSpec(size, slippery, args.frozen_prob) for size in map_sizes for slippery in slippery_values]

    unknown = sorted(set(algorithms) - set(ALL_ALGORITHMS))
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}. Available: {sorted(ALL_ALGORITHMS)}")

    runs = []
    for spec in specs:
        for algorithm in algorithms:
            for seed in range(cfg.seeds):
                try:
                    print(f"Running {algorithm} on {spec.label}, seed={seed}")
                    run = train_run(algorithm, spec, seed, cfg)
                except RuntimeError as exc:
                    print(f"Skipping {algorithm}: {exc}")
                    continue
                runs.append(run)
                stem = f"{algorithm}_{spec.label}_seed{seed}"
                write_rows(dirs["logs"] / f"{stem}_train.csv", run["train_rows"])
                write_rows(dirs["logs"] / f"{stem}_eval.csv", run["eval_rows"])

    summary = summarize_runs(runs)
    aggregate = aggregate_summary(summary)
    all_eval = pd.DataFrame([row for run in runs for row in run["eval_rows"]])
    summary.to_csv(dirs["tables"] / "enhanced_run_summary.csv", index=False)
    aggregate.to_csv(dirs["tables"] / "enhanced_aggregate_summary.csv", index=False)
    all_eval.to_csv(dirs["tables"] / "enhanced_eval_curves.csv", index=False)

    if not all_eval.empty:
        plot_eval_curves(all_eval, dirs["figures"] / "algorithm_eval_curves.png")
        plot_final_bars(summary, dirs["figures"] / "final_success_bar.png")
        plot_boxplot(summary, dirs["figures"] / "final_success_boxplot.png")
        plot_heatmap(aggregate, dirs["figures"] / "algorithm_env_heatmap.png")

    print(f"Saved enhanced benchmark to {dirs['root'].resolve()}")


def optuna_objective(
    trial,
    algorithm: str,
    spec: EnvSpec,
    base_cfg: ExperimentConfig,
    seed_count: int,
) -> float:
    cfg = replace(
        base_cfg,
        alpha=trial.suggest_float("alpha", 0.01, 0.8, log=True),
        gamma=trial.suggest_float("gamma", 0.90, 0.999),
        epsilon_start=trial.suggest_float("epsilon_start", 0.2, 1.0),
        epsilon_min=trial.suggest_float("epsilon_min", 0.001, 0.10, log=True),
        epsilon_decay=trial.suggest_float("epsilon_decay", 0.990, 0.9998),
        lam=trial.suggest_float("lambda", 0.2, 0.95),
        n_step=trial.suggest_int("n_step", 2, 8),
    )
    cfg.optimistic_init = trial.suggest_float("optimistic_init", 0.1, 2.0)
    cfg.planning_steps = trial.suggest_int("planning_steps", 1, 30)
    scores = []
    for seed in range(seed_count):
        run = train_run(algorithm, spec, seed, cfg)
        scores.append(float(run["eval_rows"][-1]["success_rate"]))
    return float(np.mean(scores))


def run_optuna(args: argparse.Namespace) -> None:
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Optuna is not installed.") from exc

    cfg = ExperimentConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        seeds=args.seeds,
        output_dir=args.output_dir,
    )
    dirs = ensure_dirs(cfg.output_dir)
    algorithms = parse_list(args.algorithms)
    spec = EnvSpec(args.optuna_map, args.optuna_slippery, args.frozen_prob)
    records = []
    for algorithm in algorithms:
        pruner = optuna.pruners.HyperbandPruner() if args.pruner == "hyperband" else optuna.pruners.MedianPruner()
        sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
        study.optimize(
            lambda trial: optuna_objective(trial, algorithm, spec, cfg, args.optuna_seeds),
            n_trials=args.trials,
            show_progress_bar=False,
        )
        out = {
            "algorithm": algorithm,
            "env": spec.label,
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        records.append(out)
        (dirs["optuna"] / f"{algorithm}_{spec.label}_best_params.json").write_text(
            json.dumps(out, indent=2),
            encoding="utf-8",
        )
        trials_df = study.trials_dataframe()
        trials_df.to_csv(dirs["optuna"] / f"{algorithm}_{spec.label}_trials.csv", index=False)
        fig = optuna.visualization.matplotlib.plot_optimization_history(study).figure
        fig.tight_layout()
        fig.savefig(dirs["optuna"] / f"{algorithm}_{spec.label}_history.png", dpi=200)
        plt.close(fig)
    (dirs["optuna"] / f"{spec.label}_optuna_summary.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    print(f"Saved Optuna studies to {dirs['optuna'].resolve()}")


def write_research_notes(output_dir: Path) -> None:
    notes_dir = output_dir / "research_reproduction"
    notes_dir.mkdir(parents=True, exist_ok=True)
    text = """# FrozenLake Research and Reproduction Notes

## Sources checked

- Gymnasium FrozenLake Q-learning tutorial: https://gymnasium.farama.org/tutorials/training_agents/frozenlake_q_learning/
- Gymnasium FrozenLake benchmark tutorial: https://gymnasium.farama.org/v0.28.0/tutorials/training_agents/FrozenLake_tuto/
- Double Q-learning, van Hasselt, NeurIPS 2010: https://papers.nips.cc/paper/3964-double-q-learning
- PPO, Schulman et al., arXiv:1707.06347: https://arxiv.org/abs/1707.06347
- Sutton and Barto, Reinforcement Learning: An Introduction: http://incompleteideas.net/book/the-book-2nd.html

## Reproduction target

Use the Gymnasium tutorial as the public baseline because it explicitly studies
FrozenLake with different map sizes and tabular Q-learning. The enhanced script
reproduces the same core setup locally without requiring Gymnasium: tabular
Q-learning, epsilon-greedy exploration, repeated runs, and map-size comparison.

## What to claim carefully

- FrozenLake is small and tabular methods are strong baselines.
- Larger maps and slippery transitions make exploration much harder because
  sparse rewards become rarer and the same intended action has noisy outcomes.
- Expected SARSA and SARSA variants can be more conservative under stochastic
  dynamics because they update using the behavior policy, while Q-learning uses
  a max target that may overestimate noisy action values.
- Double Q-learning is specifically motivated by Q-learning overestimation.
- PPO is included as an advanced policy-gradient contrast, but tabular methods
  may still win on small discrete MDPs because they are more sample efficient.
"""
    (notes_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced FrozenLake RL benchmark.")
    parser.add_argument("--mode", choices=["benchmark", "optuna", "research-notes"], default="benchmark")
    parser.add_argument(
        "--algorithms",
        default="q_learning,sarsa,expected_sarsa,double_q,n_step_sarsa,sarsa_lambda,q_lambda,mc_control,dyna_q,ucb_q,softmax_q,optimistic_q",
    )
    parser.add_argument("--maps", default="4,8,16")
    parser.add_argument("--slippery", choices=["true", "false", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--frozen-prob", type=float, default=0.85)
    parser.add_argument("--output-dir", type=Path, default=Path("enhanced_results"))
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--optuna-map", type=int, default=8)
    parser.add_argument("--optuna-slippery", action="store_true")
    parser.add_argument("--optuna-seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pruner", choices=["hyperband", "median"], default="hyperband")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "benchmark":
        run_benchmark(args)
    elif args.mode == "optuna":
        run_optuna(args)
    else:
        write_research_notes(args.output_dir)
        print(f"Saved research notes to {(args.output_dir / 'research_reproduction').resolve()}")


if __name__ == "__main__":
    main()
