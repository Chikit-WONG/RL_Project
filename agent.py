"""Q-table, exploration policies, and training/evaluation routines."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from config import Config, ExperimentSpec
from environment import make_env


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


# ---------------------------------------------------------------------------
# Double Q-table
# ---------------------------------------------------------------------------

class DoubleQTable:
    """Two Q-tables for Double Q-Learning — eliminates the maximisation bias."""

    def __init__(self, n_states: int, n_actions: int) -> None:
        self.qa = np.zeros((n_states, n_actions), dtype=np.float64)
        self.qb = np.zeros((n_states, n_actions), dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        """Averaged Q-table used for greedy action selection and logging."""
        return (self.qa + self.qb) / 2.0

    def max_q(self, state: int) -> float:
        return float(np.max(self.values[state]))

    def best_action(self, state: int, rng: np.random.Generator) -> int:
        avg = self.values[state]
        best_val = np.max(avg)
        best_actions = np.flatnonzero(np.isclose(avg, best_val))
        return int(rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        alpha: float,
        gamma: float,
        rng: np.random.Generator,
    ) -> None:
        """One Double Q-Learning update: randomly choose which table to update."""
        if rng.random() < 0.5:
            best_a = int(np.argmax(self.qa[next_state]))
            td_target = reward if done else reward + gamma * float(self.qb[next_state, best_a])
            self.qa[state, action] += alpha * (td_target - self.qa[state, action])
        else:
            best_a = int(np.argmax(self.qb[next_state]))
            td_target = reward if done else reward + gamma * float(self.qa[next_state, best_a])
            self.qb[state, action] += alpha * (td_target - self.qb[state, action])


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

    def select_action(self, q_table: object, state: int) -> int:
        """Select a random or greedy action according to epsilon."""

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, q_table.values.shape[1]))
        return q_table.best_action(state, self.rng)

    def greedy_action(self, q_table: object, state: int) -> int:
        """Select the greedy action without additional exploration."""

        return q_table.best_action(state, self.rng)

    def decay(self) -> None:
        """Decay epsilon after each episode."""

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ---------------------------------------------------------------------------
# Additional exploration policies (Phase 1)
# ---------------------------------------------------------------------------

class RandomPolicy:
    """Pure random action selection — lower-bound exploration baseline."""

    def __init__(self, n_actions: int, rng: np.random.Generator) -> None:
        self.n_actions = n_actions
        self.rng = rng
        self.epsilon = 1.0  # constant; kept for log compatibility

    def select_action(self, q_table: object, state: int) -> int:
        return int(self.rng.integers(0, self.n_actions))

    def greedy_action(self, q_table: object, state: int) -> int:
        return q_table.best_action(state, self.rng)

    def decay(self) -> None:
        pass


class UCBQPolicy:
    """Upper Confidence Bound (UCB1) action selection.

    a = argmax_a [ Q(s,a) + c * sqrt( ln(sum_a' N(s,a')) / (N(s,a)+1) ) ]
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        c: float = 2.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.counts = np.zeros((n_states, n_actions), dtype=np.float64)
        self.c = c
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = 0.0  # unused; kept for log compatibility

    def select_action(self, q_table: object, state: int) -> int:
        q_vals = q_table.values[state]
        total = float(np.sum(self.counts[state]))
        if total == 0.0:
            action = int(self.rng.integers(0, len(q_vals)))
        else:
            ln_total = float(np.log(total + 1.0))
            ucb = q_vals + self.c * np.sqrt(ln_total / (self.counts[state] + 1.0))
            action = int(np.argmax(ucb))
        self.counts[state, action] += 1.0
        return action

    def greedy_action(self, q_table: object, state: int) -> int:
        return q_table.best_action(state, self.rng)

    def decay(self) -> None:
        pass


class BoltzmannPolicy:
    """Softmax / Boltzmann exploration with temperature decay."""

    def __init__(
        self,
        temp_start: float = 1.0,
        temp_min: float = 0.01,
        temp_decay: float = 0.995,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.temperature = temp_start
        self.temp_min = temp_min
        self.temp_decay = temp_decay
        self.rng = rng if rng is not None else np.random.default_rng()
        self.epsilon = temp_start  # proxy for log compatibility

    def select_action(self, q_table: object, state: int) -> int:
        q_vals = q_table.values[state]
        scaled = (q_vals - float(np.max(q_vals))) / max(self.temperature, 1e-8)
        exp_vals = np.exp(scaled)
        probs = exp_vals / float(np.sum(exp_vals))
        return int(self.rng.choice(len(probs), p=probs))

    def greedy_action(self, q_table: object, state: int) -> int:
        return q_table.best_action(state, self.rng)

    def decay(self) -> None:
        self.temperature = max(self.temp_min, self.temperature * self.temp_decay)
        self.epsilon = self.temperature


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


# ---------------------------------------------------------------------------
# Potential-based reward shaping helper
# ---------------------------------------------------------------------------

def _manhattan_potential(state: int, goal_state: int, ncol: int, n_states: int) -> float:
    """Negative normalised Manhattan distance to the goal (higher = closer to goal)."""
    gr, gc = divmod(goal_state, ncol)
    sr, sc = divmod(state, ncol)
    dist = abs(sr - gr) + abs(sc - gc)
    return -dist / float(n_states)


def evaluate_policy(
    q_table: object,
    backend: str,
    is_slippery: bool,
    max_steps: int,
    eval_episodes: int,
    seed: int,
    map_layout=None,
) -> float:
    """Evaluate the greedy policy and return the success rate."""

    ml = list(map_layout) if map_layout is not None else None
    env, _ = make_env(backend=backend, is_slippery=is_slippery, seed=seed, map_layout=ml)
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

    from utils import smooth_curve  # local import to avoid circular dependency

    map_layout = list(spec.map_layout) if spec.map_layout is not None else None
    env, backend_used = make_env(backend=backend, is_slippery=spec.is_slippery, seed=seed, map_layout=map_layout)
    rng = np.random.default_rng(seed)

    # --- Q-table ---------------------------------------------------------
    if spec.algo == "double_q_learning":
        q_table: object = DoubleQTable(n_states=env.n_states, n_actions=env.n_actions)
    else:
        q_table = QTable(n_states=env.n_states, n_actions=env.n_actions)

    # --- Exploration policy ----------------------------------------------
    if spec.policy_type == "random":
        policy: object = RandomPolicy(n_actions=env.n_actions, rng=rng)
    elif spec.policy_type == "ucb":
        policy = UCBQPolicy(n_states=env.n_states, n_actions=env.n_actions, c=2.0, rng=rng)
    elif spec.policy_type == "boltzmann":
        policy = BoltzmannPolicy(
            temp_start=config.epsilon_start,
            temp_min=config.epsilon_min,
            temp_decay=config.epsilon_decay,
            rng=rng,
        )
    else:  # epsilon_greedy (default)
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

                # Reward shaping (potential-based + optional step penalty)
                shaped_reward = reward
                if spec.reward_shaping:
                    phi_s = _manhattan_potential(state, env.goal_state, env.ncol, env.n_states)
                    phi_s_prime = _manhattan_potential(next_state, env.goal_state, env.ncol, env.n_states)
                    shaped_reward += config.gamma * phi_s_prime - phi_s
                if spec.step_penalty:
                    shaped_reward += -0.01

                # Q-table update
                if spec.algo == "double_q_learning":
                    q_table.update(state, action, shaped_reward, next_state, done, spec.alpha, config.gamma, rng)
                else:
                    q_learning_update(
                        q_table=q_table,
                        state=state,
                        action=action,
                        reward=shaped_reward,
                        next_state=next_state,
                        done=done,
                        alpha=spec.alpha,
                        gamma=config.gamma,
                    )
                total_reward += reward  # log raw environment reward
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
                    map_layout=spec.map_layout,
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
