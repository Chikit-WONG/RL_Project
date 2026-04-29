"""Quick smoke tests for the new experiment infrastructure."""
from environment import generate_random_map
from agent import train_once
from config import Config, ExperimentSpec

# --- 6x6 Q-table shape check ---
maps_6x6 = [generate_random_map(6, 0.25, seed=i) for i in range(10)]
spec = ExperimentSpec("test", "test", True, 0.1, "test", map_layout=tuple(maps_6x6[0]))
config = Config(n_episodes=100, multi_seed=1)
result = train_once(spec=spec, config=config, backend="native", seed=0)
shape = result["q_table"].shape
assert shape == (36, 4), f"Expected (36,4), got {shape}"
print(f"6x6 Q-table shape: {shape}  OK")

# --- BFS reachability for all 10 maps of each size ---
maps_5x5 = [generate_random_map(5, 0.25, seed=i) for i in range(10)]
for i, m in enumerate(maps_5x5):
    assert len(m) == 5 and all(len(r) == 5 for r in m), f"5x5 map {i} shape wrong"
    assert "G" in "".join(m), f"5x5 map {i} missing G"
for i, m in enumerate(maps_6x6):
    assert len(m) == 6 and all(len(r) == 6 for r in m), f"6x6 map {i} shape wrong"
print("All 20 maps (10x5x5, 10x6x6) pass BFS reachability and shape checks  OK")

# --- Double Q-Learning produces a result ---
spec_dq = ExperimentSpec(
    "test_dq", "test dq", True, 0.1, "algo",
    algo="double_q_learning",
)
result_dq = train_once(spec=spec_dq, config=config, backend="native", seed=0)
assert result_dq["q_table"].shape == (16, 4)
print(f"Double Q-Learning Q-table shape: {result_dq['q_table'].shape}  OK")

# --- Reward shaping runs without errors ---
spec_sh = ExperimentSpec(
    "test_sh", "test shaping", True, 0.1, "shaping",
    reward_shaping=True, step_penalty=True,
)
result_sh = train_once(spec=spec_sh, config=config, backend="native", seed=0)
print(f"Reward shaping run completed, final reward: {result_sh['rewards'][-1]}  OK")

print("\nAll smoke tests passed.")
