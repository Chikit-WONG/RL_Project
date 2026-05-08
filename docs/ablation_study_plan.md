# Ablation Study Plan

当前正式 ablation 只服务于 stochastic FrozenLake 主线，重点解释 random 8x8 中算法差异的来源。

## 1. Exploration Ablation

目的：解释 Optimistic Q-learning 为什么在 8x8 random maps 上表现最好。

对比项：

- 普通 epsilon-greedy Q-learning
- Optimistic Q-learning
- 不同 epsilon decay / epsilon minimum

预期解释：FrozenLake 奖励稀疏，很多失败不是更新公式本身不行，而是 agent 没有足够探索到可行路径。乐观初始化能让未知动作保持吸引力，因此更容易穿过稀疏奖励瓶颈。

对应图表：

- `figures/main/exploration_ablation_fixed_8x8.png`
- `results_csv/exploration_ablation_fixed_8x8.csv`

## 2. Reward Propagation Ablation

目的：解释 Dyna-Q、SARSA(lambda)、Q(lambda) 的价值。

对比项：

- one-step TD methods
- eligibility trace methods
- planning-based Dyna-Q

预期解释：FrozenLake 成功奖励只在终点出现，one-step 方法需要很多 episode 才能把奖励传播回起点。trace 和 planning 可以加速传播，因此在 AUC 或早期学习速度上更有优势。

对应图表：

- `figures/main/reward_propagation_ablation_fixed_8x8.png`
- `results_csv/reward_propagation_ablation_fixed_8x8.csv`

## 3. Map Difficulty Ablation

目的：解释为什么同一 frozen tile probability 下方差仍然很大。

对比项：

- 标准 4x4
- designed 4x4 exact-hole maps
- random 8x8 不同 p 与不同 map id
- Value Iteration oracle gap

预期解释：`p` 只控制洞的大致比例，不直接控制路径结构、拐点、瓶颈和 slippery 后的风险。因此必须按每张地图单独画 learning curve，并用 oracle gap 区分“地图难”和“算法没学好”。

对应图表：

- `figures/main/random8_learning_curves_all_maps_panel.png`
- `figures/main/random8_oracle_gap_heatmap_by_map.png`
- `figures/main/random8_final_success_points_by_probability.png`
