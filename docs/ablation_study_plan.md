# Ablation Study 说明

本项目中的 ablation study 主要用于解释两个问题：

1. 为什么 Optimistic Q-learning 表现最好？
2. 为什么 Dyna-Q 的 final success 不是最高，但训练过程表现仍然有价值？

## Exploration Ablation

对应文件：

```text
figures/main/exploration_ablation_fixed_8x8.png
results_csv/exploration_ablation_fixed_8x8.csv
```

研究问题：在 slippery 8x8 FrozenLake 中，探索策略对学习结果有多大影响？

核心解释：

- FrozenLake 奖励非常稀疏，agent 很长时间可能完全拿不到成功奖励。
- 普通 epsilon-greedy Q-learning 会随着 epsilon 衰减逐渐减少探索。
- Optimistic initialization 让未知动作一开始看起来更有价值，因此更愿意探索不同路径。

结论：在本项目中，Optimistic Q-learning 的优势主要来自更有效的探索，而不是更复杂的 TD 更新公式。

## Reward Propagation Ablation

对应文件：

```text
figures/main/reward_propagation_ablation_fixed_8x8.png
results_csv/reward_propagation_ablation_fixed_8x8.csv
```

研究问题：在稀疏奖励环境中，如何更快把终点奖励传播回早期状态？

对比思路：

- one-step TD 方法每次只传播一步。
- eligibility traces 可以把奖励传播到近期访问过的一串状态动作。
- Dyna-Q 可以通过模型 replay 做额外 planning updates。

结论：Dyna-Q 的 final success 不是最高，但 AUC 通常较好，说明它更早学到有效策略。这个结果支持“planning 提高 sample efficiency”的解释。

## Map Difficulty Ablation

对应文件：

```text
figures/main/random8_learning_curves_all_maps_panel.png
figures/main/random8_oracle_gap_heatmap_by_map.png
figures/main/random8_final_success_points_by_probability.png
```

研究问题：为什么同一个 frozen tile probability 下，结果方差仍然很大？

核心解释：

- `p` 只控制 frozen tile 的概率，不直接控制路径长度、瓶颈位置、洞的局部结构。
- slippery transition 会让路径附近的洞产生额外风险。
- 因此同一 p 下不同地图仍可能差异很大。

结论：不能只看平均曲线。per-map learning curves 和 oracle gap 更能说明算法失败来自地图难度还是学习不足。
