# 实验路线说明

本文记录本 GitHub 上传版对应的实验路线。它不是完整历史备份，而是当前正式主线的整理版。

## 当前主线

正式主线聚焦：

- `slippery=True`
- tabular RL algorithms
- 标准 4x4
- designed 4x4
- random 8x8
- long-budget learning curves
- ablation study

不作为当前主线：

- deterministic FrozenLake
- DQN/PPO exploratory results
- 16x16 stress tests
- 早期短预算 random-map 结果

## 实验阶段

### Stage 1: Baseline

从标准 4x4 FrozenLake 的 Q-learning baseline 开始，验证环境、训练循环和评估流程。

### Stage 2: 多算法对比

加入传统 tabular RL 方法：

- Q-learning
- SARSA
- Expected SARSA
- SARSA(lambda)
- Q(lambda)
- Dyna-Q
- Optimistic Q-learning

### Stage 3: 地图难度扩展

扩展到：

- 标准 4x4
- designed 4x4 exact-hole maps
- random 8x8 maps

这样可以避免只在一个固定 4x4 map 上得出过强结论。

### Stage 4: 长预算重跑

早期短预算结果显示很多算法不收敛。为避免低估传统 tabular 方法，最终 random 8x8 使用：

```text
20 maps x 7 algorithms x 10000 episodes
```

Designed 4x4 使用：

```text
4 maps x 7 algorithms x 2000 episodes
```

### Stage 5: 图像与分析整理

最终报告使用：

- per-map learning curves
- final success heatmaps
- oracle gap heatmaps
- AUC heatmaps
- convergence episode heatmaps
- ablation figures

## 为什么保留 per-map 结果

FrozenLake random maps 的难度不仅由 frozen tile probability 决定，还由洞的位置、路径形状和 slippery 风险共同决定。把 20 张地图混成一条平均曲线会隐藏这些差异，因此本项目保留每张地图的曲线和 heatmap。
