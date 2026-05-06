# Algorithm QA Notes

本文用于答辩或代码说明时解释各算法的表现。所有讨论默认指 `slippery=True` 的 stochastic FrozenLake。

## Q-learning

Q-learning 是 off-policy TD control，更新目标使用 `max_a Q(s', a)`。

优势：

- 简单、经典、适合作为 baseline。
- 在 4x4 和部分 8x8 地图上表现强。
- 长预算后 random 8x8 mean final success 约为 0.879。

局限：

- stochastic transition 下，max target 可能放大偶然高估。
- 稀疏奖励下，如果探索不足，容易长期找不到有效路径。

答辩要点：Q-learning 很强，但它的瓶颈主要是探索与过估计，不是更新公式完全不适合。

## SARSA

SARSA 是 on-policy TD control，使用实际执行的下一步动作更新。

优势：

- 学习目标与行为策略一致。
- 在 slippery 环境下相对保守，长预算后表现稳定。
- random 8x8 mean final success 约为 0.895。

局限：

- 学习速度受 epsilon schedule 影响较大。
- 探索期动作会进入更新目标，因此前期可能慢。

答辩要点：SARSA 的最终表现并不差，但需要足够训练预算。

## Expected SARSA

Expected SARSA 使用 epsilon-greedy 策略下的期望动作价值更新。

优势：

- 比 SARSA 的单样本 next action 更新方差更低。
- 比 Q-learning 的 max target 更保守。
- random 8x8 mean final success 约为 0.902。

局限：

- 对 epsilon schedule 仍然敏感。
- 短预算下容易被低估。

答辩要点：Expected SARSA 适合 stochastic FrozenLake，因为它显式考虑当前行为策略下的期望。

## SARSA(lambda)

SARSA(lambda) 使用 eligibility traces 加速奖励传播。

优势：

- 成功奖励稀疏时，可以更快把终点奖励传播回早期状态。
- random 8x8 mean AUC 约为 0.725，说明训练过程有一定优势。

局限：

- 对 `lambda`、学习率和探索策略敏感。
- 某些地图上 trace 可能放大不稳定更新，导致 final success 偏低。

答辩要点：它展示了 reward propagation 的价值，但不是本项目中最稳定的最终策略。

## Q(lambda)

Q(lambda) 将 Q-learning 与 eligibility traces 结合。

优势：

- 比 one-step Q-learning 更快传播奖励。
- random 8x8 mean final success 约为 0.873。

局限：

- off-policy 方法与 traces 结合时更敏感。
- stochastic 环境中稳定性不如 Optimistic Q。

答辩要点：这是重要传统变种，但在本实验中没有超过探索更强的 Optimistic Q-learning。

## Dyna-Q

Dyna-Q 学习一个经验模型，并用 planning steps 做模拟更新。

优势：

- early learning 和 AUC 表现好。
- random 8x8 mean AUC 约为 0.808，高于 Q-learning、SARSA 和 Expected SARSA。
- 适合说明 planning 对稀疏奖励传播的帮助。

局限：

- 简单经验模型不能完美表达 slippery transition 的随机性。
- final success 不是最高，random 8x8 mean final success 约为 0.870。

答辩要点：Dyna-Q 的价值主要是 sample efficiency，不一定是最终策略最优。

## Optimistic Q-learning

Optimistic Q-learning 使用乐观初始化鼓励探索。

优势：

- 在稀疏奖励 FrozenLake 中非常有效。
- 未探索动作初始价值高，agent 更愿意尝试不同路径。
- random 8x8 mean final success 约为 0.988，mean oracle gap 约为 0.002。

局限：

- 乐观初始值需要调参。
- 环境更大或奖励结构变化时，可能需要重新设置。

答辩要点：它表现最好不是因为更新公式复杂，而是因为它最有效地缓解了探索瓶颈。

## Value Iteration Oracle

Value Iteration 使用已知环境模型求近似最优策略。

用途：

- 提供 oracle success，作为上限参考。
- 帮助区分地图本身难，还是算法没学好。

答辩要点：如果 oracle 高但 learned success 低，说明算法学习不足；如果 oracle 本身低，说明该 slippery map 即使最优策略也很难稳定成功。
