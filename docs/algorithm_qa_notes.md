# Algorithm QA Notes

这些笔记用于答辩时解释每个算法为什么表现如此。当前正式讨论只针对 `slippery=True` 的 stochastic FrozenLake，重点是标准 4x4、designed 4x4 和 random 8x8 long-budget 结果。

## Q-learning

Q-learning 是 off-policy TD control，用 `max_a Q(s', a)` 更新。它是强 baseline，random 8x8 long-budget mean final success 约为 0.879。

优势：实现简单、收敛后策略强、适合作为 tabular RL 基线。

局限：在 stochastic 转移和稀疏奖励下，max target 容易放大偶然高估；探索不足时会卡在局部策略。

答辩说法：Q-learning 在 4x4 和部分 8x8 地图上很强，但 random-map 泛化不如 Optimistic Q 稳定。

## SARSA

SARSA 是 on-policy TD control，用实际下一步动作更新。random 8x8 long-budget mean final success 约为 0.895。

优势：更新目标与行为策略一致，在 slippery 环境中通常更保守。

局限：依赖 epsilon schedule，探索期策略会影响学习目标，所以前期学习较慢。

答辩说法：SARSA 的最终表现并不差，但它需要更长预算才能体现稳定性。

## Expected SARSA

Expected SARSA 用 epsilon-greedy 策略下的期望动作价值更新。random 8x8 long-budget mean final success 约为 0.902。

优势：比 SARSA 的单样本 next action 更新方差更低，也比 Q-learning 的 max target 更保守。

局限：如果 epsilon schedule 不合适，期望目标仍会被探索概率拖慢；短预算结果容易低估它。

答辩说法：它适合 stochastic FrozenLake，因为它显式考虑当前探索策略下的下一步期望。

## SARSA(lambda)

SARSA(lambda) 使用 eligibility traces 加速 reward propagation。random 8x8 long-budget mean final success 约为 0.783，mean AUC 约为 0.725。

优势：稀疏奖励下能把成功回报更快传播到早期状态。

局限：对 lambda、alpha 和探索策略敏感；某些地图上 trace 会放大不稳定更新。

答辩说法：它体现了 reward propagation 的价值，但不是当前最稳定的最终策略。

## Q(lambda)

Q(lambda) 把 Q-learning 与 eligibility traces 结合。random 8x8 long-budget mean final success 约为 0.873。

优势：比 one-step Q-learning 更快传播奖励。

局限：off-policy 与 trace 结合时更敏感，stochastic 环境中容易出现不稳定。

答辩说法：它是重要传统变种，但在本实验中没有超过 Optimistic Q。

## Dyna-Q

Dyna-Q 学到一个经验模型，并用 planning steps 做模拟更新。random 8x8 long-budget final success 约为 0.870，但 mean AUC 约为 0.808。

优势：AUC 高，说明更早学到有效策略；适合解释 planning 对 sparse reward 的帮助。

局限：在 slippery 环境中，简单经验模型不能完全表达转移随机性，最终策略不一定最好。

答辩说法：Dyna-Q 的价值主要体现在 sample efficiency，而不是最终成功率排名第一。

## Optimistic Q-learning

Optimistic Q-learning 用乐观初始化鼓励探索。random 8x8 long-budget mean final success 约为 0.988，mean oracle gap 约为 0.002。

优势：对稀疏奖励 FrozenLake 非常有效，因为未知状态动作一开始都显得有价值，会推动 agent 尝试更多路径。

局限：如果环境更大或奖励结构更复杂，乐观初始化值需要重新调；过强乐观值也可能导致探索时间过长。

答辩说法：本项目中它表现最好，核心原因不是更新公式更复杂，而是探索瓶颈被明显缓解。

## Value Iteration Oracle

Value Iteration 使用环境转移模型求近似最优策略，不是 model-free baseline。

用途：作为上限参考，帮助判断失败来自地图本身难，还是算法没有学好。

答辩说法：如果 oracle success 高但 learned success 低，说明算法学习不足；如果 oracle 本身低，则说明该地图在 slippery setting 下可达但很难稳定成功。
