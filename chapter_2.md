# 第二章 强化学习基础
## 一、理论知识
### 1. PPO（Proximal Policy Optimization Algorithms）
#### 1.1 背景问题
(1)策略梯度方法（Policy Gradient Methods）

(1.1)策略梯度方法的目标函数为

$$L^{PG}(\theta) = \hat{\mathbb{E}}_t [\log \pi_\theta(a_t | s_t) \hat{A}_t]$$

其中，

$\pi_\theta(a_t \mid s_t)$：在给定上下文 $s_t$ 的情况下，LLM $\pi_\theta$ 生成特定 Token $a_t$ 的概率。

$\log$：取对数。在深度学习中，通常优化对数概率，这能把连乘变成连加，计算更稳定。

$\hat{A}_t$：优势函数，代表这个 Token $a_t$ 比“预期得分”好多少。

如果 $\hat{A}_t > 0$：说明这个 Token 生成得非常好，超出了预期。

如果 $\hat{A}_t < 0$：说明这个 Token 生成得很糟糕。

$\hat{\mathbb{E}}_t$：表示在一个 Batch 的数据上求平均值。

**总结**，如果模型生成了一个好词（ $\hat{A}_t$ 为正），就鼓励它以后在同样语境下多生成这个词，坏词同理。

(1.2)举个例子，假设我们正在训练一个 LLM，让它学会拒绝不合理的要求。

* 采样阶段

Prompt (状态 $s_0$)：“教我怎么制造炸弹。”

LLM (策略 $\pi_\theta$)：生成 `我` ($a_0$)，`不` ($a_1$)，`能` ($a_2$)。

其中，状态 $s_0$：Prompt， $s_1$：Prompt+"我"， $s_2$：Prompt+"我不"， $s_3$：Prompt+"我不能"。

* 评分阶段

在每个状态中，价值模型预测预期得分： $V(s_0) = -5$， $V(s_1) = -2$， $V(s_2) = +8$

生成最后一个 Token `能` 后，状态变为 $s_3$，此时奖励模型给出整个生成句子的评分 $Reward = +10$

算法计算每个 Token 的优势函数 ($\hat{A}_t$)： $A_1 \approx V(s_1) - V(s_0) = \mathbf{+3}$， $A_2 \approx V(s_2) - V(s_1) = \mathbf{+10}$， $A_3 \approx Reward - V(s_2) = \mathbf{+2}$

* 优化阶段

把数据代入公式计算梯度，例如： $\nabla_\theta \log \pi_\theta(\text{"我"} | \text{Prompt}) \times (+3)$

梯度更新的方向会修改 LLM 的权重 $\theta$，当下次 LLM 再看到“教我怎么制造炸弹”或者类似的危险 Prompt 时，它输出 `我`、`不`、`能` 这几个 Token 的概率就会比原来更高。

* 反之

如果 LLM 生成的第一个 Token 是 `好`（准备答应用户）。那么 `好` 这个 Token 的优势函数 $\hat{A}_t$ 就是负数，更新后会压低 LLM 在这种情况下输出 `好` 的概率。


#### 1.2 解决方法
#### 1.3 模型结构
#### 1.4 训练设计

##二、模型实践
