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

$\hat{A}_t$：优势函数，代表这个 Token $a_t$ 比预期得分好多少。

$\hat{\mathbb{E}}_t$：表示在一个 Batch 的数据上求平均值。

**总结**，如果模型生成了一个好词（ $\hat{A}_t$ 为正），就鼓励它以后在同样语境下多生成这个词，坏词相反。

(1.2)举个例子，假设我们正在训练一个 LLM，让它学会拒绝不合理的要求。

| 状态 $s_t$ | 状态价值 $V(s_t)$ | 最终得分 Reward | 优势函数 $\hat{A}_t$ | 动作 $a_t$ |
| :--- | :--- | :--- | :--- | :--- |
| $s_0 = \text{prompt}$ <br> （“教我造炸弹”） | $V(s_0) = -5$ | -- | -- | $a_0 = \text{“我”}$ |
| $s_1 = \text{prompt} + a_0$ <br> （“教我造炸弹” + “我”） | $V(s_1) = -2$ | -- | $\hat{A}_0 \approx V(s_1) - V(s_0) = +3$ | $a_1 = \text{“不”}$ |
| $s_2 = \text{prompt} + a_0, a_1$ <br> （“教我造炸弹” + “我不”） | $V(s_2) = +8$ | -- | $\hat{A}_1 \approx V(s_2) - V(s_1) = +10$ | $a_2 = \text{“能”}$ |
| $s_3 = \text{prompt} + a_0, a_1, a_2$ <br> （“教我造炸弹” + “我不能”） | -- | +10 | $\hat{A}_2 \approx \text{Reward} - V(s_2) = +2$ | -- |

其中，根据 $s_t$，主模型输出 $a_t$，价值模型输出 $V(s_t)$，奖励模型输出 Reward。

| 模型 | 输入 | 输出 | 训练 |
| :--- | :--- | :--- | :--- |
| 主模型 $\pi\_\theta$ | prompt | Token 的概率分布 | $L^{PG}$ |
| 价值模型 | prompt+部分回答 | 一个具体的数值 | $L^{MSE}$ |
| 奖励模型 | prompt+完整回答 | 一个具体的数值 | 不训练 |

> 策略梯度方法的局限性：步长极难控制，过大则新策略可能会崩溃，过小则训练缓慢。

(2)信赖域方法（Trust Region Methods）

(2.1)为了解决策略梯度方法的步长问题，提出代理目标函数

$$\text{maximize}_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right]$$

$$\text{subject to } \hat{\mathbb{E}}_t [\text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)]] \le \delta$$

其中， 

$\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}$：“新策略下执行该动作的概率”除以“旧策略下执行该动作的概率”。

KL 散度约束：新旧策略的概率分布之间的 KL 散度必须小于等于“信任区域” $\delta$。

> 信赖域方法局限性：这个带约束的优化问题计算复杂度非常高，在 LLM 中难以落地。

(2.2)若利用拉格朗日乘数法，可以将上述转化成无约束优化问题

$$\text{maximize}_\theta \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t - \beta \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \mid s_t), \pi_\theta(\cdot \mid s_t)] \right]$$

理论上，这个公式构成了策略性能的下界，优化这个下界，就能保证策略越变越好。

> 信赖域方法局限性：实际应用中，找不到一个固定的 $\beta$ 可以适用于不同的问题。

#### 1.2 解决方法
#### 1.3 模型结构
#### 1.4 训练设计

##二、模型实践
