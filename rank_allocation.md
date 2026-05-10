下面这份可以直接作为给导师看的方法梳理稿，也可以作为给 vibe coding 工具的工程说明。核心思路是：**把已有的 Low-Rank Restart 从“固定 rank 的周期性重启”升级为“Credit-Guided Dynamic Rank Allocation”，即每次 restart 时不仅做 Momentum Reset 和 Weight Refactorization，还根据过去一个 restart interval 内各矩阵的贡献与 rank 饱和度，重新分配 rank budget。**

我建议把这个方法暂时命名为：

> **CORA-Restart: Credit-guided Online Rank Allocation with Low-Rank Restarts**

或者更正式一点：

> **Credit-Guided Dynamic Rank Allocation for Low-Rank LLM Pretraining with Restart Coupling**

---

# 0. 一句话总结方法

已有 lowrank_benchmark.pdf 的核心发现是：直接把权重矩阵分解为

[
W_g = B_g A_g
]

虽然节省权重、梯度和 optimizer states 的内存，但固定低秩表达能力有限；你们提出的 **Weight Refactorization + Momentum Reset** 能显著提升 vanilla low-rank 和 SLTrain，且论文里已经说明这两个技巧可以直接用于 low-rank 和 SLTrain，并将其命名为 Low-Rank-restarts / SLTrain-restarts。实验中 1B 模型上 Low-Rank-restarts 达到 15.013 PPL，接近 Fira 15.098 和 GaLore 15.573，同时保持更低内存使用；SLTrain-restarts 达到 14.37 PPL，进一步缩小与 full-rank 的差距 。

现在我们进一步提出：

> 既然每隔 (T) 步已经要 restart，那么 restart 不仅可以 reset momentum、refactorize (B,A)，还可以成为一个天然的 **rank reallocation boundary**。
> 在每个 restart interval 内，我们统计每个矩阵的 update credit 和 rank pressure；在下一次 restart 时，把 rank 从低收益矩阵转移到高收益且 rank 不足的矩阵上。

也就是说：

[
\text{Restart}
==============

\text{Momentum Reset}
+
\text{Weight Refactorization}
+
\text{Dynamic Rank Reallocation}
]

这三者是高度耦合的。

---

# 1. 背景：为什么要做这个？

## 1.1 Vanilla low-rank pretraining 的问题

在 low-rank pretraining 中，我们直接把原本 full-rank 的矩阵

[
W_g \in \mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
]

替换为：

[
W_g = B_g A_g
]

其中：

[
B_g\in \mathbb{R}^{d_{\text{out}}\times r_g},
\qquad
A_g\in \mathbb{R}^{r_g\times d_{\text{in}}}
]

通常设：

[
r_g = \frac{d}{4}
]

或者所有同类矩阵使用统一 rank。

这个做法的优点是明显的：如果 (r_g \ll d)，那么参数、梯度、optimizer states 都大幅减少。lowrank_benchmark.pdf 里的内存表也明确显示，Low-rank 的 weight、optimizer 和 gradient memory 都是 (O((m+n)r))，明显小于 full-rank 的 (O(mn)) 级别 。

但问题也很直接：固定 rank 默认假设所有矩阵的容量需求相同。这在 LLM pretraining 里几乎不可能成立。不同层、不同模块、不同矩阵在训练不同阶段承担的功能不同，比如 embedding、attention projection、MLP up/gate/down、LM head 的优化需求显然不一样。

因此，uniform rank allocation 是一个强假设：

[
r_1=r_2=\cdots=r_G
]

而我们真正想要的是：

[
r_g \propto \text{usefulness / capacity demand of matrix } g
]

---

## 1.2 已有 restart 方法提供了天然切入点

lowrank_benchmark.pdf 已经提出两个重要技巧。

第一个是 **Weight Refactorization**。对于

[
W = BA
]

只要 (W) 不变，forward 输出就不变。因此可以周期性地把当前 (W) 做 SVD：

[
W = U\Sigma V^\top
]

然后重新设置：

[
B' = U\sqrt{\Sigma},
\qquad
A' = \sqrt{\Sigma}V^\top
]

这不会改变模型函数，但会让 (B,A) 的尺度更平衡。论文中的 Lemma 4.1 说明，当 (B^\ast A^\ast = U\Sigma V^\top)，并令 (B^\ast=U\Sigma^\alpha)、(A^\ast=\Sigma^{1-\alpha}V^\top) 时，相关 Hessian condition number 在 (\alpha=1/2) 时最优；也就是 (B'=U\sqrt\Sigma)、(A'=\sqrt\Sigma V^\top) 是一个更好的平衡分解 。

第二个是 **Momentum Reset**。论文中将其解释为周期性清空 optimizer momentum，以消除 spike batch 或过时方向对后续优化的污染；实验中采用每 200 updates reset momentum，并报告这是最优设置之一 。

这两个技巧已经形成一个周期性结构：

[
\text{train } T \text{ steps}
\rightarrow
\text{restart}
\rightarrow
\text{train } T \text{ steps}
\rightarrow
\text{restart}
]

现在我们把这个周期结构扩展为：

[
\text{train } T \text{ steps and collect credit}
\rightarrow
\text{rank reallocation}
\rightarrow
\text{refactorization}
\rightarrow
\text{momentum reset}
]

这个顺序非常自然，因为 rank 改变本身会重构 (B,A)，而重构后本来就应该清空或重置对应 optimizer states。

---

# 2. Credit Assignment 方法梳理：理论与工程实践

## 2.1 理论目标

我们希望回答的问题是：

> 在一次 optimizer update 中，第 (g) 个矩阵到底对 loss change 贡献了多少？

设模型参数被划分为 (G) 个矩阵或参数组：

[
w = {w_1,w_2,\ldots,w_G}
]

一次 optimizer update 可以写成：

[
\Delta_t = \sum_{g=1}^G \Delta_{t,g}
]

其中 (\Delta_{t,g}) 只作用于第 (g) 个矩阵。

真实 loss change 是：

[
L(w_{t+1})-L(w_t)
=================

L\left(w_t+\sum_g \Delta_{t,g}\right)-L(w_t)
]

我们想构造 credit：

[
C_{t,g}
]

满足：

[
\sum_g C_{t,g}
==============

L(w_{t+1})-L(w_t)
]

其中：

* (C_{t,g}<0)：第 (g) 个矩阵帮助 loss 下降；
* (C_{t,g}>0)：第 (g) 个矩阵让 loss 上升，或者至少对当前 step 有害。

credit_assignment.md 的核心思想是把这个问题看成一个 **update-space attribution** 问题，而不是普通 gradient norm 或 local stability proxy 。

---

## 2.2 Update-Space Integrated Credit

对每个 group 引入一个 gate：

[
z_g\in[0,1]
]

定义虚拟函数：

[
F_t(z_1,\ldots,z_G)
===================

L\left(
w_t+\sum_{g=1}^G z_g \Delta_{t,g}
\right)
]

含义是：

* (z_g=0)：第 (g) 个矩阵的 update 没有发生；
* (z_g=1)：第 (g) 个矩阵的 update 完整发生；
* (F_t(0,\ldots,0)=L(w_t))；
* (F_t(1,\ldots,1)=L(w_{t+1}))。

沿直线路径：

[
z(\lambda)=\lambda \mathbf{1},\qquad \lambda\in[0,1]
]

定义：

[
C_{t,g}
=======

\int_0^1
\frac{\partial F_t(\lambda\mathbf{1})}{\partial z_g}
d\lambda
]

展开后得到：

[
C_{t,g}
=======

\int_0^1
\left\langle
\nabla_{w_g}L(w_t+\lambda\Delta_t),
\Delta_{t,g}
\right\rangle
d\lambda
]

这个定义有两个关键优点。

第一，它满足 completeness：

[
\sum_g C_{t,g}
==============

L(w_{t+1})-L(w_t)
]

credit_assignment.md 中已经推导了这个性质：把所有 group 的 credit 相加，积分项正好变成 (\frac{d}{d\lambda}L(w_t+\lambda\Delta_t))，因此积分结果等于 endpoint loss difference 。

第二，它比单纯的一阶项：

[
\langle \nabla_{w_g}L(w_t),\Delta_{t,g}\rangle
]

更合理，因为它沿着从 (w_t) 到 (w_{t+1}) 的路径积分，能在一定程度上处理不同矩阵 update 之间的 interaction。

---

## 2.3 工程近似：梯形 credit

精确计算积分太贵。最实用版本是梯形近似：

[
C_{t,g}^{trap}
==============

\frac{1}{2}
\left[
\langle g_{t,g}, \Delta_{t,g}\rangle
+
\langle g_{t+1,g}, \Delta_{t,g}\rangle
\right]
]

其中：

[
g_{t,g}=\nabla_{w_g}L(w_t)
]

[
g_{t+1,g}=\nabla_{w_g}L(w_{t+1})
]

第一项当前 backward 已经有了。第二项可以用下一步 backward 的 gradient 近似，所以这是一个 **one-step delayed credit**。credit_assignment.md 中也明确指出，(\nabla_g L(w_{t+1})) 可以用同 batch post-update gradient 或下一 step gradient；前者更准但需要额外 backward，后者几乎免费但噪声更大，适合在线训练设置 。

---

## 2.4 在 low-rank (W=BA) 下如何定义 group credit？

这里有两种粒度。

### 粒度 A：以整个矩阵 (W_g=B_gA_g) 为 group

这是我最推荐的。对每个原始权重矩阵 (W_g)，把它的 (B_g,A_g) 看成一个整体 group。

它的 update 是：

[
\Delta W_{t,g}
==============

# W_{t+1,g}-W_{t,g}

B_{t+1,g}A_{t+1,g}-B_{t,g}A_{t,g}
]

然后 credit 用：

[
C_{t,g}^{W}
\approx
\frac{1}{2}
\left[
\langle G_{t,g}, \Delta W_{t,g}\rangle
+
\langle G_{t+1,g}, \Delta W_{t,g}\rangle
\right]
]

这里：

[
G_{t,g}=\nabla_{W_g}L
]

但 full (G_{t,g}) 不一定显式存在。工程上不建议每步显式构造 (G)。

因此实际中我们可以采用等价的 factor-space 近似。

---

### 粒度 B：以 (B_g,A_g) 参数更新为 group

直接使用 PyTorch 已经有的梯度：

[
\nabla_{B_g}L,\qquad \nabla_{A_g}L
]

以及 optimizer 对 (B_g,A_g) 的实际参数变化：

[
\Delta B_{t,g}=B_{t+1,g}-B_{t,g}
]

[
\Delta A_{t,g}=A_{t+1,g}-A_{t,g}
]

定义：

[
C_{t,g}^{BA}
============

\frac{1}{2}
\left[
\langle \nabla_{B_g}L_t,\Delta B_{t,g}\rangle
+
\langle \nabla_{A_g}L_t,\Delta A_{t,g}\rangle
+
\langle \nabla_{B_g}L_{t+1},\Delta B_{t,g}\rangle
+
\langle \nabla_{A_g}L_{t+1},\Delta A_{t,g}\rangle
\right]
]

这个量不完全等价于 (W)-space credit，但它工程上最简单，因为：

* 不需要构造 full-rank (G_g)；
* 不需要 hook activation 和 output gradient；
* 只需要记录每个 low-rank factor 的参数差或重构参数差；
* 可以直接用 PyTorch autograd 的 `.grad`。

对于第一版实现，我建议用 (C_{t,g}^{BA})。它足够作为 rank allocation 的 useful signal。

---

## 2.5 低成本实现：只保存 scalar，避免存完整 (\Delta)

credit_assignment.md 中对 AdamW 有一个重要工程技巧：AdamW 下可以根据 optimizer states 重构 (\Delta_t)，不需要保存完整 update。其核心是 AdamW step 后，optimizer state 中已有 (m_t,v_t)，而参数是 (w_{t+1})，可以根据 AdamW 公式反推出 (w_t) 和 (\Delta_t)，额外只需保存上一 step 的 learning rate、step index 和少量 scalar，额外 memory 是 (O(G)) 而不是 (O(|w|)) 。

不过对于 low-rank 动态 rank 分配，我建议工程第一版不要一上来做重构 (\Delta)。更稳的方式是：

* 每个 interval 内只在少数统计点收集 credit；
* 或者在 optimizer step 前后临时保存 (B,A) 的 fp32/bf16 copy 的差值；
* 由于 restart interval 约 (T=500)，可以每 10 或 20 steps 采样一次 credit，不需要每步计算。

第一版目标不是把 credit 算到极致精确，而是验证：

> credit-guided rank allocation 是否优于 uniform rank 和 random rank reallocation。

---

# 3. Credit-Guided Dynamic Rank Allocation 方法

## 3.1 为什么这个 idea 合理？

low-rank pretraining 的核心瓶颈是容量预算。固定 rank 低秩训练等价于给每个矩阵固定数量的 update channel：

[
r_g
]

如果某些矩阵在当前阶段对 loss 下降贡献更大，或者其新增 rank 方向有强梯度信号，那么它们更应该获得 rank budget。

这和 SOLAR 的思路是统一的。SOLAR 的论文动机是：固定 LR schedule 不能根据训练过程中的 loss dynamics、gradient norms、parameter magnitudes 自适应调整；SOLAR 通过状态驱动的 residual LR modulation 对不同 parameter groups 做在线控制 。现在我们把同样的“resource allocation”思想从 LR budget 扩展到 rank budget：

[
\text{SOLAR: allocate learning-rate budget}
]

[
\text{CORA-Restart: allocate low-rank capacity budget}
]

LR 是连续、每步可调的资源；rank 是离散、结构性的资源，不应该每步调，而应该在 restart boundary 调。

---

## 3.2 不能直接按 raw credit 排序分 rank

最 naive 的做法是：

[
P_g = \operatorname{EMA}(\max(-C_{t,g},0))
]

然后按 (P_g) 从高到低分 rank。

这个做法不可靠，原因有三个。

第一，raw credit 会偏向大矩阵。大矩阵参数多、update norm 大、内积自然大。如果直接按 raw credit 分配，可能只是把 rank 分给矩阵尺寸大的模块，而不是分给真正需要 rank 的模块。

第二，raw credit 高可能说明当前 rank 已经足够，并不代表还需要更多 rank。一个矩阵可能因为已有 rank 足够而贡献高；继续加 rank 的边际收益反而很低。

第三，raw credit 低也不一定说明不重要。它可能是因为当前 rank 太低，导致可表达 update 空间过窄，因此它“做不出贡献”。这类矩阵反而可能需要更多 rank。

所以我们需要区分：

[
\text{current contribution}
]

和：

[
\text{marginal utility of additional rank}
]

真正应该分配 rank 的依据是：

> 当前矩阵是否既有有效训练贡献，又存在 rank 不足的信号。

---

## 3.3 Credit per Rank

第一步是把 raw credit 改成单位 rank 的 credit。

定义正贡献：

[
P_{t,g}=\max(-C_{t,g},0)
]

然后定义：

[
E_{t,g}
=======

\frac{\operatorname{EMA}(P_{t,g})}{r_g+\epsilon}
]

这表示：

> 第 (g) 个矩阵当前每个 rank channel 平均产生多少有效 loss decrease。

但这还不够，因为它仍然可能受 update norm 影响。因此更稳的定义是：

[
E_{t,g}
=======

\frac{
\operatorname{EMA}(\max(-C_{t,g},0))
}{
\operatorname{EMA}(|\Delta \theta_{t,g}|_2^2)+\epsilon
}
]

其中 (\Delta \theta_{t,g}) 可以是：

[
\Delta \theta_{t,g}
===================

(\Delta B_{t,g}, \Delta A_{t,g})
]

也可以是：

[
\Delta W_{t,g}
==============

B_{t+1,g}A_{t+1,g}-B_{t,g}A_{t,g}
]

第一版建议用 factor-space：

[
|\Delta \theta_{t,g}|_2^2
=========================

|\Delta B_{t,g}|*F^2+|\Delta A*{t,g}|_F^2
]

于是：

[
E_{t,g}
=======

\frac{
\operatorname{EMA}(\max(-C_{t,g}^{BA},0))
}{
\operatorname{EMA}(|\Delta B_{t,g}|*F^2+|\Delta A*{t,g}|_F^2)+\epsilon
}
]

这表示：

> 单位 update energy 带来的 loss decrease。

它比 raw credit 更像“效率”。

---

## 3.4 Rank Pressure

Credit per Rank 衡量的是当前 rank 的有效性，但仍然没有直接回答“是否需要更多 rank”。因此我们再引入 rank pressure。

我建议定义：

[
R_{t,g}^{pressure}
==================

E_{t,g}
\cdot
\frac{1}{\sqrt{r_g}}
]

其中：

[
\frac{1}{\sqrt{r_g}}
]

是 rank scarcity factor。它的含义是：如果两个矩阵效率相近，rank 更低的矩阵更值得获得新增 rank，因为它更可能处于 capacity-starved 状态。

也可以使用：

[
\phi(r_g)=\log\frac{r_{\max}}{r_g}
]

但第一版用 (1/\sqrt{r_g}) 更简单。

---

# 4. 低 rank 饱和度：工程友好的 rank probe

## 4.1 为什么需要 saturation？

如果只看 credit per rank，仍然不够。我们还要知道：

> 如果给这个矩阵多一个 rank，它是否真的有可用下降方向？

理想指标是 full gradient 在当前 low-rank tangent space 外的残差：

[
\frac{
|G_g-\Pi_{\mathcal{T}_g}(G_g)|_F
}{
|G_g|_F+\epsilon
}
]

但你之前指出得很对：在实际 low-rank 训练中，我们没有显式 full-rank gradient (G_g=\nabla_{W_g}L)，直接算这个 residual 成本很高。

因此第一版不应该做 full residual，而应该做 **zero-output rank probe**。

---

## 4.2 Zero-output rank probe

当前矩阵：

[
W_g=B_gA_g
]

临时加入一个 probe rank：

[
W_g'
====

B_gA_g
+
B_{p,g}A_{p,g}
]

其中：

[
B_{p,g}=0
]

[
A_{p,g}\sim \mathcal{N}(0,\sigma_p^2)
]

这样：

[
B_{p,g}A_{p,g}=0
]

所以 forward 完全不变，模型输出不变，loss 也不变。

但是 backward 时：

[
\nabla_{B_{p,g}}L
=================

G_g A_{p,g}^\top
]

如果：

[
|\nabla_{B_{p,g}}L|_F
]

很大，说明沿着这个新增 rank 方向有明显梯度信号。换句话说：

> 如果给这个矩阵增加 rank，它大概率能利用这个新通道下降 loss。

所以定义 saturation score：

[
S_{t,g}^{probe}
===============

|\nabla_{B_{p,g}}L|_F
]

为了更稳，可以归一化：

[
S_{t,g}^{probe}
===============

\frac{
|\nabla_{B_{p,g}}L|*F
}{
|A*{p,g}|_F+\epsilon
}
]

如果一次加 (k_p) 个 probe rank：

[
B_{p,g}\in \mathbb{R}^{d_{\text{out}}\times k_p}
]

[
A_{p,g}\in \mathbb{R}^{k_p\times d_{\text{in}}}
]

则：

[
S_{t,g}^{probe}
===============

\frac{
|\nabla_{B_{p,g}}L|*F
}{
\sqrt{k_p}(|A*{p,g}|_F+\epsilon)
}
]

实际建议：

[
k_p=1 \text{ or } 2
]

每个 restart interval 末尾只做一次 probe backward，或者在 interval 内做少量 probe。

---

## 4.3 为什么 probe 比 full residual 更适合工程实现？

因为它不需要你显式构造：

[
G_g=\nabla_{W_g}L
]

也不需要写复杂 hook 来缓存 input activation 和 output gradient。它直接利用 autograd：

1. 给 low-rank layer 临时加 probe branch；
2. forward 不变；
3. backward 后读取 probe 参数的 gradient；
4. 删除 probe branch；
5. 用 probe grad norm 作为新增 rank 是否有价值的信号。

这对 vibe coding 工具非常友好。

---

# 5. 最终 rank score：Credit × Saturation

综合上面两类信号，我建议最终 score 定义为：

[
Q_g
===

\underbrace{
\operatorname{EMA}
\left(
\frac{
\max(-C_{t,g},0)
}{
|\Delta \theta_{t,g}|*2^2+\epsilon
}
\right)
}*{\text{Credit efficiency}}
\cdot
\underbrace{
\operatorname{EMA}(S_{t,g}^{probe})
}*{\text{Saturation / new-rank demand}}
\cdot
\underbrace{
\frac{1}{\sqrt{r_g}}
}*{\text{Rank scarcity}}
]

其中：

[
\Delta \theta_{t,g}
===================

(\Delta B_{t,g},\Delta A_{t,g})
]

第一项回答：

> 当前已有 rank 是否产生有效训练贡献？

第二项回答：

> 如果新增 rank，是否存在可用梯度信号？

第三项回答：

> 当前 rank 是否偏低，是否值得优先补充？

所以：

[
Q_g \text{ 高}
]

意味着：

> 这个矩阵当前 update 有效率高、新增 rank 有梯度需求、而且当前 rank 相对稀缺，应该加 rank。

---

# 6. 动态 rank 分配：只在 restart 时做，不每步做

rank 是结构变量，不是 learning rate。它不能每 step 抖动。你们已有 restart interval，例如论文里 refactorization 每 200 updates，momentum reset 每 200 updates；你现在提到可以设 (T\approx 500)，这也合理。关键是：**rank reallocation 必须低频发生，并和 restart 同步。**

训练流程应该是：

[
\text{for each interval } k:
]

1. 使用当前 rank allocation 训练 (T) 步；
2. 在这 (T) 步内累计 credit EMA；
3. interval 末尾做 rank probe；
4. 计算 (Q_g)；
5. 根据 (Q_g) 重新分配 rank；
6. 对新 (B,A) 做 refactorization；
7. reset optimizer momentum；
8. 进入下一个 interval。

这叫：

> **Periodic Dynamic Rank Reallocation**

也就是用户指定的 Version 2。

---

# 7. 初始设置：一开始就 factorize (W=BA)，统一 rank (r=d/4)

你希望保持已有 lowrank pretraining setup：

[
W_g = B_g A_g
]

初始：

[
r_g^{(0)} = \frac{d_g}{4}
]

更具体地，如果矩阵是：

[
W_g\in\mathbb{R}^{d_{\text{out},g}\times d_{\text{in},g}}
]

可以定义：

[
d_g = \min(d_{\text{out},g},d_{\text{in},g})
]

初始 rank：

[
r_g^{(0)} = \left\lfloor \rho d_g \right\rfloor
]

其中：

[
\rho=0.25
]

总 rank budget：

[
R_{\text{total}}=\sum_g r_g^{(0)}
]

后续动态分配必须保持：

[
\sum_g r_g = R_{\text{total}}
]

这样内存预算与 baseline uniform low-rank 保持一致。实验对比才公平。

---

# 8. Rank reallocation 规则

## 8.1 简单 TopK-BottomK 版本

每次 restart 时：

[
\mathcal{A}=\text{TopK}(Q_g)
]

[
\mathcal{R}=\text{BottomK}(Q_g)
]

对高分矩阵加 rank：

[
r_g\leftarrow r_g+\Delta r
]

对低分矩阵减 rank：

[
r_h\leftarrow r_h-\Delta r
]

保持总 rank 不变。

实际建议：

[
\Delta r = 4 \text{ or } 8
]

而不是每次只移动 1，因为 GPU kernel 和矩阵 shape 对齐通常喜欢 8、16、32 的倍数。

同时设置边界：

[
r_{\min} \le r_g \le r_{\max}
]

例如：

[
r_{\min}=0.125d_g
]

[
r_{\max}=0.5d_g
]

如果一开始是 (0.25d_g)，那么允许在：

[
[0.125d_g,\ 0.5d_g]
]

之间移动。

---

## 8.2 更稳的 soft allocation 版本

也可以直接根据 (Q_g) 重新计算 rank：

[
\tilde{Q}_g
===========

(Q_g+\epsilon)^\tau
]

[
r_g^{target}
============

r_{\min,g}
+
\left\lfloor
(R_{\text{total}}-\sum_h r_{\min,h})
\frac{\tilde Q_g}{\sum_h \tilde Q_h}
\right\rfloor
]

其中：

* (\tau=1)：正常分配；
* (\tau>1)：更激进，强者更多 rank；
* (\tau<1)：更保守，更接近 uniform。

为了避免 rank 剧烈变化，不直接跳到 target，而是限制每次变化：

[
r_g^{new}
=========

r_g
+
\operatorname{clip}
(r_g^{target}-r_g,\ -\Delta_{\max},\ \Delta_{\max})
]

其中：

[
\Delta_{\max}=8 \text{ or } 16
]

第一版我建议用 TopK-BottomK，因为更容易 debug。

---

## 8.3 Hysteresis 防抖

为了避免 rank 在两个矩阵之间来回移动，设置阈值：

[
Q_{\text{add}} > Q_{\text{remove}}(1+\delta)
]

例如：

[
\delta=0.1
]

只有当 top group 明显优于 bottom group 时才移动 rank。

否则本轮 restart 只做 refactorization + momentum reset，不做 rank 变化。

---

# 9. 与 Low-Rank Restart 的完美结合

这里是给导师看的关键叙事。

## 9.1 Restart 本来就是低秩训练的“结构刷新点”

已有论文中，restart 包括：

1. Weight Refactorization；
2. Momentum Reset。

Weight Refactorization 会重新分解 (W=BA)，改变 (B,A) 的具体数值但保持 (W) 不变；Momentum Reset 会清空 AdamW momentum，避免旧 factorization 下的 optimizer states 污染新 factorization。lowrank_benchmark.pdf 说明这两个技巧可以直接用于 low-rank 和 SLTrain，且 ablation 显示 momentum reset 对加速收敛贡献更大，weight refactorization 稳定提升性能 。

现在，rank reallocation 也是一种结构刷新：

[
B_g\in \mathbb{R}^{d_{\text{out}}\times r_g}
\rightarrow
B'*g\in \mathbb{R}^{d*{\text{out}}\times r'_g}
]

[
A_g\in \mathbb{R}^{r_g\times d_{\text{in}}}
\rightarrow
A'_g\in \mathbb{R}^{r'*g\times d*{\text{in}}}
]

它天然需要：

* 重新初始化或截断 (B,A)；
* 重新整理 optimizer states；
* 清空或部分清空 momentum。

因此它和 restart 是同一类操作。最合适的位置就是 restart boundary。

---

## 9.2 Refactorization 可以同时支持 rank 增减

如果当前：

[
W_g=B_gA_g
]

rank 改为 (r'_g)，可以先 materialize 当前矩阵：

[
W_g = B_g A_g
]

然后做 truncated SVD：

[
W_g \approx U_{r'}\Sigma_{r'}V_{r'}^\top
]

设置：

[
B'*g=U*{r'}\sqrt{\Sigma_{r'}}
]

[
A'*g=\sqrt{\Sigma*{r'}}V_{r'}^\top
]

这样 rank reallocation 和 weight refactorization 完全合并为一个操作。

如果 (r'_g=r_g)，这就是原来的 refactorization。
如果 (r'_g>r_g)，这是 rank expansion。
如果 (r'_g<r_g)，这是 rank compression。

因此：

> Dynamic Rank Reallocation can be implemented as rank-adaptive weight refactorization.

这是一个非常漂亮的 coupling。

---

## 9.3 Momentum Reset 是 rank 改变后的必要步骤

rank 改变后，原来的 AdamW optimizer states：

[
m_B,v_B,m_A,v_A
]

shape 可能变化，也不再对应新的 (B',A')。因此最简单、最稳的做法是：

[
m_{B'}=0,\quad v_{B'}=0,\quad m_{A'}=0,\quad v_{A'}=0
]

这正好就是 Momentum Reset。

所以不是“顺便 reset”，而是：

> rank reallocation 后必须 reset 或重建 optimizer states；已有 Momentum Reset 正好提供了理论和实验支持。

---

# 10. PyTorch 工程实现教程

下面按照模块拆解。

---

## 10.1 LowRankLinear 模块设计

建议把所有需要低秩化的 `nn.Linear` 替换成：

```python
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.B = nn.Parameter(torch.empty(out_features, rank))
        self.A = nn.Parameter(torch.empty(rank, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        self.reset_parameters()

        # statistics
        self.credit_ema = 0.0
        self.energy_ema = 0.0
        self.probe_ema = 0.0
        self.score = 0.0
```

forward：

```python
def forward(self, x):
    # x: [..., in_features]
    # W = B @ A, but avoid explicitly materializing W when possible
    y = F.linear(x, self.A)      # [..., rank]
    y = F.linear(y, self.B)      # [..., out_features]
    if self.bias is not None:
        y = y + self.bias
    return y
```

对于大模型，这比每次显式算 (W=B@A) 更省。

---

## 10.2 支持 probe branch

给 `LowRankLinear` 加 probe 参数，但默认关闭：

```python
def enable_probe(self, probe_rank=1, sigma=1e-3):
    self.probe_rank = probe_rank
    self.B_probe = nn.Parameter(
        torch.zeros(self.out_features, probe_rank, device=self.B.device, dtype=self.B.dtype)
    )
    self.A_probe = nn.Parameter(
        torch.randn(probe_rank, self.in_features, device=self.A.device, dtype=self.A.dtype) * sigma
    )
    self.probe_enabled = True
```

forward 中：

```python
def forward(self, x):
    y = F.linear(F.linear(x, self.A), self.B)

    if getattr(self, "probe_enabled", False):
        # B_probe is zero, so this branch does not change output initially
        y_probe = F.linear(F.linear(x, self.A_probe), self.B_probe)
        y = y + y_probe

    if self.bias is not None:
        y = y + self.bias
    return y
```

probe backward 后读取：

```python
def get_probe_score(self):
    if not getattr(self, "probe_enabled", False):
        return 0.0
    if self.B_probe.grad is None:
        return 0.0
    score = self.B_probe.grad.detach().float().norm().item()
    denom = self.A_probe.detach().float().norm().item() + 1e-12
    return score / denom
```

然后删除 probe：

```python
def disable_probe(self):
    if hasattr(self, "B_probe"):
        del self.B_probe
    if hasattr(self, "A_probe"):
        del self.A_probe
    self.probe_enabled = False
```

注意：probe 参数不要加入长期 optimizer，或者只在 probe step 临时加入但不 step。最简单做法是：probe backward 只用来读 grad，然后立刻删除，不调用 optimizer.step 更新它。

---

## 10.3 Credit 统计实现

第一版最简单：直接在 optimizer step 前后保存 (B,A) 的 copy，用于计算 (\Delta B,\Delta A)。

在每个统计 step：

```python
for module in lowrank_modules:
    module.B_prev = module.B.detach().clone()
    module.A_prev = module.A.detach().clone()
    module.B_grad_prev = module.B.grad.detach().clone()
    module.A_grad_prev = module.A.grad.detach().clone()
```

执行 optimizer step 后：

```python
for module in lowrank_modules:
    dB = module.B.detach() - module.B_prev
    dA = module.A.detach() - module.A_prev

    # endpoint/current simple credit
    c_now = (module.B_grad_prev.float() * dB.float()).sum()
    c_now += (module.A_grad_prev.float() * dA.float()).sum()

    energy = dB.float().pow(2).sum() + dA.float().pow(2).sum()

    module.last_credit = c_now.item()
    module.last_energy = energy.item()
```

如果要用 trapezoidal credit，则下一步 backward 后再加 post-gradient：

```python
c_post = (module.B.grad.detach().float() * dB.float()).sum()
c_post += (module.A.grad.detach().float() * dA.float()).sum()

credit = 0.5 * (c_now + c_post)
```

但是这需要保存 `dB,dA` 到下一步。第一版可以只用 endpoint 或 current term：

[
C_{t,g}^{first}
===============

\langle \nabla_B L_t,\Delta B_t\rangle
+
\langle \nabla_A L_t,\Delta A_t\rangle
]

或者便宜版 delayed endpoint：

[
C_{t,g}^{end}
=============

\langle \nabla_B L_{t+1},\Delta B_t\rangle
+
\langle \nabla_A L_{t+1},\Delta A_t\rangle
]

如果担心保存 (\Delta B,\Delta A) 的 memory，可以每 20 steps 采样一次，开销可控。后续再优化为 AdamW reconstruction。

---

## 10.4 EMA 更新

每次拿到 credit 后：

```python
def update_credit_stats(module, credit, energy, beta=0.95, eps=1e-12):
    positive_credit = max(-credit, 0.0)
    efficiency = positive_credit / (energy + eps)

    module.credit_ema = beta * module.credit_ema + (1 - beta) * positive_credit
    module.energy_ema = beta * module.energy_ema + (1 - beta) * energy
    module.eff_ema = beta * getattr(module, "eff_ema", 0.0) + (1 - beta) * efficiency
```

probe 后：

```python
def update_probe_stats(module, probe_score, beta=0.9):
    module.probe_ema = beta * module.probe_ema + (1 - beta) * probe_score
```

最终 score：

```python
def compute_rank_score(module, eps=1e-12):
    rank = module.rank
    credit_eff = getattr(module, "eff_ema", 0.0)
    saturation = getattr(module, "probe_ema", 0.0)
    scarcity = 1.0 / math.sqrt(rank + eps)
    module.score = credit_eff * saturation * scarcity
    return module.score
```

---

## 10.5 Restart 时的 rank allocation

收集所有 lowrank modules：

```python
modules = list_lowrank_modules(model)
```

计算 score：

```python
scores = [(m, compute_rank_score(m)) for m in modules]
```

TopK / BottomK：

```python
add_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
remove_candidates = sorted(scores, key=lambda x: x[1])
```

Rank move：

```python
def allocate_ranks(modules, delta_rank=8, top_k=4, r_min_ratio=0.125, r_max_ratio=0.5, hysteresis=0.1):
    scores = [(m, m.score) for m in modules]

    add_list = sorted(scores, key=lambda x: x[1], reverse=True)
    remove_list = sorted(scores, key=lambda x: x[1])

    moves = []

    i, j = 0, 0
    while len(moves) < top_k and i < len(add_list) and j < len(remove_list):
        m_add, s_add = add_list[i]
        m_rem, s_rem = remove_list[j]

        max_rank = int(r_max_ratio * min(m_add.out_features, m_add.in_features))
        min_rank = int(r_min_ratio * min(m_rem.out_features, m_rem.in_features))

        if m_add.rank + delta_rank > max_rank:
            i += 1
            continue

        if m_rem.rank - delta_rank < min_rank:
            j += 1
            continue

        if s_add <= s_rem * (1.0 + hysteresis):
            break

        moves.append((m_rem, m_add, delta_rank))
        i += 1
        j += 1

    return moves
```

---

## 10.6 Rank 改变 + Refactorization

最关键函数：

```python
@torch.no_grad()
def refactorize_with_new_rank(module, new_rank):
    # Materialize W
    W = module.B.float() @ module.A.float()

    # SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    r = new_rank
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]

    sqrtS = torch.sqrt(S_r)

    B_new = U_r * sqrtS.unsqueeze(0)
    A_new = sqrtS.unsqueeze(1) * Vh_r

    # Replace parameters
    module.rank = r
    module.B = nn.Parameter(B_new.to(dtype=module.B.dtype, device=module.B.device))
    module.A = nn.Parameter(A_new.to(dtype=module.A.dtype, device=module.A.device))
```

注意：这个函数替换了 `nn.Parameter`，所以 optimizer 里的 param references 会失效。因此必须重建 optimizer，或者写一个专门的 optimizer state patch。

第一版最简单：

> 每次 restart 后，重建 optimizer。

例如：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
```

因为 restart 本来就要 reset momentum，所以重建 optimizer 是合理的。

如果有 scheduler，需要保留当前 global step：

```python
scheduler.last_epoch = global_step
```

或者使用手写 LR schedule，不依赖 optimizer 内部 step。

---

## 10.7 Restart 主流程伪代码

```python
for step in range(total_steps):

    loss = forward_backward(batch)

    # optional: collect pre-step grads for credit
    if step % credit_sample_interval == 0:
        save_pre_step_lowrank_state(model)

    optimizer.step()
    optimizer.zero_grad()

    # update credit after step or next backward depending on chosen estimator
    if step % credit_sample_interval == 0:
        compute_and_update_credit_stats(model)

    if (step + 1) % restart_interval == 0:

        # 1. Probe saturation
        enable_probe_for_all_lowrank_modules(model, probe_rank=1)
        probe_loss = forward_backward(probe_batch_or_current_batch)
        update_probe_stats_from_probe_grads(model)
        disable_probe_for_all_lowrank_modules(model)
        zero_grad_all(model)

        # 2. Compute final rank score
        for m in lowrank_modules:
            compute_rank_score(m)

        # 3. Allocate ranks
        moves = allocate_ranks(lowrank_modules)

        new_rank_dict = {m: m.rank for m in lowrank_modules}
        for m_remove, m_add, delta in moves:
            new_rank_dict[m_remove] -= delta
            new_rank_dict[m_add] += delta

        # 4. Refactorize with new ranks
        for m in lowrank_modules:
            refactorize_with_new_rank(m, new_rank_dict[m])

        # 5. Momentum reset: rebuild optimizer
        optimizer = build_optimizer(model, lr=current_lr, weight_decay=wd)

        # 6. Clear interval statistics or decay them
        reset_or_decay_interval_stats(lowrank_modules)
```

---

# 11. 实现中的关键注意事项

## 11.1 SVD 成本

对每个矩阵每 500 steps 做 SVD，可能有一定成本，但可接受。

已有论文中每 200 updates refactorize 一次，说明周期性 SVD 在你们设置中是可行的 。如果改为 500 steps，开销更低。

---

## 11.2 不同矩阵 shape 不同，rank 要按 block 对齐

建议 rank 始终是 8 或 16 的倍数：

[
r_g \in 8\mathbb{Z}
]

这样更利于 GPU kernel 和 tensor core。

初始：

[
r_g=\text{round_to_multiple}(0.25d_g,8)
]

最小：

[
r_{\min,g}=\text{round_to_multiple}(0.125d_g,8)
]

最大：

[
r_{\max,g}=\text{round_to_multiple}(0.5d_g,8)
]

---

## 11.3 Probe batch 的选择

有三种选择：

1. 当前训练 batch；
2. 下一训练 batch；
3. 固定小 probe batch。

第一版建议用当前 batch 或下一个 batch，避免额外 dataloader。更严谨版本用固定小 probe batch，因为它让不同 restart interval 的 saturation score 更可比。

credit_assignment.md 也明确区分了 training-batch credit、fresh-batch credit 和 probe-batch credit：training-batch 最接近 exact，但可能过拟合当前 batch；fresh-batch 更接近 population 但噪声更大；probe-batch 更稳定但有额外成本和 probe bias 。

---

## 11.4 Score 需要 cross-module normalization

不同矩阵 shape 不同，score 尺度可能差很多。restart 前可以对所有 score 做 z-score 或 percentile normalization：

[
\hat Q_g
========

\frac{Q_g-\operatorname{median}(Q)}
{\operatorname{MAD}(Q)+\epsilon}
]

或者直接用 rank ordering，不用绝对值。

第一版 TopK-BottomK 用排序即可。




## 12.2 指标

主要指标：

[
\text{Validation PPL}
]

辅助指标：

1. Rank distribution over time；
2. 每类模块平均 rank：embedding、Q/K/V/O、gate/up/down、LM head；
3. score (Q_g) 与 rank 变化的关系；
4. 每次 restart 前后 loss spike；
5. rank reallocation 后的短期 recovery；
6. memory usage；
7. throughput overhead。

建议画图：

* x-axis: training step；
* y-axis: module-level average rank；
* 不同颜色：operator type。

这和 SOLAR 论文里的 module-wise LR allocation 图形成呼应。SOLAR 论文中 Figure 4 展示了 module-wise LR allocation 和 LR–GradNorm relation，用于说明不是简单 scalar LR inflation，而是 module-aware control 。这里你可以做一个对应图：

> module-wise rank allocation over training.

这样叙事非常统一。



# 14. 最终算法草案

可以写成：

## Algorithm: CORA-Restart

**Input:** model with low-rank matrices (W_g=B_gA_g), initial rank (r_g=d_g/4), total budget (R), restart interval (T), credit EMA factor (\beta_c), probe EMA factor (\beta_p).

1. Initialize all low-rank matrices with uniform rank (r_g^{(0)}=d_g/4).
2. For each training step (t):

   1. Run forward/backward.
   2. Perform optimizer step.
   3. Periodically estimate update credit:
      [
      C_{t,g}^{BA}
      ============

      \langle \nabla_{B_g}L,\Delta B_g\rangle
      +
      \langle \nabla_{A_g}L,\Delta A_g\rangle
      ]
   4. Update credit efficiency EMA:
      [
      E_g
      \leftarrow
      \beta_c E_g
      +
      (1-\beta_c)
      \frac{\max(-C_{t,g},0)}
      {|\Delta B_g|_F^2+|\Delta A_g|_F^2+\epsilon}
      ]
3. If (t \mod T=0):

   1. Enable zero-output rank probes:
      [
      W_g'=B_gA_g+B_{p,g}A_{p,g}
      ]
      with (B_{p,g}=0).
   2. Run one probe forward/backward.
   3. Compute:
      [
      S_g=\frac{|\nabla_{B_{p,g}}L|*F}{|A*{p,g}|_F+\epsilon}
      ]
   4. Update probe EMA.
   5. Compute rank score:
      [
      Q_g=E_g\cdot S_g\cdot \frac{1}{\sqrt{r_g}}
      ]
   6. Move rank budget from bottom-(Q) groups to top-(Q) groups under (r_{\min},r_{\max}) and total budget constraints.
   7. For each matrix, materialize (W_g=B_gA_g), perform truncated SVD at new rank (r'*g), and set:
      [
      B'*g=U*{r'}\sqrt{\Sigma*{r'}},\qquad
      A'*g=\sqrt{\Sigma*{r'}}V_{r'}^\top
      ]
   8. Reset optimizer states for all low-rank factors.
   9. Continue training.

---

# 15. 最小可行实现版本

为了最快验证，我建议第一版做如下简化：

1. 初始 rank：所有矩阵 (r=d/4)；
2. restart interval：先用 (T=500)，如果想对齐已有论文可以也测试 (T=200)；
3. credit：只用 factor-space first-order credit；
4. probe：每次 restart 前做一次 zero-output probe；
5. score：
   [
   Q_g=
   \operatorname{EMA}
   \left(
   \frac{\max(-C_g,0)}
   {|\Delta B_g|^2+|\Delta A_g|^2+\epsilon}
   \right)
   \cdot
   \operatorname{EMA}(S_g)
   \cdot
   \frac{1}{\sqrt{r_g}}
   ]
6. allocation：TopK-BottomK，每次移动 rank 8；
7. rank 范围：
   [
   r_g\in[0.125d_g,\ 0.5d_g]
   ]
8. restart 后：SVD refactorization + 重建 AdamW optimizer。

这个版本已经足够形成实验结果。

---

# 16. 最重要的风险和应对

## 风险 1：Probe score 太 noisy

应对：

* 用 EMA；
* probe_rank 从 1 增加到 2 或 4；
* 使用固定 probe batch；
* 只根据 rank 排序，不用绝对值。

## 风险 2：SVD 开销大

应对：

* 只在 restart时做；


## 风险 3：rank reallocation 破坏训练稳定性

应对：

* 每次只移动少量 rank；
* 使用 hysteresis；
* restart 后降低 LR 10-50 steps 做 mini warmup；
* rank 变化后一定 reset optimizer states。

## 风险 4：score 偏向大矩阵

应对：

* 使用 energy-normalized credit；
* 使用 per-rank scarcity；
* 按 operator type 内部分配，比如 attention 内部、MLP 内部分配，而不是全模型一起排序。

