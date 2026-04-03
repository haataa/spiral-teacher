# FlashAttention 深度教程

## Overview

FlashAttention 是一个为 GPU 提供**快速、省内存、IO 感知**的精确注意力计算库。它的核心洞察是：标准 Transformer 的注意力计算瓶颈不在于浮点运算量，而在于 GPU 显存（HBM）的读写带宽。通过将计算分块到片上高速缓存（SRAM）中完成，并利用一种名为 Online Softmax 的增量算法避免在显存中生成完整的 $N \times N$ 注意力矩阵，FlashAttention 在不改变任何数学结果的前提下，将内存占用从 $O(N^2)$ 降到 $O(N)$，同时实现了 2-4 倍的实际加速。本教程将从直觉出发，逐步深入到算法细节、数值分析和工程实现，帮助你彻底理解这项技术。

## Core Concepts at a Glance

我们将按照以下逻辑链条展开——每一步自然地引出下一步的问题：

1. **Scaled Dot-Product Attention** — FlashAttention 精确实现的基础计算。不理解它就无法理解"优化了什么"。
2. **IO 感知与内存层级** — 标准注意力为什么慢？答案不是"计算太多"，而是"搬数据太多"。
3. **分块（Tiling）策略** — 既然搬数据是瓶颈，就把计算拆成小块，全在高速缓存里完成。但 softmax 需要全局信息，怎么办？
4. **Online Softmax** — 一种增量算法，让分块 softmax 成为可能，且结果精确等价于一次性计算。
5. **线性内存复杂度** — Tiling + Online Softmax 的直接收益：$N \times N$ 矩阵"消失了"。
6. **FlashAttention Python API** — 理解了底层原理后，来看用户实际怎么调用。
7. **反向传播与重计算** — 前向不存 attention 矩阵了，反向怎么算梯度？答案：重算，而且更快。

---

## 1. Scaled Dot-Product Attention

### Why This Matters

这是 Transformer 的核心计算单元——FlashAttention 精确实现的就是这个公式。不理解它，就无法理解后续的每一个优化到底在"优化什么"。

### The Key Insight

Attention 的本质是一个"加权查询"操作：给定一组查询（Q）、一组键（K）和一组值（V），计算每个查询应该"关注"哪些值，然后按权重加权求和。

$$O = \text{softmax}\!\left(\frac{Q \, K^T}{\sqrt{d_k}}\right) V$$

### How It Works

逐步拆解这个公式：

**第一步：$QK^T$ — 计算相关性分数。** 每个 query 向量与每个 key 向量做点积，得到一个 $N \times N$ 的"匹配分数"矩阵。点积越大，表示 query 和 key 越"匹配"。

假设 $d_k = 4$，一个 query $[1, 0, 1, 0]$，两个 key $[1, 0, 1, 0]$ 和 $[0, 1, 0, 1]$：
- 与 key₁ 的点积：$1+0+1+0 = 2$（很匹配）
- 与 key₂ 的点积：$0+0+0+0 = 0$（完全不匹配）

**第二步：除以 $\sqrt{d_k}$ — 控制方差。** 如果 Q 和 K 的每个分量都是均值为 0、方差为 1 的独立随机变量，那么它们点积的方差等于 $d_k$。除以 $\sqrt{d_k}$ 将方差拉回 1，防止 softmax 进入饱和区导致梯度消失。

**第三步：Softmax — 归一化为概率分布。** 对 $[1, 0]$ 做 softmax：$e^1/(e^1+e^0) \approx 0.731$, $e^0/(e^1+e^0) \approx 0.269$。每个 query 把 73.1% 的注意力放在第一个位置，26.9% 放在第二个。

**第四步：乘以 $V$ — 按权重融合信息。** 用上面算出的权重对 value 向量做加权求和，得到融合了上下文信息的输出。

### 与 FlashAttention 的关系

**FlashAttention 计算的数学结果与上面的公式完全一致。** 它不是近似算法。如 `README.md` 中所述：

> The main functions implement scaled dot product attention (softmax(Q @ K^T * softmax_scale) @ V)

在 `flash_attn/__init__.py:8-15` 中导出的 `flash_attn_func` 等函数，默认的 `softmax_scale` 就是 `1 / sqrt(headdim)`。FlashAttention 的"魔法"不在于改变了**算什么**，而在于改变了**怎么算**。

| 步骤 | 做什么 | 为什么 |
|------|--------|--------|
| $QK^T$ | 计算相关性分数 | 衡量每对 query-key 的匹配程度 |
| $\div \sqrt{d_k}$ | 缩放 | 控制分数的方差，防止 softmax 饱和 |
| softmax | 归一化 | 将分数转为概率权重 |
| $\times V$ | 加权求和 | 根据权重融合信息 |

---

## 2. IO 感知与内存层级

### Why This Matters

现在我们知道了 attention 计算的四步。下一个关键问题是：**这四步的计算量并不大，那为什么标准实现还是很慢？** 理解这个问题，就理解了 FlashAttention 存在的全部理由。

### The Key Insight

**标准 attention 是 memory-bound（内存带宽瓶颈）的，不是 compute-bound（计算瓶颈）的。** GPU 的计算核心大部分时间在"等数据从显存搬过来"，而不是在忙着算。

GPU 上有两层存储：

| 存储层 | 名称 | 容量 | 带宽 |
|--------|------|------|------|
| 片上缓存 | **SRAM**（Shared Memory） | 很小（A100 全芯片约 20MB） | 极快（~19 TB/s） |
| 显存主存 | **HBM** | 很大（A100 上 40-80GB） | 相对慢（~2 TB/s） |

SRAM 的带宽约是 HBM 的 **10 倍**。

### 厨房做菜的类比

想象你在一个厨房里做菜。**灶台**（台面空间很小）= SRAM，在灶台上切菜、炒菜非常快。**冰箱**（容量很大，在隔壁房间）= HBM，每次去冰箱拿东西都需要走一趟，非常耗时。

标准 Attention 的做法相当于：
1. 去冰箱取出 Q 和 K → 在灶台上算 $QK^T$ → 把整个 $N \times N$ 的分数矩阵**送回冰箱**
2. 再从冰箱取出分数矩阵 → 在灶台上做 softmax → 把结果**再送回冰箱**
3. 再从冰箱取出 softmax 结果和 V → 在灶台上做加权求和 → 把最终输出**送回冰箱**

炒菜本身（计算）很快，但你大部分时间都花在**跑腿搬东西**上了。

### 具体数字验证

假设 $N = 2048$，$d = 64$，FP16：

- **计算 $QK^T$ 的 FLOPs**：约 5.4 亿次浮点运算
- **$N \times N$ 矩阵在 HBM 上的读写**：矩阵大小 8MB，被读写 4 次共 32MB
- A100 的 HBM 带宽约 2 TB/s，传输 32MB 需约 **16 微秒**
- 5.4 亿次 FP16 运算在 A100 上只需约 **1.7 微秒**

**搬数据的时间是计算时间的近 10 倍！** GPU 的计算核心大部分时间在空等。

### Common Pitfall

> **你可能会想**："FlashAttention 更快，是因为它减少了浮点运算量。"
>
> **实际恰恰相反。** FlashAttention 做了**相同甚至更多的 FLOPs**（分块 softmax 需要额外的修正计算）。但它大幅减少了 HBM 的读写量，而 HBM 读写才是真正的瓶颈。

正如 `README.md` 论文标题所示：**"Fast and Memory-Efficient Exact Attention with IO-Awareness"**——"IO-Awareness"就是指算法在设计时就把内存读写成本作为首要优化目标。

### SRAM 为什么放不下整个注意力矩阵？

你可能会问：20MB 的 SRAM 不是能放下一个 8MB 的矩阵吗？

关键在于：**SRAM 是分散在每个 SM（Streaming Multiprocessor）上的**。A100 有 108 个 SM，每个 SM 的 shared memory 最大约 **164 KB**。整栋楼的厨房总面积有 20 平米，但被 108 个厨师分了，每人只有不到 0.2 平米的灶台。

所以单个 SM 只有 ~164 KB，而 $N=2048$ 的注意力矩阵就有 8 MB——**比单个 SM 的 SRAM 大了约 50 倍**。更不用说现代模型动辄 $N = 8192$ 或更长。

| 序列长度 N | 注意力矩阵大小（FP16） |
|-----------|----------------------|
| 2,048 | 8 MB |
| 8,192 | 128 MB |
| 65,536 | 8 GB |
| 131,072 | 32 GB |

这就是为什么 FlashAttention 必须**分块（tiling）**——每次只计算一小块，使这小块能放进单个 SM 的 shared memory。

---

## 3. 分块（Tiling）策略

### Why This Matters

上一节我们确认了两件事：(1) 标准 attention 的瓶颈在 HBM 读写；(2) 整个 $N \times N$ 矩阵放不进 SRAM。那怎么办？答案就是 tiling：把大计算拆成小块，每块都在 SRAM 中完成。

### The Key Insight

FlashAttention 将 Q 按行分成若干组（每组 $B_r$ 行），将 K/V 按行分成若干组（每组 $B_c$ 行），形成一个"块网格"。每个 $B_r \times B_c$ 的小块可以完全放在 SRAM 里。

```
         K 块 0      K 块 1      K 块 2    ...
Q 块 0  [128×128]  [128×128]  [128×128]
Q 块 1  [128×128]  [128×128]  [128×128]
Q 块 2  [128×128]  [128×128]  [128×128]
  ...
```

**关键创新：这些小块在 SRAM 里被计算、被用完、然后被丢弃——$N \times N$ 矩阵从未完整出现在 HBM 中。**

### How It Works：双层循环结构

FlashAttention-2 的计算流程是：

```
外层循环：for each Q 块 i = 1, 2, ..., T_r
    从 HBM 加载 Q_i 到 SRAM
    初始化统计量 m_i = -∞, ℓ_i = 0, o_i = 0
    内层循环：for each K/V 块 j = 1, 2, ..., T_c
        从 HBM 加载 K_j, V_j 到 SRAM
        在 SRAM 中计算 S_ij = Q_i × K_j^T  (B_r × B_c 的小矩阵)
        用 Online Softmax 更新 (m_i, ℓ_i, o_i)
        丢弃 S_ij（不写回 HBM）
    内层循环结束 → o_i 是精确的最终输出
    将 o_i / ℓ_i 写回 HBM（只写一次！）
```

### 理解独立性：softmax 是按行操作的

你可能会担心：不同 Q 块之间的 softmax 会不会互相影响？

**答案是不会。** softmax 在 attention 中是**按行**操作的——每个 query 位置独立地对所有 key 位置做归一化。第 1 行的 softmax 分母和第 129 行的 softmax 分母毫无关系。所以：

- **外层循环**（遍历 Q 块）：纯粹的并行/顺序遍历，块之间独立
- **内层循环**（遍历 K/V 块）：这里是 Online Softmax 发挥作用的地方——同一个 Q 块需要"看完"所有 K/V 块，才能确定最终的 softmax 归一化

### 块大小怎么选？

块大小 $B_r$ 和 $B_c$ 由 SRAM 容量决定。需要同时在 SRAM 中放下：

| 数据 | 大小（FP16） |
|------|-------------|
| Q 块 | $B_r \times d \times 2$ bytes |
| K 块 | $B_c \times d \times 2$ bytes |
| V 块 | $B_c \times d \times 2$ bytes |
| Scores 块 $S_{ij}$ | $B_r \times B_c \times 2$ bytes |
| 输出 O 块 | $B_r \times d \times 2$ bytes |

论文给出理论上界：$B_c = \lceil M/(4d) \rceil$，$B_r = \min(\lceil M/(4d) \rceil, d)$，其中 $M$ 是 SRAM 大小（以 elements 计）。

以 A100、$d=128$、$B_r=128$、$B_c=64$ 为例验证 SRAM 预算：

- Q_i: 128×128×2 = 32 KB
- K_j: 64×128×2 = 16 KB
- V_j: 64×128×2 = 16 KB
- S_ij: 128×64×2 = 16 KB
- O_i: 128×128×2 = 32 KB
- **总计约 112 KB**，在 164 KB 限制内 ✅

注意 $d=128$ 时**不是** $B_r = B_c = 128$（那需要约 192 KB，超标），而是牺牲 $B_c$ 到 64。这也解释了为什么 `setup.py:203-262` 中按 `hdim32/64/96/128/192/256` 分别编译不同的 CUDA kernel——不同的 $d$ 对应不同的最优分块参数。

### FlashAttention-1 vs FlashAttention-2 的循环顺序

两个版本的循环顺序恰好相反：

- **FA-1**：外层遍历 K/V 块，内层遍历 Q 块
- **FA-2**：外层遍历 Q 块，内层遍历 K/V 块

为什么 FA-2 更好？差别在于**输出矩阵 O 的读写模式**：

- FA-2（外层 Q）：$O_i$ 在外层循环开始时在 SRAM 中创建，整个内层循环中一直驻留 SRAM，最后一次写回 HBM
- FA-1（外层 K/V）：$O_i$ 在每次内层迭代时都要从 HBM 读出、更新、再写回——因为下一步要换到不同的 Q 块，SRAM 空间要让给新的 Q 块

**O 作为一个可变累积量（read-modify-write），其读写模式是两种方案差异的主要来源。** 以 $N=1024$、$B_r=128$、$B_c=64$ 为例：

| 指标 | FA-1（外层 K/V） | FA-2（外层 Q） |
|------|-----------------|---------------|
| O 每个块 HBM 读取次数 | $T_c = 16$ 次 | 0 次 |
| O 每个块 HBM 写入次数 | $T_c = 16$ 次 | **1 次** |

仅 O 矩阵一项就节省了数千 KB 的 HBM 传输。

---

## 4. Online Softmax（数值稳定的增量 Softmax）

### Why This Matters

Tiling 的核心矛盾是：softmax 需要**全行的信息**（分母是整行的指数和），但每次只看一个 K/V 块。如果这个矛盾无法解决，分块计算就不可能精确。Online Softmax 就是解决这个矛盾的数学工具。

### The Key Insight

利用指数函数的一个代数恒等式——$\exp(a - c) = \exp(a - b) \cdot \exp(b - c)$——在全局最大值发生变化时，可以精确追溯修正之前所有的计算结果。这不是近似，是**精确的数学变换**。

### How It Works

算法维护三个"运行中的统计量"：
- $m$：到目前为止见过的最大 score 值
- $\ell$：以 $m$ 为基准的指数和
- $\mathbf{o}$：以 $m$ 为基准的加权输出累加器

每处理一个新的 K/V 块：

```
1. 计算局部分数: S_block = Q_i · K_j^T
2. 找局部最大值: m_local = max(S_block)
3. 更新全局最大值: m_new = max(m, m_local)
4. 修正因子: α = exp(m - m_new)
5. 修正旧的指数和: ℓ = α · ℓ + sum(exp(S_block - m_new))
6. 修正旧的输出: o = α · o + exp(S_block - m_new) · V_j
7. 更新 m = m_new
```

修正因子 $\alpha = \exp(m_\text{old} - m_\text{new})$ 的作用：**把之前所有项的基准从旧最大值替换为新最大值。** 由于 $e^{s - m_\text{old}} \cdot e^{m_\text{old} - m_\text{new}} = e^{s - m_\text{new}}$，这正是一次精确的基底变换。

### 数值验证

取 8 个 scores 分成两块：Block 1 = $[2, 4, 1, 3]$，Block 2 = $[5, 1, 3, 2]$。

**处理 Block 1**：$m_1 = 4$，$\ell_1 = e^{-2} + 1 + e^{-3} + e^{-1} = 1.553$

**处理 Block 2**：$m_\text{local} = 5 > m_1 = 4$，所以 $m_2 = 5$

修正因子 $\alpha = e^{4-5} = 0.368$

$\ell_2 = 0.368 \times 1.553 + (1 + e^{-4} + e^{-2} + e^{-3}) = 0.571 + 1.203 = 1.774$

**直接算全局 softmax**：$m = 5$，$\ell = e^{-3} + e^{-1} + e^{-4} + e^{-2} + 1 + e^{-4} + e^{-2} + e^{-3} = 1.774$ ✅

**完全一致！**

### 正确性证明（归纳法要点）

**命题**：处理完前 $t$ 个块后，$m^{(t)} = \max_{j \in \mathcal{S}_t} s_j$，$\ell^{(t)} = \sum_{j \in \mathcal{S}_t} e^{s_j - m^{(t)}}$。

**归纳步骤**：$e^{m^{(t-1)} - m^{(t)}} \cdot \ell^{(t-1)}$ 利用指数乘法律，每一项 $e^{s_k - m^{(t-1)}} \cdot e^{m^{(t-1)} - m^{(t)}} = e^{s_k - m^{(t)}}$，加上新块的贡献，就得到 $\ell^{(t)} = \sum_{j \in \mathcal{S}_t} e^{s_j - m^{(t)}}$。$\square$

### 额外 FLOPs 的量级

Online Softmax 引入的额外计算主要是步骤 6 中对 $\mathbf{o}$（$B_r \times d$ 大小）的逐元素缩放，每块一次。额外 FLOPs 与主计算量的比值约为 $1/B_c$。当 $B_c = 64 \sim 256$ 时，**额外 FLOPs 仅为主计算量的 0.4%-1.6%**，且与 $N$ 无关。

### FA-1 vs FA-2 的输出累积策略

两种实现有一个微妙但重要的差异：

- **FA-1**：维护未归一化的 $\tilde{\mathbf{o}} = \ell \cdot \mathbf{o}$，最后才除以 $\ell$
- **FA-2**：始终维护归一化的 $\mathbf{o}$，每步除以 $\ell_\text{new}$

### Common Pitfall

> **你可能会想**：FA-1 和 FA-2 只是代码风格差异，数值上应该没区别。
>
> **实际上区别很大。** FA-1 的 $\tilde{\mathbf{o}} = \ell \cdot \mathbf{o}$ 量级约为 $N \times \|V\|$。当 $N$ 很大时，FP16 的有限精度会导致**大数吃小数（swamping）**：后续 token 的微小贡献在加到很大的 $\tilde{\mathbf{o}}$ 上时被舍入为零。例如，当 $\tilde{o} \approx 32768$ 时，FP16 相邻可表示数的间距约为 32，任何小于 16 的新贡献都直接消失。
>
> FA-2 始终维护归一化的 $\mathbf{o}$（value 的凸组合，量级 $\sim \|V\|$），完全避免了这个问题。实践中，kernel 内部的累加器使用 FP32，FA-2 的归一化设计确保即使在 FP32 下也永远不会 swamp。

---

## 5. 线性 vs 二次内存复杂度

### Why This Matters

前面我们从底层理解了 tiling 和 online softmax。现在来看它们带来的最重要的实际收益：**内存从 $O(N^2)$ 降到 $O(N)$**。这不是渐近理论，而是决定了"你的模型能不能训练"的硬约束。

### The Key Insight

标准 attention 的三个独立 kernel 之间，$N \times N$ 的中间矩阵必须在 HBM 中"活着"。FlashAttention 用 tiling + online softmax 把三步融合成单个 kernel，中间矩阵只在 SRAM 中短暂存在（$B_r \times B_c$ 大小），用完即丢。

### 具体数字对比

假设 $B=4$、$h=32$、$d=128$、$N=8192$、FP16：

| 张量 | 标准 Attention | FlashAttention |
|------|---------------|----------------|
| Q, K, V | 768 MB | 768 MB |
| $S = QK^T$（$N \times N$） | **16 GB** | **0** |
| $P = \text{softmax}(S)$（$N \times N$） | **16 GB** | **0** |
| O（输出） | 256 MB | 256 MB |
| logsumexp 统计量 | — | 4 MB |
| **总计** | **≈ 33 GB** | **≈ 1 GB** |

**$S$ 和 $P$ 两个 $N \times N$ 矩阵占标准实现总内存的 97%。** FlashAttention 直接把它们消灭了。

当 $N$ 增长时差异更加惊人：

| 序列长度 $N$ | 标准额外内存（$S + P$） | FlashAttention 额外内存（LSE） | 节省倍数 |
|---|---|---|---|
| 2,048 | 1 GB | 0.5 MB | 2048× |
| 8,192 | 16 GB | 2 MB | 8192× |
| 65,536 | **1 TB** | 16 MB | 65536× |

$N = 65536$ 时，标准 attention 需要 1TB——A100 只有 80GB 显存，根本无法运行。而 FlashAttention 只需约 8GB，轻松放入。**没有 FlashAttention，超长序列训练不只是"慢"，而是物理不可能。**

### 计算量不变

必须再次强调：**FLOPs 完全不变，仍然是 $O(N^2 d)$。** 每一对 token 之间的注意力分数还是都要算的。速度提升 100% 来自 HBM 读写次数减少。

如 README 中的实测数据：

> Memory savings are proportional to sequence length. We see **10X memory savings at sequence length 2K, and 20X at 4K.**

节省倍数正比于 $N$，因为 $O(N^2) / O(N) = O(N)$。

---

## 6. FlashAttention Python API

### Why This Matters

理解了底层原理后，现在从用户视角来看：这个库到底给你暴露了哪些函数，什么时候用哪个？

### The Key Insight

FlashAttention 的底层 CUDA kernel 只做一件事：高效计算 scaled dot-product attention。但用户的 Q、K、V 数据可能以不同方式组织。Python API 为不同场景提供方便的入口。

### How It Works

从 `flash_attn/__init__.py:8-16` 可以看到 7 个导出函数，本质上是两个维度的组合：

**打包维度** × **序列维度** + 推理专用变体

| 函数 | 布局 | 典型场景 |
|------|------|---------|
| `flash_attn_func` | Q, K, V 分离 | 通用训练/推理 |
| `flash_attn_qkvpacked_func` | Q+K+V 全打包 | 标准自注意力训练（反向更快） |
| `flash_attn_kvpacked_func` | K+V 打包，Q 分离 | Cross-attention |
| `flash_attn_varlen_func` | 分离 + 变长序列 | batch 内序列长度不等 |
| `flash_attn_varlen_qkvpacked_func` | 全打包 + 变长 | 变长自注意力 |
| `flash_attn_varlen_kvpacked_func` | KV 打包 + 变长 | 变长 cross-attention |
| `flash_attn_with_kvcache` | Q + KV cache | 推理自回归解码 |

**选择决策树：**

```
推理/解码？ → flash_attn_with_kvcache
训练：
  ├── batch 内序列等长？
  │   ├── QKV 已打包？ → qkvpacked（反向最快）
  │   └── 三者分离？   → func
  └── batch 内序列变长？ → varlen 系列（避免 padding 浪费）
```

核心原则：**能 pack 就 pack**（反向传播避免显式 concatenation），**能不 pad 就不 pad**（前向计算不浪费）。

### Common Pitfall

> **你可能会想**：`qkvpacked` 和 `func` 会产生不同的注意力结果。
>
> **事实上它们在数学上完全等价。** 差异仅在反向传播的工程实现效率上——`qkvpacked` 知道梯度在内存中连续，可以直接写入一个张量，避免 concat 操作。

所有函数共享以下关键参数：
- `softmax_scale`：默认 $1/\sqrt{d}$
- `causal`：因果掩码（自回归必须开启）
- `window_size`：滑动窗口注意力
- `deterministic`：确定性反向传播（稍慢但结果可复现）

---

## 7. FlashAttention 反向传播（重计算）

### Why This Matters

前向传播用 tiling + online softmax 成功避免了在 HBM 中物化 $N \times N$ 矩阵。但反向传播需要 attention 矩阵 $P$ 来计算 Q、K、V 的梯度——如果把 $P$ 存回去，之前所有努力就前功尽弃了。

### The Key Insight

**不存 $P$，而是在反向传播时逐 block 重算。** 前向传播只额外保存一个"体积极小的统计摘要"——logsumexp（每行一个 FP32 标量），反向传播利用它按需恢复任意 block 的 $P_{ij}$。

这就像考试时只在答题卡上记最终答案和一个"校验码"，草稿纸用完就扔。需要检查时，根据原始题目和校验码重新演算。

### How It Works：数学推导

设 $S = QK^T$，$P = \text{softmax}(S)$，$O = PV$。backward 需要计算 $dQ, dK, dV$。

**dV 最简单**：$dV = P^T \cdot dO$

**dP**：$dP = dO \cdot V^T$

**从 dP 到 dS**（这是关键！）：softmax 的 Jacobian 包含一个"全行依赖"项 $P_i P_i^T$。但经过巧妙化简后：

$$dS_{ij} = P_{ij}(dP_{ij} - D_i)$$

其中 $D_i = \sum_k dO_{ik} \cdot O_{ik}$ 是 $dO$ 和 $O$ 的逐行点积。

**这个化简是核心**：它利用了 $O_i = \sum_j P_{ij} V_j$ 这个恒等式，把需要遍历所有 $j$ 的内积转化为只需看 $O_i$ 和 $dO_i$ 的局部量。

所以 softmax 梯度的"全行依赖"被分解为**两个全局标量**：
- **$\text{LSE}_i$**（从 forward 保存）：恢复 $P_{ij} = \exp(S_{ij} - \text{LSE}_i)$
- **$D_i$**（backward 开始时预计算）：$D_i = dO_i \cdot O_i$

有了这两个标量，剩下的所有计算都是 block-local 的——tiling 完美工作。

### logsumexp 的保存：forward 和 backward 的接口契约

你可能会问：logsumexp 是前向 online softmax 过程中"顺手"记录的，还是需要特殊设计？

答案是**几乎顺手**：online softmax 本来就维护 $m$（最大值）和 $\ell$（指数和），处理完最后一个 block 后，只需执行一次 `LSE = m + log(ℓ)`，以 FP32 精度写回 HBM。这是一条指令的事。

| 保存项 | 大小 | 精度 |
|--------|------|------|
| LSE | $(B, h, N)$ | FP32（必须） |

存储量对比：以 $B=4, h=32, N=8192$ 为例，LSE 占 4MB，而标准实现存储 $P$ 矩阵需要 16GB——**差了 4000 倍**。

### FlashAttention-2 的 Two-Pass Backward

单 pass backward 存在一个 IO 不对称性：无论循环顺序怎么选，三个梯度（dQ、dK、dV）中至少有一个必须在 HBM 中做 read-modify-write：

| 循环顺序 | SRAM 累加（高效） | HBM read-modify-write（低效） |
|---------|-------------------|-------------------------------|
| 外层 K/V，内层 Q | dK, dV（2 个矩阵） | dQ（1 个矩阵） |
| 外层 Q，内层 K/V | dQ（1 个矩阵） | dK, dV（2 个矩阵） |

FA-2 的解决方案是 **two-pass backward**：

- **Pass 1**：外层 K/V，内层 Q → dK/dV 在 SRAM 中累加
- **Pass 2**：外层 Q，内层 K/V → dQ 在 SRAM 中累加

三个梯度都享受了 SRAM 累加 + 一次写出的待遇，彻底消除了 read-modify-write。代价是 $S$ 和 $P$ 的重计算量翻倍，但由于 FlashAttention 是 compute-bound，这个代价被 IO 节省的收益超越。实测加速约 1.3-1.5×。

---

## Putting It All Together

让我们用一个完整的端到端场景，追踪数据如何流过所有概念。

### 场景：GPT 模型训练，$N=4096$，$d=128$，$h=32$，$B=4$，FP16，A100 GPU

**第一步：用户调用 API**

```python
from flash_attn import flash_attn_func
output = flash_attn_func(q, k, v, causal=True)
```

调用 `flash_attn_func`（Q/K/V 分离布局）。`causal=True` 启用因果掩码。

**第二步：确定分块参数**

$d=128$，SRAM 约 164 KB。SRAM 预算分析得出 $B_r=128$, $B_c=64$（总占用约 112 KB）。加载对应的 `flash_fwd_hdim128_fp16_causal_sm80.cu` kernel。

$T_r = 4096/128 = 32$ 个 Q 块，$T_c = 4096/64 = 64$ 个 K/V 块。

**第三步：前向传播（Tiling + Online Softmax）**

对每个 Q 块 $i$（外层循环，32 次）：
- 从 HBM 加载 $Q_i$（32 KB）到 SRAM
- 初始化 $m_i = -\infty$，$\ell_i = 0$，$o_i = 0$

  对每个 K/V 块 $j$（内层循环，64 次）：
  - 从 HBM 加载 $K_j, V_j$（各 16 KB）到 SRAM
  - 在 SRAM 中计算 $S_{ij} = Q_i K_j^T$（128×64 矩阵，16 KB）
  - Online Softmax 更新 $(m_i, \ell_i, o_i)$
    - 如果 $S_{ij}$ 中有更大的值 → $\alpha = \exp(m_\text{old} - m_\text{new})$ 修正旧的 $\ell_i$ 和 $o_i$
    - 否则 $\alpha = 1$，直接累加
  - **丢弃 $S_{ij}$——从不写回 HBM**

- 内层循环结束：$o_i / \ell_i$ 就是精确的 attention 输出
- 将 $o_i$ 写回 HBM（32 KB，写一次），保存 $\text{LSE}_i = m_i + \log(\ell_i)$（FP32，0.5 KB）

**内存占用**：Q/K/V/O 各 32MB + LSE 2MB ≈ **130 MB**。标准 attention 还需要额外 4GB 的 $S$ 和 $P$ 矩阵——**FlashAttention 节省了 97%**。

**第四步：反向传播（重计算）**

损失函数反传到 $dO$。

预计算 $D_i = \text{rowsum}(dO \odot O)$（4096 个标量，一次完成）。

Two-pass backward：
- **Pass 1**（外层 K/V）：逐 block 重算 $S_{ij}$，用 LSE 恢复 $P_{ij} = \exp(S_{ij} - \text{LSE}_i)$，累加 $dK_j$ 和 $dV_j$ 在 SRAM 中
- **Pass 2**（外层 Q）：同样重算，累加 $dQ_i$ 在 SRAM 中

整个反向传播过程中，$N \times N$ 的 $P$ 矩阵从未完整存在于 HBM——它以 128×64 的小块形式在 SRAM 中短暂出现，用完即弃。

---

## Key Takeaways

1. **Scaled Dot-Product Attention** — FlashAttention 计算的是**完全相同的数学公式**，精确到浮点精度级别。它不是近似算法。

2. **IO 感知** — 标准 attention 的瓶颈不是计算量而是 HBM 带宽。A100 上搬数据的时间可以是计算时间的 10 倍。优化的正确方向是减少 HBM 读写，而不是减少 FLOPs。

3. **分块策略** — 将 Q/K/V 切成小块，每块在 SRAM 中完成所有中间计算。块大小由 SRAM 容量和 head dimension 共同决定，按 hdim 编译时硬编码到不同的 CUDA kernel 中。

4. **Online Softmax** — 利用 $\exp(a-c) = \exp(a-b) \cdot \exp(b-c)$ 的恒等式，在最大值更新时精确追溯修正之前的结果。额外 FLOPs 不到 1%，且不随 $N$ 增长。FA-2 始终维护归一化输出以避免 FP16 下的 swamping。

5. **线性内存** — $N \times N$ 的 attention 矩阵从未在 HBM 中出现，内存从 $O(N^2)$ 降到 $O(N)$。$N=65536$ 时节省超过 65000 倍，从"物理不可能"变为"轻松放入"。

6. **Python API** — 7 个函数是"打包维度 × 序列维度 + 推理变体"的二维组合。数学上完全等价，差异在工程效率。能 pack 就 pack，能不 pad 就不 pad。

7. **反向传播重计算** — 前向只保存 logsumexp（$O(N)$ 的 FP32 标量），反向逐 block 重算 $P_{ij} = \exp(S_{ij} - \text{LSE}_i)$。FA-2 的 two-pass backward 让三个梯度全部在 SRAM 中累加，彻底消除 read-modify-write。用 ~25% 的额外计算换取数千倍的内存节省——在 GPU 上这笔买卖极其划算。