![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-green.svg)
![CI](https://github.com/haataa/spiral-teacher/actions/workflows/ci.yml/badge.svg)

# Spiral Teacher

> Give me a code repo, I'll teach you how it works — like a patient tutor who adapts to your level.
>
> 给我一个代码仓库，我把它变成一份从浅到深的教学文档。

A multi-agent system that reads a code repository and generates **adaptive-depth teaching tutorials**. Unlike single-pass LLM explainers, Spiral Teacher simulates a multi-round Teacher-Learner dialogue with built-in anti-leniency mechanisms — the simulated learner pushes back, asks for examples, and demands deeper explanations until the teaching truly reaches the underlying principles. The conversation is then synthesized into a structured tutorial document. One command, one tutorial.

**Key differentiators vs. asking ChatGPT / feeding code to an LLM:**
- **Adaptive depth** (Level 0–5): from one-line summary to mathematical derivation, automatically calibrated
- **Anti-leniency guardrails**: triple-layer validation prevents shallow "got it" responses
- **Importance-first ordering**: core algorithms first, CLI boilerplate last
- **Audience profiles**: same repo → different tutorials for different readers

---

你有没有过这种体验：打开一个 star 很高的开源项目，想理解它的核心设计，但 README 只有 API 文档，源码注释稀疏，只能从 `main.py` 开始一个文件一个文件地跳转，两小时后对架构仍然一头雾水？

Spiral Teacher 用一条命令解决这个问题。它模拟一位教师和一位**挑剔的**学习者之间的多轮对话，从最核心的概念开始，逐层深入，最终生成一份结构化的教学文档。

## 为什么不直接问 ChatGPT / Claude？

|  | 直接读源码 | 问 LLM | **Spiral Teacher** |
|--|-----------|--------|-------------------|
| 学习路径 | 自己摸索，容易迷失 | 你得知道问什么 | **自动规划，按重要性排序** |
| 解释深度 | 取决于注释质量 | 通常止于表面 | **内置"挑剔学习者"，追问到原理层** |
| 产出物 | 无 | 一次性聊天记录 | **可分享的完整教程文档** |
| 系统性 | 碎片化理解 | 碎片化回答 | **概念间有依赖关系，逐步构建** |

## 效果展示

以 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 为例，**一条命令生成 533 行深度教程**。

> [**点击查看完整教程 →**](examples/flash-attention/tutorial.md)

更多示例：
| 项目 | 领域 | 教程 |
|------|------|------|
| [FlashAttention](https://github.com/Dao-AILab/flash-attention) | GPU 注意力优化 | [查看](examples/flash-attention/tutorial.md) |
| [Eureka](https://github.com/eureka-research/Eureka) | LLM 自动生成 RL 奖励函数 | [查看](examples/eureka/tutorial.md) |
| [RF-Agent](https://github.com/ZhihaoAIRobotic/RF-Agent) | MCTS + LLM 奖励函数搜索 | [查看](examples/rf-agent/tutorial.md) |

教程开头：

> FlashAttention 是一个为 GPU 提供快速、省内存、IO 感知的精确注意力计算库。它的核心洞察是：标准 Transformer 的注意力计算瓶颈不在于浮点运算量，而在于 GPU 显存（HBM）的读写带宽。通过将计算分块到片上高速缓存（SRAM）中完成，并利用一种名为 Online Softmax 的增量算法避免在显存中生成完整的 N x N 注意力矩阵，FlashAttention 在不改变任何数学结果的前提下，将内存占用从 O(N^2) 降到 O(N)，同时实现了 2-4 倍的实际加速。

### 同一概念，不同深度

"螺旋式教学"意味着同一个概念会从直觉到原理逐层展开。以 **IO 感知与内存层级** 为例：

<details>
<summary><b>Level 2 — 直觉类比</b>：用厨房做菜解释 GPU 内存瓶颈</summary>

> 想象你在一个厨房里做菜。**灶台**（台面空间很小）= SRAM，在灶台上切菜、炒菜非常快。**冰箱**（容量很大，在隔壁房间）= HBM，每次去冰箱拿东西都需要走一趟，非常耗时。
>
> 标准 Attention 的做法相当于：
> 1. 去冰箱取出 Q 和 K → 在灶台上算 QK^T → 把整个 N x N 的分数矩阵**送回冰箱**
> 2. 再从冰箱取出分数矩阵 → 在灶台上做 softmax → 把结果**再送回冰箱**
> 3. 再从冰箱取出 softmax 结果和 V → 在灶台上做加权求和 → 把最终输出**送回冰箱**
>
> 炒菜本身（计算）很快，但你大部分时间都花在**跑腿搬东西**上了。

</details>

<details>
<summary><b>Level 4 — 深入机制</b>：用具体数字验证瓶颈</summary>

> 假设 N = 2048，d = 64，FP16：
>
> | 存储层 | 名称 | 容量 | 带宽 |
> |--------|------|------|------|
> | 片上缓存 | **SRAM** | A100 全芯片约 20MB | ~19 TB/s |
> | 显存主存 | **HBM** | 40-80GB | ~2 TB/s |
>
> - 计算 QK^T 的 FLOPs：约 5.4 亿次浮点运算
> - N x N 矩阵在 HBM 上的读写：矩阵大小 8MB，被读写 4 次共 32MB
> - A100 的 HBM 带宽约 2 TB/s，传输 32MB 需约 **16 微秒**
> - 5.4 亿次 FP16 运算在 A100 上只需约 **1.7 微秒**
>
> **搬数据的时间是计算时间的近 10 倍！** GPU 的计算核心大部分时间在空等。

</details>

## 工作原理

```
代码仓库 → Reader (知识提取) → Teacher ↔ Learner (多轮对话) → Synthesizer (教程合成)
```

- **Reader** 扫描仓库，提取概念、依赖关系和教学顺序
- **Teacher** 按重要性逐概念讲解，支持 6 个深度层级：

  | Level | 名称 | 做什么 |
  |-------|------|--------|
  | 0 | 一句话概括 | 用一句话说清这个概念做什么 |
  | 1 | 鸟瞰全局 | 整体流程是什么，每一步做什么（不涉及"怎么做"） |
  | 2 | 直觉类比 | 为什么这样设计？用类比和具体数字建立直觉 |
  | 3 | 机制细节 | 怎么实现的？算法步骤、伪代码、关键实现细节 |
  | 4 | 严格推导 | 数学推导、定理证明、逐行源码分析 |
  | 5 | 边界与对比 | 为什么不用其他方案？Trade-off 分析、相关工作对比 |

- **Learner** 模拟目标读者，**不会手下留情**——教程不会止于表面解释，系统会自动追问直到触及真正的原理
- **Synthesizer** 将对话重组织为连贯教程，保留好的类比和"你可能以为...但实际上..."

## 适用场景

- **新人入职**：快速理解团队代码库的核心设计，不用找人一个个问
- **研究者**：理解论文的参考实现，从算法直觉到数学推导一次讲透
- **开源维护者**：为自己的项目生成深度文档，降低贡献者门槛

## 快速开始

```bash
# 安装
uv sync

# 配置 API（二选一）
export ANTHROPIC_API_KEY=sk-ant-...          # 官方 API
# 或
export ANTHROPIC_AUTH_TOKEN=your-token       # 兼容 API 代理
export ANTHROPIC_BASE_URL=https://your-proxy.example.com

# 生成教程
uv run spiral-teacher generate --repo /path/to/repo
```

生成过程中可以实时查看 `output/conversation.md` 观察对话进展。

> **想先试试？** 选一个小仓库（<20 个文件），设 `--max-rounds 5`，费用约 $1-2。

## 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | (必需) | 代码仓库路径 |
| `--output` | `output` | 输出目录 |
| `--language` | `zh` | 输出语言 |
| `--audience` | `ml_engineer` | 读者画像 |
| `--max-rounds` | `20` | 最大对话轮次 |
| `--min-importance` | `3` | 跳过低重要性概念 (1-5) |
| `--resume` | - | 断点续跑 |

## 费用

约 **$5-8** / 完整教程（20 轮对话）。`--max-rounds` 可控制预算。

## 开发

```bash
uv sync --extra dev
uv run pytest tests/ -m "not integration"  # 127 tests
```

## License

[Apache License 2.0](LICENSE)
