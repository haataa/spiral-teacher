# Spiral Teacher

> Give me a code repo, I'll teach you how it works — like a patient tutor who adapts to your level.

Spiral Teacher 是一个多 Agent 教学系统：给它一个代码仓库，它会自动生成**螺旋式深入**的教学文档。系统模拟一位耐心的教师和一位真实的学习者之间的多轮对话，然后将对话合成为结构化教程。

## 效果展示

以 [FlashAttention](https://github.com/Dao-AILab/flash-attention) 为例，生成的教程 ([完整版](examples/flash-attention/tutorial.md)) 开头：

> FlashAttention 是一个为 GPU 提供快速、省内存、IO 感知的精确注意力计算库。它的核心洞察是：标准 Transformer 的注意力计算瓶颈不在于浮点运算量，而在于 GPU 显存（HBM）的读写带宽。通过将计算分块到片上高速缓存（SRAM）中完成，并利用一种名为 Online Softmax 的增量算法避免在显存中生成完整的 N x N 注意力矩阵，FlashAttention 在不改变任何数学结果的前提下，将内存占用从 O(N^2) 降到 O(N)，同时实现了 2-4 倍的实际加速。

教程包含 7 个核心概念，从 Level 2（直觉类比）到 Level 5（跨方法对比分析），533 行，一条命令生成。

## 工作原理

```
代码仓库 → Reader (知识提取) → Teacher ↔ Learner (多轮对话) → Synthesizer (教程合成)
```

- **Reader** 扫描仓库，提取概念、依赖关系和教学顺序
- **Teacher** 按重要性逐概念讲解，支持 6 个深度层级（一句话概括 → 数学推导）
- **Learner** 模拟目标读者：提出困惑、构建错误假设、要求具体例子、追问细节
- **Synthesizer** 将对话重组织为连贯教程，保留好的类比和"你可能以为...但实际上..."

Learner 是系统的核心——三层防宽容机制确保它不会"手下留情"，迫使 Teacher 给出真正有深度的解释。

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

[MIT](LICENSE)
