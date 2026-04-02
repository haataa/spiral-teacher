# Spiral Teacher

基于多 Agent 协作的自适应教学内容生成系统。模拟"螺旋式深入"的人类学习互动过程，自动为代码仓库生成多层级教学文档。

## Quick Start

```bash
# 安装
pip install -e .

# 生成教程（默认中文）
spiral-teacher generate --repo /path/to/your/repo

# 断点续跑（从上次停下的概念继续）
spiral-teacher generate --repo /path/to/your/repo --resume --max-rounds 10

# 只重新合成教程（不重跑对话）
spiral-teacher synthesize --output output
```

需要设置 `ANTHROPIC_API_KEY` 环境变量。

## How It Works

```
代码仓库 → Reader (提取知识图谱) → Teacher ↔ Learner (多轮对话) → Synthesizer (合成教程)
```

1. **Reader** 扫描仓库，用 LLM 提取概念、依赖关系和教学顺序
2. **Teacher** 按重要性顺序逐概念讲解（Level 0-5，从直觉到严格推导）
3. **Learner** 模拟目标读者的理解过程，提出困惑、错误假设、要求深入或举例
4. **Synthesizer** 将对话重组织为连贯的教学文档（保留好的类比和"你可能以为...但实际上..."）

## CLI Options

```
spiral-teacher generate
  --repo PATH              代码仓库路径（必需）
  --topic TEXT              聚焦主题（可选）
  --audience TEXT           读者画像 (default: ml_engineer)
  --language TEXT           输出语言 (default: zh)
  --output PATH             输出目录 (default: output)
  --max-rounds N            最大对话轮次 (default: 20)
  --max-rounds-per-concept N 单概念最大轮次 (default: 4)
  --resume                  从上次运行继续
  --no-synthesize           跳过教程合成步骤

spiral-teacher synthesize
  --output PATH             包含 conversation.json 和 knowledge.json 的目录
  --audience TEXT            读者画像
  --language TEXT            输出语言
```

## Output Files

运行后在 `output/` 目录生成：

| 文件 | 生成时机 | 内容 |
|------|---------|------|
| `knowledge.json` | Reader 完成后立即 | 概念列表、依赖关系、教学顺序 |
| `conversation.md` | 每条对话即时追加 | Markdown 格式的完整对话记录（可实时查看） |
| `conversation.json` | 每个概念完成后更新 | 结构化对话数据（用于 resume 和 synthesize） |
| `tutorial.md` | Synthesizer 完成后 | 最终教学文档 |
| `summary.txt` | 全部完成后 | 统计摘要 |

## Cost

约 $5-8 / 完整教程（20-40 轮对话）。可通过 `--max-rounds` 控制。

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -m "not integration"  # 117 unit tests
```

详见 [CLAUDE.md](CLAUDE.md) 的开发指引和 [PLAN.md](PLAN.md) 的项目路线图。
