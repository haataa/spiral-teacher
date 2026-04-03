# Spiral Teacher

多 Agent 协作的自适应教学系统。自动为代码仓库生成螺旋式深入的教学文档。

```
代码仓库 → Reader (知识提取) → Teacher ↔ Learner (多轮对话) → Synthesizer (教程合成)
```

## 安装

```bash
uv sync
```

## 配置

设置 Anthropic API 访问（二选一）：

```bash
# 方式一：官方 API
export ANTHROPIC_API_KEY=sk-ant-...

# 方式二：兼容 API 代理
export ANTHROPIC_AUTH_TOKEN=your-token
export ANTHROPIC_BASE_URL=https://your-proxy.example.com
```

## 使用

```bash
# 生成教程
uv run spiral-teacher generate --repo /path/to/repo

# 断点续跑
uv run spiral-teacher generate --repo /path/to/repo --resume

# 从已有对话重新合成教程
uv run spiral-teacher synthesize --output output
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--repo` | (必需) | 代码仓库路径 |
| `--output` | `output` | 输出目录 |
| `--language` | `zh` | 输出语言 |
| `--audience` | `ml_engineer` | 读者画像 |
| `--max-rounds` | `20` | 最大对话轮次 |
| `--min-importance` | `3` | 跳过低重要性概念 (1-5) |
| `--resume` | - | 从上次停下的概念继续 |
| `--no-synthesize` | - | 跳过教程合成步骤 |

### 输出

| 文件 | 说明 |
|------|------|
| `knowledge.json` | 概念、依赖关系、教学顺序 |
| `conversation.md` | 对话记录（实时更新，可边跑边看） |
| `tutorial.md` | 最终教学文档 |
| `summary.txt` | 统计摘要 |

### 费用

约 $5-8 / 完整教程（20 轮对话）。`--max-rounds` 可控制。

## 开发

```bash
uv sync --extra dev
uv run pytest tests/ -m "not integration"
```

## License

MIT
