# 合约：Reader Agent

## 目标

阅读代码仓库，提取结构化知识图谱（`Knowledge` 模型），作为后续 Teacher-Learner 对话的输入基础。

Phase 1 只支持本地代码仓库输入，论文/文档模式后续扩展。

---

## 接口

```python
class ReaderAgent:
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        client: anthropic.AsyncAnthropic | None = None,
    ):
        """
        Args:
            model: 模型 ID，开发阶段可切换为 claude-haiku-4-5
            client: 可选，注入 Anthropic 客户端（便于测试 mock）。
                    None 时从环境变量 ANTHROPIC_API_KEY 创建。
        """

    async def read_repository(
        self,
        repo_path: str,
        topic: str | None = None,
        max_input_tokens: int = 60_000,
    ) -> Knowledge:
        """扫描代码仓库，提取结构化知识图谱。

        Args:
            repo_path: 本地代码仓库路径
            topic: 可选，聚焦主题（如 "KV cache quantization"）
            max_input_tokens: 仓库内容截断阈值，默认 60K

        Returns:
            Knowledge 模型实例

        Raises:
            FileNotFoundError: repo_path 不存在
            ValueError: 仓库中没有可读取的源码文件
            ReaderError: LLM 调用失败或多次重试后仍无法解析
        """
```

使用 `async` 是因为 Anthropic SDK 的 `AsyncAnthropic` 在高并发场景性能更好，也与后续 Orchestrator 的异步编排保持一致。

---

## 实现策略

### 两阶段流程

**阶段 1：仓库扫描（代码，非 LLM）**

收集仓库信息，组装为 LLM 可消化的上下文。

1. 遍历仓库文件树（排除 .git、__pycache__、node_modules、.venv 等）
2. 读取文件内容，按优先级排序：

| 优先级 | 文件类型 | 示例 |
|--------|---------|------|
| 1（必须保留） | README、项目配置 | README.md, pyproject.toml, package.json, Cargo.toml |
| 2（高优先） | src/ 或 lib/ 下的源码，按目录深度排序（浅层优先） | src/main.py, lib/core.rs |
| 3（中优先） | 测试文件 | tests/test_core.py |
| 4（低优先） | 其他文档（README 除外） | docs/design.md |
| 丢弃 | 锁文件、生成文件、二进制、图片、数据文件 | package-lock.json, dist/, *.png, *.csv |

3. **Topic 感知截断**：当 `topic` 参数存在时，文件路径/文件名中包含 topic 关键词的文件提升一个优先级（将 topic 分词后做简单关键词匹配）。

4. **单文件截断**：单文件超过 50KB 时，保留前 500 行 + `[... truncated, total N lines ...]`。

5. **编码处理**：使用 `errors="replace"` 读取文件，对无法解码的文件跳过并记录 warning。

6. **Token 估算**：使用 `len(text) / 4` 近似估算。按优先级从高到低累加，超过 `max_input_tokens` 阈值时停止。

7. 输出格式：

```
## Repository Structure
├── src/
│   ├── main.py
│   └── utils.py
└── README.md

## File: README.md
<content>

## File: src/main.py
<content>
...
```

8. **日志输出**：通过 `logging.info` 记录扫描文件数、总估算 token 数、被截断/跳过的文件列表。

**阶段 2：知识提取（LLM 调用）**

将阶段 1 的输出作为上下文，调用 LLM 提取结构化知识。

**LLM 负责生成的字段：** `project_summary`、`concepts`（含所有子字段）、`dependencies`、`teaching_order`。

**代码层面硬编码的字段：** `source_type` 固定为 `"repository"`，`source_path` 由 `repo_path` 参数填充。

**结果解析：**
- 通过 Pydantic 解析 LLM 返回的 JSON
- 最多 3 次尝试（1 次初始 + 2 次重试）
- 重试时将 Pydantic 的 `ValidationError` 完整错误信息附加到消息中，要求模型修正
- 每次重试使用完整的上下文（不是只修正 JSON）
- 3 次尝试后仍失败，抛出 `ReaderError`，包含最后一次的原始 LLM 响应和解析错误信息

**API 错误处理：**
- 超时：设置 timeout 300 秒
- Rate limit (429)：指数退避重试，最多 3 次
- 认证错误 (401)：直接抛出，不重试
- 上下文溢出：捕获后自动将 `max_input_tokens` 减半，重新扫描后再试一次

### 关于 Claude Agent SDK 的使用

Reader 是一次性调用（非多轮对话），不需要 Agent SDK 的编排能力。直接用 Anthropic Python SDK 即可。Agent SDK 留给需要多轮编排的 Orchestrator。

---

## System Prompt 设计要点

Reader 的 system prompt 需要引导模型：

1. **识别核心概念** -- 不是列出所有函数/类，而是提取"值得教学的概念"
2. **建立依赖关系** -- 理解哪个概念是另一个的前置知识
3. **推荐教学顺序** -- 拓扑排序 + 由易到难，**必须覆盖所有概念**（`teaching_order` 应包含每个概念的 ID）
4. **标注难度和常见误解** -- `difficulty` 1-5；每个核心概念（difficulty >= 3）**至少标注一个** `common_misconception`，这些将作为下游质量检查的依据
5. **提取关键公式** -- 对涉及数学推导或算法的概念，提取核心公式到 `key_equations`（LaTeX 格式或 Python 表达式），Teacher 在 Level 4 讲解时会引用
6. **识别相关概念** -- 标注概念间的横向联系 `related_concepts`（非依赖关系，如"两种不同的量化方法"互为相关）
7. **定位源码** -- 每个概念关联到具体文件（相对路径），至少 80% 的概念有非空 `source_files`
8. **如果有 topic 参数** -- 围绕该主题聚焦，但仍覆盖必要的前置概念

Prompt 写在 `src/spiral_teacher/prompts/reader.md` 中，代码中读取。

---

## 文件结构

```
src/spiral_teacher/
├── agents/
│   └── reader.py          # ReaderAgent 类实现
├── prompts/
│   └── reader.md          # Reader system prompt
└── utils.py               # scan_repository() 等工具函数
```

- `utils.py`：仓库扫描逻辑（文件遍历、优先级排序、截断、token 估算）
- `agents/reader.py`：ReaderAgent 类（LLM 调用 + 结果解析 + 重试）
- `prompts/reader.md`：system prompt

---

## 依赖

- `anthropic` Python SDK（直接 API 调用）
- 不依赖 claude-agent-sdk（Reader 是单次调用）

---

## 验证成功标准

1. 给定一个真实的 Python 代码仓库（用本项目 spiral-teacher 自身），能输出合法的 `Knowledge` 实例
2. `Knowledge` 通过 models.py 中的所有硬校验
3. `teaching_order` 包含所有 `concepts` 的 ID
4. 至少 80% 的概念有非空 `source_files`，且路径（规范化后）对应仓库中实际存在的文件
5. 对于包含 3 个以上概念的结果，`dependencies` 非空
6. `scan_repository()` 对大仓库不会 OOM（截断逻辑生效）
7. LLM 调用失败或返回非法 JSON 时，抛出 `ReaderError` 并包含有用的错误信息

## 不做的事

- 不支持论文/文档输入（Phase 3）
- 不做缓存（同一仓库重复扫描暂时接受）
- 不做增量更新（Phase 3）
- scan_repository 不做复杂的语言感知解析（AST 等），纯文本读取即可
