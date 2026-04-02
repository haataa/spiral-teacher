# 合约：Teacher Agent

## 目标

根据 Learner 的反馈和当前理解状态，生成匹配层级的讲解内容（`TeachingResponse`）。Teacher 是对话的"生成器"端，负责多层级表达、源码引用和类比解释。

---

## 接口

```python
class TeacherAgent:
    def __init__(
        self,
        model: str = "claude-opus-4-6",
        client: anthropic.AsyncAnthropic | None = None,
    ):
        ...

    async def give_overview(
        self,
        knowledge: Knowledge,
    ) -> tuple[TeachingResponse, str]:
        """生成项目整体鸟瞰讲解（Level 1）。

        这是对话的第一条 teacher 消息。

        Returns:
            (结构化 TeachingResponse, 原始文本 raw_text)
        """

    async def introduce_concept(
        self,
        concept_id: str,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
        level: int = 2,
    ) -> tuple[TeachingResponse, str]:
        """引入一个新概念。

        当 Learner 对前一个概念 understood 后，Orchestrator 调用此方法
        推进到下一个概念。level 可传 1 或 2，默认 2。

        Raises:
            ValueError: concept_id 不在 knowledge.concepts 中

        Returns:
            (结构化 TeachingResponse, 原始文本 raw_text)
        """

    async def respond_to_feedback(
        self,
        feedback: Feedback,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
        current_level: int,
    ) -> tuple[TeachingResponse, str]:
        """响应 Learner 的反馈。

        根据反馈类型和层级切换规则，生成对应层级的讲解。
        current_level 应从 conversation 中最近一条 teacher 条目的
        teaching_response.level 获取。

        如果收到 type='understood' 的反馈，发出 warning 并等效于
        introduce_concept 进入下一个概念。

        Returns:
            (结构化 TeachingResponse, 原始文本 raw_text)
        """
```

### 返回值说明

每个方法返回 `tuple[TeachingResponse, str]`：
- `TeachingResponse`：结构化数据，用于 Orchestrator 逻辑判断（level、concept_id 等）
- `str`（raw_text）：LLM 生成的原始 Markdown 讲解文本，用于传给 Learner 阅读和 Synthesizer 合成

Orchestrator 构造 ConversationEntry 的方式：
```python
tr, raw = await teacher.respond_to_feedback(...)
entry = ConversationEntry(role="teacher", teaching_response=tr, raw_text=raw)
```

---

## 实现策略

### LLM 调用方式

Teacher 的每次调用都是独立的 API 调用（非多轮对话），因为：
- 对话历史由 Orchestrator 管理，以 `conversation` 参数传入
- 每次调用的 system prompt 固定，user message 包含当前上下文

使用 Anthropic SDK 直接调用，不需要 Agent SDK。

不做重试：与 Reader 不同，Teacher 的关键字段（concept_id, level）可由调用方推断，JSON 解析失败时直接退化即可，不需要额外 API 调用。

### 消息构造

每次调用的 user message 包含：

1. **知识图谱摘要**：当前概念的信息（description、prerequisites、source_files、key_equations、common_misconceptions）
2. **对话历史**：最近 10 条 `raw_text`（避免上下文过长）。conversation 为空列表时正常工作。
3. **当前任务**：根据调用方法不同：
   - `give_overview`：生成项目鸟瞰
   - `introduce_concept`：引入指定概念
   - `respond_to_feedback`：响应具体反馈
4. **源码片段**：如果当前概念有 `source_files`，读取相关文件内容附加到消息中

### 源码读取

Teacher 通过 `knowledge.source_path` 获取仓库根路径，以解析 `source_files` 中的相对路径。

- `introduce_concept` 和 `respond_to_feedback` 时，如果当前概念有 `source_files`，读取这些文件内容
- 使用 `utils.py` 中的 `_truncate_file_content()` 做单文件截断（与 Reader 共用）
- **多文件总预算**：所有 source_files 的总内容不超过 1000 行，超出时按文件在列表中的顺序截断
- **source_files 中的文件不存在时**，跳过并记录 warning，不抛异常

### 层级切换规则

在 `respond_to_feedback` 中根据 PLAN.md 的规则决定输出 level：

| Learner 反馈 | Teacher 动作 |
|-------------|-------------|
| `confused` + current_level >= 3 | 降到 Level 2，用类比重新解释 |
| `confused` + current_level <= 2 | 保持当前 Level，换一个类比或给具体数字 |
| `go_deeper` | 升一级（min(current + 1, 5)） |
| `wrong_assumption` | 保持当前 Level，先纠正再继续 |
| `understood` | 发 warning，等效于 introduce_concept 进入下一概念 |
| `request_example` | 保持当前 Level，插入数值例子 |

层级切换逻辑在**代码中实现**（不依赖 LLM 判断），计算出目标 level 后传给 LLM。
此函数 `compute_next_level(feedback_type, current_level) -> int` 应独立可测试。

### 结果解析

Prompt 要求 LLM 先输出 JSON 元数据块（用 `---JSON---` 分隔符），再输出 Markdown 正文。

解析流程：
```python
def parse_response(raw: str, concept_id: str, level: int) -> tuple[TeachingResponse, str]:
    try:
        json_part, markdown_part = split_on_separator(raw, "---JSON---")
        data = json.loads(json_part)
        data["content"] = markdown_part.strip()
        return (TeachingResponse.model_validate(data), raw)
    except (json.JSONDecodeError, ValidationError, ValueError):
        # 退化：用已知信息填充
        return (
            TeachingResponse(
                concept_id=concept_id,
                level=level,
                content=raw,
            ),
            raw,
        )
```

退化时 `analogies_used` 和 `code_references` 默认空列表，`next_concept_id` 默认 None。

---

## System Prompt 设计要点

1. **角色定位**：你是一个经验丰富的技术导师，擅长多层级讲解
2. **讲解约束**：
   - 每次讲解聚焦一个概念，不铺开多个
   - 引用源码时给出文件名和行号
   - 使用类比时标注"这是类比，严格来说..."
   - 给出数学公式时同时给出直觉解释
3. **Level 指引**：根据传入的目标 level，调整讲解深度
4. **避免重复**：参考对话历史，不重复已经讲过的内容，换角度解释
5. **输出格式**：先输出 JSON 元数据块（`---JSON---` 分隔符），再输出 Markdown 讲解正文

Prompt 写在 `src/spiral_teacher/prompts/teacher.md` 中。

---

## 文件结构

```
src/spiral_teacher/
├── agents/
│   └── teacher.py         # TeacherAgent 类
└── prompts/
    └── teacher.md         # Teacher system prompt
```

---

## 依赖

- `anthropic` Python SDK
- 源码访问通过 `knowledge.source_path`

---

## 验证成功标准

1. `give_overview` 返回 level=1 的 TeachingResponse，content 非空
2. `introduce_concept` 返回指定 concept_id 和 level 的 TeachingResponse
3. `introduce_concept` 对不存在的 concept_id 抛出 ValueError
4. `respond_to_feedback` 正确执行层级切换规则（confused 时降级，go_deeper 时升级）
5. `compute_next_level` 单元测试：覆盖所有 feedback type × level 组合
6. JSON 解析失败时优雅退化，不抛异常，返回以 raw 文本为 content 的 TeachingResponse
7. raw_text 包含 Markdown 格式的讲解内容
8. source_files 中的文件不存在时，不抛异常，正常生成讲解
9. conversation 为空列表时正常工作

## 不做的事

- 不管理对话状态（由 Orchestrator 负责）
- 不决定何时推进到下一个概念（由 Orchestrator 根据 Learner 反馈决定）
- 不做自我评估（这是 Harness 文章的关键教训——评估交给 Learner）
