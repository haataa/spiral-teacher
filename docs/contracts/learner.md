# 合约：Learner Agent

## 目标

模拟目标读者的学习过程，阅读 Teacher 的讲解并产生结构化反馈（`Feedback`）。Learner 是系统中**最关键的 Agent**——驱动教学质量的不是讲解者的知识量，而是学习者的提问质量。

Learner 在架构上等同于 Harness 文章中的 Evaluator：它是独立于 Generator (Teacher) 的评估者，负责诚实地暴露讲解中的问题，而非"手下留情"。

---

## 接口

```python
class LearnerAgent:
    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: anthropic.AsyncAnthropic | None = None,
    ):
        ...

    async def react(
        self,
        teacher_response: str,
        concept_id: str,
        teaching_level: int,
        audience: AudienceProfile,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
    ) -> tuple[Feedback, str]:
        """阅读 Teacher 的讲解，产生学习者反馈。

        Args:
            teacher_response: Teacher 的原始讲解文本 (raw_text)
            concept_id: 当前讨论的概念 ID
            teaching_level: Teacher 当前讲解的层级 (0-5)
            audience: 目标读者画像
            knowledge: 知识图谱
            conversation: 对话历史

        Returns:
            (结构化 Feedback, 原始文本 raw_text)
        """
```

### 返回值说明

返回 `tuple[Feedback, str]`：
- `Feedback`：结构化数据，Orchestrator 用 `type` 和 `confidence` 决定流程
- `str`（raw_text）：LLM 生成的完整文本，用于 Synthesizer 和调试

---

## 实现策略

### LLM 调用方式

与 Teacher 相同：独立 API 调用，对话历史以参数传入。使用 Anthropic SDK 直接调用。

### 模型选择

默认 `claude-sonnet-4-6`。Sonnet 足够模拟理解过程，响应快，降低多轮对话成本。如果在"手下留情"测试中通过率不达标，升级到 Opus 作为 fallback。

### 消息构造

每次调用的 user message 包含：

1. **读者画像**：math_level, coding_level, domain_knowledge, confusion_triggers
2. **当前概念信息**：description, difficulty, prerequisites, common_misconceptions
3. **Teacher 的讲解**：完整 raw_text + 当前 teaching_level
4. **对话历史**：最近 10 条 raw_text
5. **覆盖情况**：知识图谱中已覆盖 / 未覆盖的概念列表

### 结果解析

与 Teacher 相同的 `---JSON---` 分隔符格式。

解析失败时退化：
- `type` 默认 `"confused"`（宁可多问不可少问）
- `confidence` 默认 `0.3`（保守估计）
- `understanding_summary` 填 LLM 原始输出
- `concept_id` 由调用方提供

### 防止 Learner "手下留情"——双层防线

**第一层：Prompt 层（LLM 侧）**

1. **不允许无条件 understood**：必须能用自己的话复述概念，且复述写入 `understanding_summary`
2. **强制使用 confusion_triggers**：对照画像中的 confusion_triggers 列表逐条检查
3. **主动构建错误假设**：对 difficulty >= 3 的概念，至少尝试构建一个 wrong_assumption
4. **confidence 校准锚点**：
   - 0.0-0.3: 完全不理解
   - 0.3-0.5: 有模糊的感觉
   - 0.5-0.7: 能复述但不确定细节
   - 0.7-0.9: 理解了，有具体疑问
   - 0.9-1.0: 能完整解释

**第二层：代码层无状态硬校验（`validate_feedback()`）**

Prompt 不能完全约束 LLM。以下校验由 `validate_feedback()` 函数实现（纯函数，独立可测试）：

5. **confidence-type 一致性**：
   - `understood` + `confidence < 0.85` → 降级为 `go_deeper`
   - `confused` + `confidence > 0.6` → 修正 confidence 为 0.5
6. **understanding_summary 质量**：
   - `understood` + `len(understanding_summary) < 50` → 降级为 `go_deeper`
7. **连续 understood 熔断**：
   - Orchestrator 跟踪连续 understood 计数，超过 3 次时对下一个概念强制注入额外 challenge prompt

**第三层：代码层有状态概念级校验（`validate_concept_feedback()`）**

需要上下文信息（概念难度、轮次、历史反馈类型）的校验规则，在 `validate_feedback()` 之后调用：

8. **高难度首轮 understood 拦截**：
   - `understood` + `difficulty >= 3` + `rounds_on_current <= 1` → 降级为 `go_deeper`（confidence 上限 0.75）
9. **高难度概念 request_example 保障**：
   - `understood` + `difficulty >= 3` + 历史中无 `request_example` → 降级为 `request_example`（confidence 上限 0.80）

两个函数均定义在 `agents/learner.py` 中，由 Orchestrator 和 CLI resume 路径调用。

---

## System Prompt 设计要点

1. **角色定位**：你是一个 {audience.display_name} 水平的学习者（动态注入画像）
2. **理解检查协议**（7 步）：
   - Step 1 前置知识检查：讲解中用到的概念，在 {math_level} 和 {domain_knowledge} 水平下能理解吗？
   - Step 2 逻辑跳跃检查：讲解中有没有跳过的推理步骤？
   - Step 3 复述检查：能用自己的话说吗？卡在哪一步？
   - Step 4 Confusion Trigger 检查：对照画像中的 triggers 逐条检查，触发时应用 `confused`
   - Step 5 误解构建：对难概念主动构建 wrong_assumption
   - Step 6 首次接触规则：difficulty >= 3 的概念首次接触不允许直接 understood
   - Step 7 例子要求：difficulty >= 3 的概念在未看过具体例子前必须 request_example
3. **对抗性要求**：
   - 你的任务不是假装不懂，也不是轻易说懂——而是**真实模拟**理解过程
4. **深度控制**：
   - 类比层面还没懂时不要急着要数学
   - 直觉已建立时主动要求看公式和代码
   - 当前 teaching_level 信息会告知你 Teacher 在什么深度讲，据此判断是否需要更深
5. **输出格式**：JSON + `---JSON---` + Markdown

Prompt 写在 `src/spiral_teacher/prompts/learner.md` 中。画像信息动态注入。

---

## 文件结构

```
src/spiral_teacher/
├── agents/
│   └── learner.py         # LearnerAgent 类 + validate_feedback() + validate_concept_feedback()
└── prompts/
    └── learner.md         # Learner system prompt（含画像占位符，7 步理解检查协议）
```

---

## 依赖

- `anthropic` Python SDK

---

## 验证成功标准

1. `react` 返回合法的 `Feedback` 实例，所有字段符合 models.py 校验
2. `understanding_summary` 非空
3. `validate_feedback` 单元测试：覆盖所有降级/修正场景
   - understood + confidence 0.5 → 降级为 go_deeper
   - understood + understanding_summary 10 字符 → 降级为 go_deeper
   - confused + confidence 0.8 → 修正为 0.5
4. `validate_concept_feedback` 单元测试：覆盖概念级校验场景
   - 高难度首轮 understood → go_deeper
   - 高难度无 request_example 历史 understood → request_example
   - 低难度 / 有 example 历史时 → 通过
4. 解析失败时退化为 `type="confused"` + `confidence=0.3`，不抛异常
5. conversation 为空列表时正常工作
6. 集成测试（标记 integration）：不同 audience 画像对同一讲解产生不同反馈

## 不做的事

- 不决定何时推进概念（由 Orchestrator 负责）
- 不直接与 Teacher 通信（通过 Orchestrator 中转）
- 不记忆跨次调用的状态（每次调用独立，状态由 conversation 参数传入）
