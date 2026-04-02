# 合约：models.py 数据模型

## 目标

定义 Spiral Teacher 各 Agent 之间通信的结构化数据类型。所有 Agent 的输入输出都通过这些模型约束，确保可序列化（JSON）、可验证、可测试。

---

## 模型清单

### 1. Concept（单个概念）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| id | str | Y | 概念唯一标识，snake_case，如 `wht_rotation` |
| name | str | Y | 人类可读名称，如 "Walsh-Hadamard 旋转" |
| category | str | Y | 概念分类，如 "core_algorithm", "math_foundation" 等，不做枚举限制 |
| description | str | Y | 一句话描述这个概念做什么 |
| prerequisites | list[str] | N | 前置概念 ID 列表，默认空 |
| difficulty | int | Y | 1-5，1 最简单 |
| source_files | list[str] | N | 相关源文件路径 |
| key_equations | list[str] | N | 关键公式（LaTeX 格式） |
| related_concepts | list[str] | N | 相关概念 ID |
| common_misconceptions | list[str] | N | 容易被误解的点 |

### 2. ConceptDependency（概念依赖关系）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| source | str | Y | 前置概念 ID（被依赖方） |
| target | str | Y | 依赖概念 ID（依赖方，target 需要先理解 source） |
| reason | str | Y | 为什么存在这个依赖 |

**示例：** `source="linear_algebra", target="wht_rotation"` 表示 "理解 wht_rotation 需要先理解 linear_algebra"。

> 注：PLAN.md 中示例使用 `from`/`to` 命名，实现中统一为 `source`/`target`（避免 Python 关键字 `from`），需同步更新 PLAN.md。

### 3. Knowledge（Reader 输出，知识图谱）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| project_summary | str | Y | 一句话描述项目做什么 |
| concepts | list[Concept] | Y | 所有提取的概念 |
| dependencies | list[ConceptDependency] | Y | 概念间依赖关系 |
| teaching_order | list[str] | Y | 推荐教学顺序（概念 ID 列表） |
| source_type | Literal["repository", "paper", "document"] | Y | 输入源类型 |
| source_path | str | Y | 输入源路径或 URL |

**验证规则：**
- `teaching_order` 中的每个 ID 必须在 `concepts` 中存在（硬校验）
- `dependencies` 中的 source/target 必须在 `concepts` 中存在（硬校验）
- `teaching_order` 应尽量不违反依赖关系，但**不做硬校验**（LLM 生成的依赖图可能不完美，违反时记录 warning 由 orchestrator 决定是否调整）

### 4. FeedbackDetail（反馈详情）

根据 `Feedback.type` 填写对应字段，其他字段留空。但**不做条件强制校验**——LLM 可能把信息放在"错误"的字段里，orchestrator 层面做宽容处理。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| stuck_point | str | N | 具体卡在哪一步（confused 时填） |
| assumption | str | N | "我的理解是…" （wrong_assumption 时填） |
| request | str | N | 具体请求（request_example 时填） |
| deeper_question | str | N | 想深入的具体问题（go_deeper 时填） |

### 5. Feedback（Learner 输出）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| type | Literal["confused", "go_deeper", "wrong_assumption", "understood", "request_example"] | Y | 反馈类型 |
| concept_id | str | Y | 当前讨论的概念 ID |
| detail | FeedbackDetail | Y | 结构化反馈详情 |
| confidence | float | Y | 0.0-1.0，当前理解信心 |
| understanding_summary | str | Y | 用自己的话描述当前理解 |

**验证规则：**
- confidence 范围 [0.0, 1.0]（硬校验）
- type 与 detail 字段的对应关系**不做硬校验**，仅作为使用约定记录于此

> 注：PLAN.md 中示例使用 `topic` 字段名，实现中统一为 `concept_id`（更精确），需同步更新 PLAN.md。

**使用约定（非强制）：**

| type | 建议填写的 detail 字段 |
|------|----------------------|
| confused | stuck_point |
| go_deeper | deeper_question |
| wrong_assumption | assumption |
| request_example | request |
| understood | （无需填 detail） |

### 6. TeachingResponse（Teacher 输出）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| concept_id | str | Y | 当前讲解的概念 ID |
| level | int | Y | 讲解层级 0-5（ge=0, le=5） |
| content | str | Y | Markdown 格式讲解内容 |
| analogies_used | list[str] | N | 本次使用的类比 |
| code_references | list[CodeReference] | N | 引用的源码 |
| next_concept_id | str | N | 如果推进到下一个概念，填写 |

**Level 含义参考：**

| Level | 含义 |
|-------|------|
| 0 | 一句话概括 |
| 1 | 鸟瞰（整体流程） |
| 2 | 直觉（类比和具体数字） |
| 3 | 机制（算法步骤、伪代码） |
| 4 | 严格（数学推导、源码逐行） |
| 5 | 边界（trade-off、与相关工作对比） |

### 7. CodeReference（源码引用）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| file_path | str | Y | 文件路径 |
| start_line | int | N | 起始行号 |
| end_line | int | N | 结束行号 |
| snippet | str | N | 代码片段 |
| explanation | str | N | 这段代码做什么 |

### 8. ConversationEntry（对话记录）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| role | Literal["teacher", "learner"] | Y | 发言角色，同时作为 content 的反序列化鉴别器 |
| teaching_response | TeachingResponse | N | role="teacher" 时填写 |
| feedback | Feedback | N | role="learner" 时填写 |
| raw_text | str | Y | 原始文本（content 的渲染文本表示，用于传给 Synthesizer） |

**验证规则：**
- `teaching_response` 和 `feedback` 恰好有一个非空（model_validator）
- `role="teacher"` 时 `teaching_response` 不应为空
- `role="learner"` 时 `feedback` 不应为空

> 改动说明：原设计使用联合类型 `content: TeachingResponse | Feedback`，反序列化时缺乏鉴别器。改为两个 Optional 字段 + model_validator，工程上更简洁。移除了 `timestamp` 字段（Phase 1 对话顺序由列表索引决定，无需时间戳）。

### 9. AudienceProfile（读者画像）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| name | str | Y | 画像名称，如 "ml_engineer" |
| display_name | str | Y | 显示名称，如 "ML 工程师" |
| math_level | str | Y | 数学水平描述 |
| coding_level | str | Y | 编程水平描述 |
| domain_knowledge | str | Y | 领域知识描述 |
| confusion_triggers | list[str] | Y | 容易触发困惑的模式 |

**来源：** YAML 配置文件，非 LLM 生成。

### 10. TutorialConfig（运行配置）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| source | str | Y | 代码仓库路径 / 论文 URL |
| source_type | Literal["repository", "paper", "document"] | Y | 输入源类型 |
| topic | str | N | 聚焦主题 |
| audience | str | Y | 读者画像 key（有效性由 orchestrator 在运行时校验） |
| output_path | str | Y | 输出文档路径 |
| max_rounds | int | N | 最大对话轮次，默认 30 |
| max_rounds_per_concept | int | N | 单概念最大轮次，默认 6 |

### 11. SynthesisInput（Synthesizer 输入）

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| conversation | list[ConversationEntry] | Y | 完整对话历史 |
| knowledge | Knowledge | Y | Reader 输出的知识图谱 |
| audience | AudienceProfile | Y | 目标读者画像 |

> 说明：Synthesizer 的输出是纯 Markdown 字符串，不单独建模。

---

## Orchestrator 内部状态说明

以下变量是 Orchestrator 运行时内部状态，不跨 Agent 传递，因此不建模为独立合约：

- `covered_concepts: set[str]` -- 已覆盖的概念 ID
- `current_concept_idx: int` -- 当前概念索引
- `rounds_on_current: int` -- 当前概念已对话轮次

---

## 验证成功标准

1. 所有模型可序列化为 JSON 并反序列化回来（round-trip）
2. 硬校验规则通过 Pydantic validator 实现，非法输入抛出 ValidationError
3. 可从 PLAN.md 中的示例 JSON 构造出对应模型实例（需先同步 PLAN.md 中的字段名：`from`/`to` → `source`/`target`，`topic` → `concept_id`）
4. AudienceProfile 可从 YAML 文件加载
5. ConversationEntry 的 round-trip 序列化正确（根据 role 字段反序列化为正确的内容类型）

## 不做的事

- 不实现任何 Agent 逻辑，纯数据定义
- 不实现文件 I/O 工具函数（放在 utils.py）
- 不过度设计：如果某字段在 Phase 1 用不到，标 Optional 即可，不做提前抽象
- 不对 LLM 输出做过严的条件校验（宽进严出，在 orchestrator 层面做容错处理）
