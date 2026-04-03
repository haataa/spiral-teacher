# Spiral Teacher

> 基于多 Agent 协作的自适应教学内容生成系统。
> 模拟"螺旋式深入"的人类学习互动过程，自动为代码仓库/论文/技术文档生成多层级教学文档。

---

## 灵感来源

在一次人工互动中观察到高效学习的四个特征：

1. **螺旋式深入**：先鸟瞰再聚焦，先直觉再严格
2. **精确定位卡点**：不说"都看不懂"，而是指出具体哪一步断了
3. **主动切换抽象层级**：在比喻和数学之间按需跳转
4. **固化为连贯叙事**：对话结束后重组织成结构化文档

核心洞察：**驱动教学质量的不是讲解者的知识量，而是学习者的提问质量。** 因此系统的核心挑战是模拟一个高质量的学习者 Agent。

---

## 系统架构

```
                        ┌───────────────┐
                        │  Orchestrator  │
                        │  (流程控制)     │
                        └───────┬───────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
      ┌──────────────┐  ┌─────────────┐  ┌──────────────┐
      │    Reader     │  │   Learner    │  │  Synthesizer  │
      │ (知识提取)    │  │ (模拟学习者) │  │  (文档合成)   │
      └──────┬───────┘  └──────┬──────┘  └──────────────┘
             │                 │                 ▲
             │  结构化知识       │  问题/反馈        │ 对话历史
             ▼                 ▼                 │
      ┌────────────────────────────────┐         │
      │           Teacher              │─────────┘
      │       (多层级讲解生成)          │
      └────────────────────────────────┘
```

### Agent 1: Reader（知识提取）

**职责：** 阅读代码仓库/论文/文档，提取结构化知识。

**输入：**
- 代码仓库路径 或 论文 URL 或 文档路径
- 可选：用户指定的关注主题

**输出：** 结构化知识图谱（JSON）

```json
{
  "project_summary": "一句话描述项目做什么",
  "concepts": [
    {
      "id": "wht_rotation",
      "name": "Walsh-Hadamard 旋转",
      "category": "core_algorithm",
      "description": "通过正交变换使 KV 向量各维度分布均匀化",
      "prerequisites": ["linear_algebra_basics", "orthogonal_matrix"],
      "difficulty": 3,
      "source_files": ["turboquant/rotation.py"],
      "key_equations": ["y = D2 @ H @ D1 @ x"],
      "related_concepts": ["polar_quant", "fwht"]
    }
  ],
  "concept_dependencies": [
    {"from": "wht_rotation", "to": "polar_quant", "reason": "PolarQuant 依赖旋转后的高斯分布假设"}
  ],
  "teaching_order": ["kv_cache_problem", "distribution_problem", "wht_rotation", "polar_quant", "kv_asymmetry", "sparse_v"]
}
```

**关键能力：**
- 识别概念间的前置依赖关系
- 推荐教学顺序（拓扑排序 + 难度排序）
- 标注每个概念的难度等级和最容易被误解的点
- 定位源码中的关键文件和行号

**模型选择：** Opus（需要深度代码理解和架构分析能力）

---

### Agent 2: Learner（模拟学习者）——最核心的 Agent

**职责：** 阅读 Teacher 的讲解，模拟真实学习者的理解过程，产生高质量反馈。

**输入：**
- Teacher 的当前讲解
- 目标读者画像配置
- 已建立的理解（对话历史摘要）
- Reader 提供的知识图谱（用于判断是否还有未覆盖的概念）

**输出：** 结构化反馈

```json
{
  "type": "confused | go_deeper | wrong_assumption | understood | request_example",
  "topic": "当前讨论的概念 ID",
  "detail": {
    "stuck_point": "具体卡在哪一步",
    "assumption": "我的理解是..., 这对吗？",
    "request": "能举一个具体数字的例子吗？"
  },
  "confidence": 0.7,
  "understanding_summary": "我目前的理解：旋转是为了让分布变均匀，但我不确定为什么 Hadamard 矩阵能做到这一点"
}
```

**反馈类型详解：**

| type | 含义 | 触发条件 | Teacher 响应 |
|------|------|---------|-------------|
| `confused` | 某个具体概念不理解 | 前置知识缺失或逻辑跳跃 | 降低抽象层级，用类比重新解释 |
| `go_deeper` | 直觉已建立，要严格理解 | confidence > 0.7 但缺乏数学/代码支撑 | 升高抽象层级，给出公式推导和源码 |
| `wrong_assumption` | 构建了可能的错误理解 | 概念容易被误解 | 纠正错误，对比正确理解 |
| `understood` | 当前概念理解完成 | confidence > 0.9 | 推进到下一个概念 |
| `request_example` | 需要具体例子 | 纯抽象描述难以消化 | 给出数值例子或代码演示 |

**目标读者画像配置：**

```yaml
audience_profiles:
  undergraduate:
    math_level: "线性代数基础，概率统计入门"
    coding_level: "Python 基础，会读 NumPy"
    domain_knowledge: "知道什么是神经网络，不了解 Transformer 细节"
    confusion_triggers:
      - "跳过直觉直接给公式"
      - "使用未定义的术语"
      - "没有具体数字的抽象描述"

  ml_engineer:
    math_level: "本科数学完整，熟悉优化和概率"
    coding_level: "Python/C++ 熟练，读过 llama.cpp 源码"
    domain_knowledge: "熟悉 Transformer、KV 缓存、量化基本概念"
    confusion_triggers:
      - "理论为什么在工程上成立的 gap"
      - "多个近似叠加后误差是否可控"
      - "与已知方法（GPTQ, AWQ）的区别"

  researcher:
    math_level: "研究生数学，熟悉信息论和高维几何"
    coding_level: "能读懂论文伪代码并实现"
    domain_knowledge: "熟悉量化理论、JL 引理、Rate-Distortion"
    confusion_triggers:
      - "理论界限的 tightness"
      - "偏离论文的工程决策是否有理论支撑"
      - "与同期工作的对比"
```

**Learner 的 prompt 核心设计：**

```
你是一个 {audience_profile} 级别的学习者。你的任务不是假装不懂，
而是真实地模拟理解过程：

1. 阅读讲解后，检查每个概念：
   - 这个概念的前置知识我有吗？（对照你的 math_level 和 domain_knowledge）
   - 讲解中有没有跳跃的逻辑步骤？
   - 我能用自己的话复述吗？如果不能，卡在哪一步？

2. 主动构建可能的错误理解：
   - 这个概念最容易被误解成什么？
   - 和什么表面相似但本质不同的概念容易混淆？
   - 提出这个错误假设，让 Teacher 纠正

3. 控制深度节奏：
   - 如果类比/直觉层面还没懂，不要急着看数学
   - 如果直觉已经建立，主动要求看公式和代码确认
   - 如果纯抽象描述，要求给出具体数字的例子

4. 诚实评估理解程度：
   - 给出 confidence 分数（0-1）
   - 写出当前理解的摘要（Teacher 可以据此判断是否有遗漏或偏差）
```

**模型选择：** Sonnet（足够模拟理解过程，响应速度快，降低多轮对话成本）

---

### Agent 3: Teacher（多层级讲解生成）

**职责：** 根据 Learner 的反馈类型和当前理解状态，生成匹配层级的讲解。

**输入：**
- Learner 的反馈（类型 + 具体内容）
- Reader 提供的知识图谱和源码引用
- 对话历史（避免重复）
- 当前讲解深度层级

**输出：** 讲解文本（markdown 格式）

**讲解层级体系：**

```
Level 0: 一句话概括（项目做什么）
Level 1: 鸟瞰（整体流程，每步做什么，不深入怎么做）
Level 2: 直觉（每步为什么这样做，用类比和具体数字说明）
Level 3: 机制（怎么做的，算法步骤，伪代码）
Level 4: 严格（数学推导，定理证明，源码逐行解读）
Level 5: 边界（为什么不用其他方法，trade-off 分析，与相关工作对比）
```

**层级切换规则：**

| Learner 反馈 | Teacher 动作 |
|-------------|-------------|
| `confused` + 当前 Level ≥ 3 | 降到 Level 2，用类比重新解释 |
| `confused` + 当前 Level ≤ 2 | 保持当前 Level，换一个类比或给具体数字 |
| `go_deeper` | 升一级 |
| `wrong_assumption` | 保持当前 Level，先纠正再继续 |
| `understood` | 保持 Level 1-2 进入下一个概念 |
| `request_example` | 保持当前 Level，插入数值例子 |

**讲解约束：**
- 每次讲解聚焦一个概念，不铺开多个
- 引用源码时给出文件名和行号
- 使用类比时标注"这是类比，严格来说..."
- 给出数学公式时同时给出直觉解释

**模型选择：** Opus（需要深度推理 + 多层级表达能力 + 源码理解）

---

### Agent 4: Synthesizer（文档合成）

**职责：** 将多轮对话历史重组织为连贯的教学文档。

**输入：**
- 完整对话历史（所有 Learner-Teacher 交互）
- Reader 的知识图谱
- 目标读者画像

**输出：** 结构化 Markdown 教学文档

**合成原则：**
1. 不是简单拼接对话，而是**重新组织成连贯叙事**
2. 保留对话中自然产生的好类比和好例子
3. 填补对话中跳跃的逻辑缝隙（对话中靠上下文隐含的，文档中要显式写出）
4. 保留"螺旋式"结构：每个概念先给直觉，再给严格
5. 保留 wrong_assumption 交互——"你可能会以为...但实际上..."是很好的教学手法

**文档结构模板：**

```markdown
# {项目名} 原理深度教学

## 一、项目做什么？（Level 0-1）
## 二、核心问题是什么？（Level 1-2）
## 三、解决方案概览（Level 1）
## 四、{概念A} 详解（Level 2 → 3 → 4）
  ### 直觉理解
  ### 算法细节
  ### 数学推导
  ### 常见误解
## 五、{概念B} 详解
  ...
## N、完整数据流（Level 1，串联所有概念）
## N+1、总结
```

**模型选择：** Opus（需要跨越多轮对话的重组织能力）

---

## Orchestrator 流程控制

### 主循环

```python
def generate_tutorial(
    source: str,              # 代码仓库路径 / 论文 URL / 文档路径
    topic: str | None,        # 可选：聚焦主题
    audience: str,            # 读者画像 key
    output_path: str,         # 输出文档路径
    max_rounds: int = 30,     # 最大对话轮次
    max_rounds_per_concept: int = 6,  # 单概念最大轮次
    min_importance: int = 3,          # 跳过低重要性概念（1-5）
):
    # ─── Phase 1: 知识提取 ───
    knowledge = reader.extract(source, topic)
    teaching_order = knowledge["teaching_order"]

    # ─── Phase 2: 多轮对话 ───
    conversation = []
    covered_concepts = set()
    current_concept_idx = 0
    rounds_on_current = 0

    # 初始鸟瞰
    overview = teacher.give_overview(knowledge)
    conversation.append({"role": "teacher", "content": overview, "level": 1})

    while current_concept_idx < len(teaching_order) and len(conversation) < max_rounds * 2:
        current_concept = teaching_order[current_concept_idx]

        # Learner 反应
        feedback = learner.react(
            teacher_response=conversation[-1]["content"],
            audience=audience,
            knowledge=knowledge,
            conversation_summary=summarize(conversation),
        )
        conversation.append({"role": "learner", "content": feedback})

        if feedback["type"] == "understood":
            covered_concepts.add(current_concept)
            current_concept_idx += 1
            rounds_on_current = 0

            if current_concept_idx < len(teaching_order):
                next_concept = teaching_order[current_concept_idx]
                response = teacher.introduce(next_concept, knowledge, level=2)
            else:
                break
        else:
            rounds_on_current += 1
            if rounds_on_current >= max_rounds_per_concept:
                # 强制推进，避免死循环
                covered_concepts.add(current_concept)
                current_concept_idx += 1
                rounds_on_current = 0
                response = teacher.introduce(
                    teaching_order[current_concept_idx], knowledge, level=2
                )
            else:
                response = teacher.respond(feedback, knowledge, conversation)

        conversation.append({"role": "teacher", "content": response})

    # ─── Phase 3: 文档合成 ───
    document = synthesizer.compile(
        conversation=conversation,
        knowledge=knowledge,
        audience=audience,
    )

    write_file(output_path, document)
    return document
```

### 终止条件

- 所有概念的 Learner confidence > 0.9（正常完成）
- 达到 max_rounds（超时兜底）
- 单概念超过 max_rounds_per_concept 轮（避免死循环，强制推进）

### 质量控制

- **概念覆盖率**：知识图谱中的概念是否全部覆盖
- **深度覆盖率**：每个概念是否至少到达 Level 3
- **错误假设覆盖率**：Reader 标注的易混淆点是否至少被 Learner 提出过一个
- 如果质量指标不达标，Orchestrator 可以追加额外轮次

---

## 技术选型

| 组件 | 技术 | 理由 |
|------|------|------|
| Agent 框架 | Claude Agent SDK (Python) | 原生支持多 Agent 编排、tool use、结构化输出 |
| Reader 模型 | claude-opus-4-6 | 需要深度代码理解和架构分析 |
| Learner 模型 | claude-sonnet-4-6 | 足够模拟理解过程，响应快，降低多轮成本 |
| Teacher 模型 | claude-opus-4-6 | 需要多层级表达 + 源码引用 + 数学推导 |
| Synthesizer 模型 | claude-opus-4-6 | 需要跨多轮对话的重组织能力 |
| 输出格式 | Markdown | 兼容 Obsidian、GitHub、博客 |

### 成本估算（单次运行）

- Reader: 1 次 Opus 调用，~50K input tokens（读代码），~5K output → ~$0.80
- Learner: ~15 次 Sonnet 调用，~3K input/output each → ~$0.45
- Teacher: ~15 次 Opus 调用，~5K input, ~3K output each → ~$3.00
- Synthesizer: 1 次 Opus 调用，~30K input, ~10K output → ~$1.00
- **总计约 $5-6 per tutorial**

---

## 项目结构

```
spiral-teacher/
├── PLAN.md                      # 本文件
├── README.md                    # 使用说明
├── pyproject.toml               # 项目配置
├── src/
│   └── spiral_teacher/
│       ├── __init__.py
│       ├── orchestrator.py      # 主循环控制
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── reader.py        # 知识提取 Agent
│       │   ├── learner.py       # 模拟学习者 Agent
│       │   ├── teacher.py       # 讲解生成 Agent
│       │   └── synthesizer.py   # 文档合成 Agent
│       ├── prompts/
│       │   ├── reader.md        # Reader system prompt
│       │   ├── learner.md       # Learner system prompt
│       │   ├── teacher.md       # Teacher system prompt
│       │   └── synthesizer.md   # Synthesizer system prompt
│       ├── profiles/
│       │   ├── undergraduate.yaml
│       │   ├── ml_engineer.yaml
│       │   └── researcher.yaml
│       ├── models.py            # Pydantic 数据模型（Knowledge, Feedback, etc.）
│       └── utils.py             # 文件读写、token 计数、摘要生成
├── tests/
│   ├── test_reader.py
│   ├── test_learner.py
│   ├── test_teacher.py
│   ├── test_synthesizer.py
│   └── test_orchestrator.py
├── examples/
│   └── turboquant/              # TurboQuant+ 作为第一个测试案例
│       ├── config.yaml          # 运行配置
│       └── output.md            # 生成的教学文档
└── docs/
    └── design-decisions.md      # 设计决策记录
```

---

## 开发计划

### Phase 1: 核心骨架 ✅

- [x] 定义数据模型（Knowledge, Feedback, TeachingResponse 等 Pydantic models）
- [x] 实现 Reader Agent（代码仓库模式，论文模式后续扩展）
- [x] 实现 Learner Agent（先支持 ml_engineer 画像）
- [x] 实现 Teacher Agent（支持 Level 0-5 全部层级，层级切换为确定性代码逻辑）
- [x] 实现 Orchestrator 主循环
- [x] 实现 Synthesizer Agent
- [x] CLI 工具（`spiral-teacher generate/synthesize`）+ 断点续跑（`--resume`）
- [x] 全链路中文默认输出
- [x] 127 个单元测试
- [x] E2E 验证（agent-world-model、Eureka、RF-Agent 仓库）

**Phase 1 实现过程中的关键设计演进：**
- 概念排序从"难度优先"改为"重要性优先"（Concept 新增 importance 字段）
- Learner 三层防线：prompt 层（7 步理解检查协议）+ validate_feedback() 无状态校验 + validate_concept_feedback() 概念级有状态校验
- JSON 解析从严格 separator 格式改为多策略容错提取（utils.extract_json_from_text）
- 渐进式输出：知识图谱/对话/教程在各阶段即时写文件

### Phase 2: 验证与调优

- [x] 用 agent-world-model 作为测试案例，完整跑多轮（已验证 resume 功能）
- [x] 调优 Learner prompt（读 conversation.md trace，找判断偏差）
  - 强化 confusion_triggers checklist（Step 4：从"不许 understood"升级为"应当 confused"）
  - 新增 Step 6（First Encounter Rule）和 Step 7（Example Requirement）
  - 新增 `validate_concept_feedback()` 概念级有状态校验：Rule 4（高难度首轮 understood → go_deeper）、Rule 5（无 request_example 就 understood → request_example）
  - E2E 验证（Eureka 仓库）：request_example 0→4, go_deeper 2→10, 每概念深度 2→5 轮
- [x] 用 Eureka 作为第二个测试案例验证调优效果
- [x] 用 RF-Agent 作为第三个测试案例验证覆盖率优化（每概念 5.0→3.3 轮）
- [ ] 对比人工互动产出的文档 vs 系统产出的文档，评估质量差距
- [ ] 调优讲解层级切换策略
- [ ] 处理边界情况（概念循环依赖、Learner 持续 confused、空仓库等）
- [ ] 解决 confused=0 问题（Sonnet 不愿承认困惑，倾向用 go_deeper 替代）
- [ ] 优化覆盖率与深度的平衡（当前每概念 ~5 轮，20 轮仅覆盖 4/23 概念）

### Phase 3: 扩展

- [ ] 支持论文 URL 输入（通过 alphaXiv MCP 获取内容）
- [ ] 支持多读者画像（undergraduate, researcher）
- [x] 支持中英文输出（language 参数，默认中文）
- [ ] 支持增量更新（代码更新后只重新生成受影响的章节）

---

## 关键风险与应对

| 风险 | 影响 | 应对 |
|------|------|------|
| Learner 太"聪明"，不提问 | 文档太浅 | 在 prompt 中显式要求基于画像的 confusion_triggers 提问 |
| Learner 太"笨"，无限循环 | 超时 / 成本爆炸 | max_rounds_per_concept 兜底 + 强制推进 |
| Teacher 重复相同的解释 | 死循环 | 在 prompt 中注入对话历史摘要，要求换角度 |
| Reader 遗漏关键概念 | 文档不完整 | 概念覆盖率检查 + Learner 可以发现并提出新概念 |
| 合成文档逻辑不连贯 | 可读性差 | Synthesizer 二次审阅 + 结构模板约束 |
| 单次成本 $5-6 | 调试期间成本累积 | 开发阶段用 Haiku 替代 Opus 做快速迭代 |

---

## 设计哲学

1. **学习者驱动**：Learner Agent 的质量决定一切。好的问题比好的答案更稀缺。
2. **螺旋式而非线性**：同一个概念会被多次触及，每次深入一层，而不是一次讲完。
3. **错误是教学工具**：wrong_assumption 交互产生的"你可能以为...但实际上..."是最有教学价值的内容。
4. **可验证**：每个 Agent 的输出都是结构化的（JSON/YAML），可以用程序检查质量指标。
5. **面向真实项目**：不是讲教科书概念，而是解读具体的代码仓库，引用真实的文件和行号。
