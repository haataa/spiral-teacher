# 合约：Orchestrator

## 目标

编排 Reader → Teacher ↔ Learner 的完整对话流程，管理对话状态，执行质量控制，最终输出 `SynthesisInput`（供 Synthesizer 使用）。

---

## 接口

```python
async def generate_tutorial(config: TutorialConfig) -> SynthesisInput:
    """运行完整的教学对话生成流程。

    流程：
    1. 加载读者画像
    2. Reader 扫描仓库，提取知识图谱
    3. Teacher 给出鸟瞰概述
    4. 按 teaching_order 逐概念进行 Teacher-Learner 对话
    5. 返回 SynthesisInput

    Args:
        config: 运行配置（仓库路径、读者画像、输出路径等）

    Returns:
        SynthesisInput（conversation + knowledge + audience）

    Raises:
        FileNotFoundError: 仓库或画像文件不存在
        ReaderError: Reader 提取失败
    """
```

另外提供一个回调参数用于实时输出进度：

```python
OnProgressCallback = Callable[[str, ConversationEntry | None], None]

async def generate_tutorial(
    config: TutorialConfig,
    on_progress: OnProgressCallback | None = None,
) -> SynthesisInput:
```

`on_progress` 在每轮对话后调用，参数为 (状态描述, 最新对话条目)。

---

## 主循环逻辑

```
Phase 1: 初始化
  ├── 加载 AudienceProfile (YAML)
  ├── ReaderAgent.read_repository() → Knowledge
  └── TeacherAgent.give_overview() → 第一条对话

Phase 2: 逐概念对话
  for concept in knowledge.teaching_order:
    ├── TeacherAgent.introduce_concept(concept)
    └── loop (max_rounds_per_concept):
          ├── LearnerAgent.react() → Feedback
          ├── validate_feedback() → 可能修正
          ├── if understood → break, 进入下一概念
          └── TeacherAgent.respond_to_feedback()

Phase 3: 组装输出
  └── 返回 SynthesisInput(conversation, knowledge, audience)
```

### 状态变量

- `conversation: list[ConversationEntry]` -- 完整对话历史
- `covered_concepts: set[str]` -- 已理解的概念
- `current_concept_idx: int` -- 当前概念在 teaching_order 中的索引
- `rounds_on_current: int` -- 当前概念已对话轮次
- `consecutive_understood: int` -- 连续 understood 计数（熔断用）

### 终止条件

1. **正常完成**：所有概念的 Learner feedback.type == understood
2. **全局超时**：总对话轮次达到 `config.max_rounds`
3. **单概念超时**：当前概念对话轮次达到 `config.max_rounds_per_concept`，强制推进

### 连续 understood 熔断

如果 `consecutive_understood >= 3`，对下一个概念的 Teacher 讲解追加 challenge prompt：
"The learner has been understanding concepts quickly. For this concept, dig deeper into potential misconceptions and edge cases."

`consecutive_understood` 在任何非 understood 反馈后重置为 0。

### 画像加载

从 `src/spiral_teacher/profiles/{config.audience}.yaml` 加载。如果文件不存在，抛出 `FileNotFoundError`。

---

## 文件结构

```
src/spiral_teacher/
├── orchestrator.py        # generate_tutorial() + 辅助函数
```

Orchestrator 是一个模块级函数，不是类（没有需要跨调用保持的状态）。

---

## 依赖

- `agents/reader.py` -- ReaderAgent
- `agents/teacher.py` -- TeacherAgent
- `agents/learner.py` -- LearnerAgent + validate_feedback
- `models.py` -- 所有数据模型
- `profiles/*.yaml` -- 读者画像

---

## 验证成功标准

1. 给定 mock 的 Reader/Teacher/Learner，主循环正确执行：概念按 teaching_order 推进
2. validate_feedback 在每轮 Learner 输出后被调用
3. 单概念超时（max_rounds_per_concept）触发强制推进
4. 全局超时（max_rounds）触发终止
5. 连续 understood 熔断在 3 次后触发 challenge prompt
6. 画像加载：存在的画像正常加载，不存在的抛异常
7. 返回的 SynthesisInput 包含完整 conversation + knowledge + audience

## 不做的事

- 不实现 Synthesizer 逻辑（Step 8）
- 不写入输出文件（由调用方决定是否调用 Synthesizer）
- 不做并发（Teacher 和 Learner 串行调用）
