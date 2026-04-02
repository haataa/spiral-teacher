"""Spiral Teacher 数据模型。

定义各 Agent 之间通信的结构化数据类型。
合约文档：docs/contracts/models.md
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


# ── 基础模型 ──


class CodeReference(BaseModel):
    """源码引用。"""

    file_path: str
    start_line: int | None = None
    end_line: int | None = None
    snippet: str | None = None
    explanation: str | None = None


class Concept(BaseModel):
    """知识图谱中的单个概念。"""

    id: str = Field(description="snake_case 唯一标识")
    name: str = Field(description="人类可读名称")
    category: str = Field(description="概念分类，如 core_algorithm, math_foundation")
    description: str = Field(description="一句话描述")
    prerequisites: list[str] = Field(default_factory=list)
    difficulty: int = Field(ge=1, le=5)
    importance: int = Field(ge=1, le=5, default=3, description="1=peripheral, 5=core to understanding the project")
    source_files: list[str] = Field(default_factory=list)
    key_equations: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    common_misconceptions: list[str] = Field(default_factory=list)


class ConceptDependency(BaseModel):
    """概念间依赖关系。

    source 是前置概念（被依赖方），target 是依赖概念（依赖方）。
    例：source="linear_algebra", target="wht_rotation"
    表示 "理解 wht_rotation 需要先理解 linear_algebra"。
    """

    source: str
    target: str
    reason: str


# ── Reader 输出 ──


class Knowledge(BaseModel):
    """Reader Agent 输出的结构化知识图谱。"""

    project_summary: str
    concepts: list[Concept]
    dependencies: list[ConceptDependency] = Field(default_factory=list)
    teaching_order: list[str]
    source_type: Literal["repository", "paper", "document"]
    source_path: str

    @model_validator(mode="after")
    def validate_references(self) -> Knowledge:
        concept_ids = {c.id for c in self.concepts}

        # teaching_order 中的 ID 必须存在于 concepts 中
        for cid in self.teaching_order:
            if cid not in concept_ids:
                raise ValueError(
                    f"teaching_order 引用了不存在的概念: {cid!r}"
                )

        # dependencies 中的 source/target 必须存在于 concepts 中
        for dep in self.dependencies:
            if dep.source not in concept_ids:
                raise ValueError(
                    f"dependency.source 引用了不存在的概念: {dep.source!r}"
                )
            if dep.target not in concept_ids:
                raise ValueError(
                    f"dependency.target 引用了不存在的概念: {dep.target!r}"
                )

        # teaching_order 应覆盖所有概念：软检查
        missing = concept_ids - set(self.teaching_order)
        if missing:
            logger.warning(
                "teaching_order 未覆盖以下概念: %s", missing,
            )

        # 拓扑排序一致性：软检查，只记录 warning
        order_index = {cid: i for i, cid in enumerate(self.teaching_order)}
        for dep in self.dependencies:
            src_idx = order_index.get(dep.source)
            tgt_idx = order_index.get(dep.target)
            if src_idx is not None and tgt_idx is not None and src_idx > tgt_idx:
                logger.warning(
                    "teaching_order 中 %r (index %d) 出现在 %r (index %d) 之后，"
                    "但 %r 依赖 %r",
                    dep.source, src_idx, dep.target, tgt_idx,
                    dep.target, dep.source,
                )

        return self


# ── Learner 输出 ──


class FeedbackDetail(BaseModel):
    """Learner 反馈详情。

    根据 Feedback.type 填写对应字段，其他字段留空。
    不做条件强制校验——LLM 可能把信息放在"错误"的字段里。
    """

    stuck_point: str | None = None
    assumption: str | None = None
    request: str | None = None
    deeper_question: str | None = None


class Feedback(BaseModel):
    """Learner Agent 输出的结构化反馈。"""

    type: Literal[
        "confused", "go_deeper", "wrong_assumption", "understood", "request_example"
    ]
    concept_id: str
    detail: FeedbackDetail = Field(default_factory=FeedbackDetail)
    confidence: float = Field(ge=0.0, le=1.0)
    understanding_summary: str


# ── Teacher 输出 ──


class TeachingResponse(BaseModel):
    """Teacher Agent 输出的讲解内容。"""

    concept_id: str
    level: int = Field(ge=0, le=5)
    content: str = Field(description="Markdown 格式讲解内容")
    analogies_used: list[str] = Field(default_factory=list)
    code_references: list[CodeReference] = Field(default_factory=list)
    next_concept_id: str | None = None


# ── 对话记录 ──


class ConversationEntry(BaseModel):
    """单条对话记录。

    role 作为鉴别器：teacher 时填 teaching_response，learner 时填 feedback。
    """

    role: Literal["teacher", "learner"]
    teaching_response: TeachingResponse | None = None
    feedback: Feedback | None = None
    raw_text: str

    @model_validator(mode="after")
    def validate_content(self) -> ConversationEntry:
        has_tr = self.teaching_response is not None
        has_fb = self.feedback is not None

        if has_tr == has_fb:
            raise ValueError(
                "teaching_response 和 feedback 必须恰好有一个非空"
            )

        if self.role == "teacher" and not has_tr:
            raise ValueError(
                "role='teacher' 时 teaching_response 不应为空"
            )
        if self.role == "learner" and not has_fb:
            raise ValueError(
                "role='learner' 时 feedback 不应为空"
            )

        return self


# ── 配置 ──


class AudienceProfile(BaseModel):
    """读者画像，从 YAML 文件加载。"""

    name: str
    display_name: str
    math_level: str
    coding_level: str
    domain_knowledge: str
    confusion_triggers: list[str]


class TutorialConfig(BaseModel):
    """单次运行配置。"""

    source: str
    source_type: Literal["repository", "paper", "document"]
    topic: str | None = None
    audience: str = Field(description="读者画像 key，运行时校验有效性")
    output_path: str
    language: str = "zh"
    max_rounds: int = 30
    max_rounds_per_concept: int = 6
    min_importance: int = Field(default=3, ge=1, le=5, description="Skip concepts with importance below this threshold")


# ── Synthesizer 输入 ──


class SynthesisInput(BaseModel):
    """Synthesizer Agent 的输入，聚合所有必要上下文。"""

    conversation: list[ConversationEntry]
    knowledge: Knowledge
    audience: AudienceProfile
