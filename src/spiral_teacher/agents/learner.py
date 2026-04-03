"""Learner Agent: 模拟学习者的理解过程。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from string import Template

import anthropic
from pydantic import ValidationError

from spiral_teacher.models import (
    AudienceProfile,
    ConversationEntry,
    Feedback,
    FeedbackDetail,
    Knowledge,
)
from spiral_teacher.utils import extract_json_from_text

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "learner.md"

SEPARATOR = "---JSON---"

MAX_HISTORY_ENTRIES = 10


# ── validate_feedback: 代码层硬校验 ──


def validate_feedback(feedback: Feedback) -> Feedback:
    """对 Learner 输出的 Feedback 做代码层硬校验。

    防止 Learner "手下留情"。此函数由 Orchestrator 在收到 Feedback 后调用。

    规则：
    1. understood + confidence < 0.85 → 降级为 go_deeper
    2. understood + understanding_summary < 50 字符 → 降级为 go_deeper
    3. confused + confidence > 0.6 → 修正 confidence 为 0.5

    返回修正后的 Feedback（可能是新实例）。
    """
    fb_type = feedback.type
    confidence = feedback.confidence

    # 规则 1 + 2: understood 降级检查
    if fb_type == "understood":
        if confidence < 0.85:
            logger.info(
                "validate_feedback: understood 但 confidence=%.2f < 0.85，降级为 go_deeper",
                confidence,
            )
            return Feedback(
                type="go_deeper",
                concept_id=feedback.concept_id,
                detail=FeedbackDetail(
                    deeper_question="I said I understood but my confidence is low. Can you help me verify my understanding?",
                ),
                confidence=confidence,
                understanding_summary=feedback.understanding_summary,
            )

        if len(feedback.understanding_summary) < 50:
            logger.info(
                "validate_feedback: understood 但 understanding_summary 太短 (%d 字符)，降级为 go_deeper",
                len(feedback.understanding_summary),
            )
            return Feedback(
                type="go_deeper",
                concept_id=feedback.concept_id,
                detail=FeedbackDetail(
                    deeper_question="I think I understood but I can't fully articulate it. Can you help me verify?",
                ),
                confidence=min(confidence, 0.7),
                understanding_summary=feedback.understanding_summary,
            )

    # 规则 3: confused 但 confidence 过高
    if fb_type == "confused" and confidence > 0.6:
        logger.info(
            "validate_feedback: confused 但 confidence=%.2f > 0.6，修正为 0.5",
            confidence,
        )
        return Feedback(
            type=feedback.type,
            concept_id=feedback.concept_id,
            detail=feedback.detail,
            confidence=0.5,
            understanding_summary=feedback.understanding_summary,
        )

    return feedback


def validate_concept_feedback(
    feedback: Feedback,
    concept_difficulty: int,
    rounds_on_current: int,
    past_feedback_types: list[str],
) -> Feedback:
    """概念级反馈校验（有状态）。

    在 validate_feedback 之后调用，增加需要上下文信息的校验规则。

    规则：
    4. 高难度概念首轮 understood → 降级为 go_deeper
    5. 高难度概念从未 request_example 就 understood → 降级为 request_example
    """
    if feedback.type != "understood":
        return feedback

    # Rule 4: 高难度概念首轮不允许直接 understood (difficulty >= 4)
    if concept_difficulty >= 4 and rounds_on_current <= 1:
        logger.info(
            "validate_concept_feedback: 高难度概念 (difficulty=%d) 首轮 understood，降级为 go_deeper",
            concept_difficulty,
        )
        return Feedback(
            type="go_deeper",
            concept_id=feedback.concept_id,
            detail=FeedbackDetail(
                deeper_question="This is a complex concept. Let me verify my understanding more carefully.",
            ),
            confidence=min(feedback.confidence, 0.75),
            understanding_summary=feedback.understanding_summary,
        )

    # Rule 5: 高难度概念从未 request_example 就不允许 understood (difficulty >= 3)
    if concept_difficulty >= 3 and "request_example" not in past_feedback_types:
        logger.info(
            "validate_concept_feedback: 高难度概念 (difficulty=%d) 未经 request_example 就 understood，降级",
            concept_difficulty,
        )
        return Feedback(
            type="request_example",
            concept_id=feedback.concept_id,
            detail=FeedbackDetail(
                request="Can you give me a concrete example with specific numbers to verify my understanding?",
            ),
            confidence=min(feedback.confidence, 0.80),
            understanding_summary=feedback.understanding_summary,
        )

    return feedback


# ── LearnerAgent ──


class LearnerAgent:
    """模拟学习者的理解过程。"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        client: anthropic.AsyncAnthropic | None = None,
        language: str = "zh",
    ):
        self.model = model
        self.client = client or anthropic.AsyncAnthropic()
        self.language = language
        self._prompt_template = PROMPT_PATH.read_text(encoding="utf-8")

    async def react(
        self,
        teacher_response: str,
        concept_id: str,
        teaching_level: int,
        audience: AudienceProfile,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
    ) -> tuple[Feedback, str]:
        """阅读 Teacher 的讲解，产生学习者反馈。"""
        system_prompt = self._build_system_prompt(audience, teaching_level)
        user_msg = self._build_user_message(
            teacher_response, concept_id, knowledge, conversation,
        )

        raw = await self._call_llm(system_prompt, user_msg)
        feedback = self._parse_response(raw, concept_id)
        return feedback, raw

    # ── Prompt 构造 ──

    def _build_system_prompt(
        self,
        audience: AudienceProfile,
        teaching_level: int,
    ) -> str:
        """动态注入画像信息到 system prompt。"""
        # 构造 confusion_triggers 检查清单
        checklist = "\n".join(
            f"- [ ] {trigger}" for trigger in audience.confusion_triggers
        )

        base = Template(self._prompt_template).safe_substitute(
            display_name=audience.display_name,
            math_level=audience.math_level,
            coding_level=audience.coding_level,
            domain_knowledge=audience.domain_knowledge,
            confusion_triggers=", ".join(audience.confusion_triggers),
            confusion_triggers_checklist=checklist,
            teaching_level=teaching_level,
        )
        return base + "\n\n" + _learner_extra_instructions(self.language)

    @staticmethod
    def _build_user_message(
        teacher_response: str,
        concept_id: str,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
    ) -> str:
        parts = []

        # 当前概念信息
        concept = next((c for c in knowledge.concepts if c.id == concept_id), None)
        if concept:
            parts.append(f"## Current Concept: {concept.name}\n")
            parts.append(f"- Difficulty: {concept.difficulty}/5\n")
            parts.append(f"- Description: {concept.description}\n")
            if concept.prerequisites:
                parts.append(f"- Prerequisites: {', '.join(concept.prerequisites)}\n")
            if concept.common_misconceptions:
                parts.append(f"- Known misconceptions: {'; '.join(concept.common_misconceptions)}\n")
            parts.append("")

        # Teacher 的讲解
        parts.append("## Teacher's Explanation\n")
        parts.append(teacher_response)
        parts.append("")

        # 对话历史
        if conversation:
            recent = conversation[-MAX_HISTORY_ENTRIES:]
            parts.append("## Previous Conversation\n")
            for entry in recent:
                role_label = "Teacher" if entry.role == "teacher" else "Learner"
                text = entry.raw_text[:500]
                if len(entry.raw_text) > 500:
                    text += "\n[... truncated ...]"
                parts.append(f"**{role_label}:** {text}\n")

        # 覆盖情况
        covered = set()
        for entry in conversation:
            if entry.feedback and entry.feedback.type == "understood":
                covered.add(entry.feedback.concept_id)
        uncovered = [c.id for c in knowledge.concepts if c.id not in covered]

        if covered:
            parts.append(f"\n## Progress\n- Covered: {', '.join(sorted(covered))}")
        if uncovered:
            parts.append(f"- Remaining: {', '.join(uncovered)}")

        return "\n".join(parts)

    # ── LLM 调用 ──

    async def _call_llm(self, system_prompt: str, user_message: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                timeout=120.0,
            )
        except anthropic.AuthenticationError:
            raise
        except anthropic.APIError as e:
            raise RuntimeError(f"Learner API 调用失败: {e}") from e

        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    # ── 结果解析 ──

    @staticmethod
    def _parse_response(raw: str, concept_id: str) -> Feedback:
        """解析 LLM 输出为 Feedback。失败时退化为 confused。"""
        parsed, md_part = extract_json_from_text(raw)

        if parsed is not None:
            try:
                parsed["concept_id"] = concept_id
                return Feedback.model_validate(parsed)
            except ValidationError as e:
                logger.warning("Learner JSON 校验失败，退化为 confused: %s", e)

        # 退化
        logger.warning("Learner 输出无法提取 JSON，退化为 confused")
        return Feedback(
            type="confused",
            concept_id=concept_id,
            detail=FeedbackDetail(),
            confidence=0.3,
            understanding_summary=raw[:500] if raw else "Failed to parse learner response",
        )


def _learner_extra_instructions(language: str) -> str:
    """Learner 的额外关注点和语言指令。"""
    parts = [
        "## Extra Focus Areas",
        "",
        "As a learner, you particularly care about:",
        "",
        "1. **Core Innovation**: What is novel or unique about this approach? "
        "How is it different from existing methods? Always ask about the key insight.",
        "",
        "2. **Concrete Examples**: Abstract explanations are not enough. "
        "Refer to Step 7 in your Understanding Check Protocol for when to use `request_example`.",
        "",
        "3. **Practical Usage**: How do I actually use this? What are the benefits? "
        "What problems does it solve that I couldn't solve before? "
        "Ask 'what does this enable me to do?' and 'what would happen without this?'",
        "",
        "4. **Connections**: When moving to a new concept, briefly ask how it connects "
        "to what you already learned. This helps build a coherent mental model.",
    ]

    if language == "zh":
        parts.extend([
            "",
            "## Language",
            "",
            "IMPORTANT: Write ALL your responses (understanding_summary, detail fields, "
            "and your natural response) in Chinese (中文). "
            "Keep code, file paths, and technical terms in English.",
        ])
    elif language != "en":
        parts.extend([
            "",
            f"## Language",
            "",
            f"Write all responses in {language}. Keep code and technical terms in English.",
        ])

    return "\n".join(parts)
