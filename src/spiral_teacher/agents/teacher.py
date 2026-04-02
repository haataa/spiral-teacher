"""Teacher Agent: 多层级讲解生成。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic
from pydantic import ValidationError

from spiral_teacher.models import (
    ConversationEntry,
    Feedback,
    Knowledge,
    TeachingResponse,
)
from spiral_teacher.utils import extract_json_from_text

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "teacher.md"

SEPARATOR = "---JSON---"

MAX_HISTORY_ENTRIES = 10
MAX_SOURCE_LINES = 1000


class TeacherError(Exception):
    """Teacher Agent 错误。"""

    def __init__(self, message: str, raw_response: str | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.cause = cause


# ── 层级切换（纯函数，独立可测试）──


def compute_next_level(feedback_type: str, current_level: int) -> int:
    """根据 Learner 反馈类型和当前层级，计算下一个教学层级。

    规则来自 PLAN.md：
    - confused + level >= 3 → 降到 2
    - confused + level <= 2 → 保持
    - go_deeper → +1 (max 5)
    - wrong_assumption → 保持
    - request_example → 保持
    - understood → 保持（不应到达此函数，但做防御）
    """
    if feedback_type == "confused":
        return 2 if current_level >= 3 else current_level
    if feedback_type == "go_deeper":
        return min(current_level + 1, 5)
    # wrong_assumption, request_example, understood
    return current_level


# ── 源码读取 ──


def _read_source_files(
    source_files: list[str],
    repo_path: str,
) -> str:
    """读取概念关联的源码文件，返回格式化的代码片段。

    多文件总行数不超过 MAX_SOURCE_LINES。
    文件不存在时跳过并记录 warning。
    """
    root = Path(repo_path)
    parts: list[str] = []
    total_lines = 0

    for rel_path in source_files:
        full_path = root / rel_path
        if not full_path.exists():
            logger.warning("源码文件不存在，跳过: %s", full_path)
            continue

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            logger.warning("无法读取源码文件 %s: %s", full_path, e)
            continue

        lines = content.splitlines(keepends=True)
        remaining = MAX_SOURCE_LINES - total_lines
        if remaining <= 0:
            break

        if len(lines) > remaining:
            content = "".join(lines[:remaining])
            content += f"\n[... truncated, total {len(lines)} lines ...]\n"
            total_lines = MAX_SOURCE_LINES
        else:
            total_lines += len(lines)

        parts.append(f"### Source: {rel_path}\n```\n{content}\n```\n")

    return "\n".join(parts)


# ── TeacherAgent ──


class TeacherAgent:
    """多层级讲解生成 Agent。"""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        client: anthropic.AsyncAnthropic | None = None,
        language: str = "zh",
    ):
        self.model = model
        self.client = client or anthropic.AsyncAnthropic()
        self.language = language
        base_prompt = PROMPT_PATH.read_text(encoding="utf-8")
        lang_instruction = _language_instruction(language)
        self._system_prompt = base_prompt + "\n\n" + lang_instruction

    async def give_overview(
        self,
        knowledge: Knowledge,
    ) -> tuple[TeachingResponse, str]:
        """生成项目整体鸟瞰讲解（Level 1）。"""
        concept_id = knowledge.teaching_order[0] if knowledge.teaching_order else "overview"

        user_msg = self._build_overview_message(knowledge)
        raw = await self._call_llm(user_msg)
        tr = self._parse_response(raw, concept_id=concept_id, level=1)
        return tr, raw

    async def introduce_concept(
        self,
        concept_id: str,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
        level: int = 2,
    ) -> tuple[TeachingResponse, str]:
        """引入一个新概念。"""
        concept = self._get_concept(concept_id, knowledge)

        user_msg = self._build_introduce_message(
            concept, knowledge, conversation, level,
        )
        raw = await self._call_llm(user_msg)
        tr = self._parse_response(raw, concept_id=concept_id, level=level)
        return tr, raw

    async def respond_to_feedback(
        self,
        feedback: Feedback,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
        current_level: int,
    ) -> tuple[TeachingResponse, str]:
        """响应 Learner 的反馈。"""
        # understood 防御
        if feedback.type == "understood":
            logger.warning(
                "respond_to_feedback 收到 understood 反馈，"
                "应由 Orchestrator 调用 introduce_concept"
            )
            # 尝试推进到下一个概念
            next_id = self._find_next_concept(feedback.concept_id, knowledge)
            if next_id:
                return await self.introduce_concept(
                    next_id, knowledge, conversation, level=2,
                )
            # 没有下一个概念了，讲当前概念的总结
            current_level = 1

        target_level = compute_next_level(feedback.type, current_level)

        concept = self._get_concept(feedback.concept_id, knowledge)
        user_msg = self._build_feedback_message(
            feedback, concept, knowledge, conversation, target_level,
        )
        raw = await self._call_llm(user_msg)
        tr = self._parse_response(raw, concept_id=feedback.concept_id, level=target_level)
        return tr, raw

    # ── 消息构造 ──

    def _build_overview_message(self, knowledge: Knowledge) -> str:
        parts = [
            "## Task\n\nGenerate a Level 1 bird's-eye overview of this project.\n",
            f"## Project Summary\n\n{knowledge.project_summary}\n",
            "## Concepts in Teaching Order\n\n",
        ]

        for cid in knowledge.teaching_order:
            concept = next((c for c in knowledge.concepts if c.id == cid), None)
            if concept:
                parts.append(f"- **{concept.name}** (difficulty {concept.difficulty}): {concept.description}\n")

        return "\n".join(parts)

    def _build_introduce_message(
        self,
        concept: object,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
        level: int,
    ) -> str:
        parts = [
            f"## Task\n\nIntroduce the concept **{concept.name}** at Level {level}.\n",
            self._format_concept_info(concept),
        ]

        source_text = _read_source_files(concept.source_files, knowledge.source_path)
        if source_text:
            parts.append(f"## Relevant Source Code\n\n{source_text}\n")

        history = self._format_history(conversation)
        if history:
            parts.append(f"## Conversation History\n\n{history}\n")

        return "\n".join(parts)

    def _build_feedback_message(
        self,
        feedback: Feedback,
        concept: object,
        knowledge: Knowledge,
        conversation: list[ConversationEntry],
        target_level: int,
    ) -> str:
        parts = [
            f"## Task\n\nRespond to learner feedback on **{concept.name}** at Level {target_level}.\n",
            f"## Learner Feedback\n\n"
            f"- Type: **{feedback.type}**\n"
            f"- Confidence: {feedback.confidence}\n"
            f"- Understanding: {feedback.understanding_summary}\n",
        ]

        # 反馈详情
        d = feedback.detail
        if d.stuck_point:
            parts.append(f"- Stuck point: {d.stuck_point}\n")
        if d.assumption:
            parts.append(f"- Assumption: {d.assumption}\n")
        if d.request:
            parts.append(f"- Request: {d.request}\n")
        if d.deeper_question:
            parts.append(f"- Deeper question: {d.deeper_question}\n")

        parts.append(self._format_concept_info(concept))

        source_text = _read_source_files(concept.source_files, knowledge.source_path)
        if source_text:
            parts.append(f"## Relevant Source Code\n\n{source_text}\n")

        history = self._format_history(conversation)
        if history:
            parts.append(f"## Conversation History\n\n{history}\n")

        return "\n".join(parts)

    @staticmethod
    def _format_concept_info(concept) -> str:
        parts = [
            f"## Concept: {concept.name}\n",
            f"- ID: {concept.id}\n",
            f"- Category: {concept.category}\n",
            f"- Difficulty: {concept.difficulty}/5\n",
            f"- Description: {concept.description}\n",
        ]
        if concept.prerequisites:
            parts.append(f"- Prerequisites: {', '.join(concept.prerequisites)}\n")
        if concept.key_equations:
            parts.append(f"- Key equations: {'; '.join(concept.key_equations)}\n")
        if concept.common_misconceptions:
            parts.append(f"- Common misconceptions: {'; '.join(concept.common_misconceptions)}\n")
        if concept.source_files:
            parts.append(f"- Source files: {', '.join(concept.source_files)}\n")
        return "\n".join(parts)

    @staticmethod
    def _format_history(conversation: list[ConversationEntry]) -> str:
        if not conversation:
            return ""

        recent = conversation[-MAX_HISTORY_ENTRIES:]
        parts = []
        for entry in recent:
            role_label = "Teacher" if entry.role == "teacher" else "Learner"
            # 截取 raw_text 前 500 字符避免过长
            text = entry.raw_text[:500]
            if len(entry.raw_text) > 500:
                text += "\n[... truncated ...]"
            parts.append(f"**{role_label}:**\n{text}\n")

        return "\n---\n".join(parts)

    # ── LLM 调用 ──

    async def _call_llm(self, user_message: str) -> str:
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self._system_prompt,
                messages=[{"role": "user", "content": user_message}],
                timeout=300.0,
            )
        except anthropic.AuthenticationError:
            raise
        except anthropic.APIError as e:
            raise TeacherError(f"API 调用失败: {e}", cause=e) from e

        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    # ── 结果解析 ──

    @staticmethod
    def _parse_response(
        raw: str,
        concept_id: str,
        level: int,
    ) -> TeachingResponse:
        """解析 LLM 输出为 TeachingResponse。

        尝试多种策略提取 JSON 元数据。
        解析失败时退化为用已知信息填充（content = 整个 raw 文本）。
        """
        parsed, markdown_part = extract_json_from_text(raw)

        if parsed is not None:
            try:
                parsed["content"] = markdown_part or parsed.get("content", "")
                parsed["concept_id"] = concept_id
                parsed["level"] = level
                return TeachingResponse.model_validate(parsed)
            except ValidationError as e:
                logger.warning("Teacher JSON 校验失败，使用退化策略: %s", e)

        # 退化：用已知信息填充，content 是完整的 raw 文本
        return TeachingResponse(
            concept_id=concept_id,
            level=level,
            content=raw,
        )

    # ── 辅助方法 ──

    @staticmethod
    def _get_concept(concept_id: str, knowledge: Knowledge):
        """从知识图谱中获取概念，不存在时抛出 ValueError。"""
        for c in knowledge.concepts:
            if c.id == concept_id:
                return c
        raise ValueError(f"概念不存在: {concept_id!r}")

    @staticmethod
    def _find_next_concept(current_id: str, knowledge: Knowledge) -> str | None:
        """在 teaching_order 中找到下一个概念。"""
        order = knowledge.teaching_order
        try:
            idx = order.index(current_id)
            if idx + 1 < len(order):
                return order[idx + 1]
        except ValueError:
            pass
        return None


def _language_instruction(language: str) -> str:
    """生成语言指令。"""
    if language == "zh":
        return (
            "## Language\n\n"
            "IMPORTANT: Write ALL explanations in Chinese (中文). "
            "Keep code snippets, file paths, function names, and technical terms "
            "(API names, library names, protocol names) in English. "
            "数学公式和代码保持英文，解释和叙述用中文。"
        )
    if language == "en":
        return ""
    return f"## Language\n\nWrite all explanations in {language}. Keep code and technical terms in English."
