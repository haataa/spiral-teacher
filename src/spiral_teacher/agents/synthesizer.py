"""Synthesizer Agent: 将对话历史合成为连贯的教学文档。"""

from __future__ import annotations

import logging
from pathlib import Path

import anthropic

from spiral_teacher.models import SynthesisInput

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "synthesizer.md"


class SynthesizerAgent:
    """将多轮对话合成为结构化教学文档。"""

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.model = model
        self.client = client or anthropic.AsyncAnthropic()
        self._system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    async def compile(self, synthesis_input: SynthesisInput, language: str = "en") -> str:
        """将对话历史合成为 Markdown 教学文档。

        Args:
            synthesis_input: 包含 conversation, knowledge, audience
            language: 输出语言，如 "zh" (中文), "en" (English)

        Returns:
            Markdown 格式的教学文档
        """
        user_msg = self._build_message(synthesis_input, language)
        return await self._call_llm(user_msg)

    def _build_message(self, si: SynthesisInput, language: str = "en") -> str:
        parts = []

        # 知识图谱摘要
        parts.append("## Knowledge Graph\n")
        parts.append(f"**Project:** {si.knowledge.project_summary}\n")
        parts.append(f"**Source:** {si.knowledge.source_path}\n")
        parts.append(f"\n### Concepts ({len(si.knowledge.concepts)} total)\n")

        for cid in si.knowledge.teaching_order:
            concept = next((c for c in si.knowledge.concepts if c.id == cid), None)
            if concept:
                parts.append(
                    f"- **{concept.name}** (difficulty {concept.difficulty}): "
                    f"{concept.description}\n"
                )

        parts.append(f"\n### Dependencies\n")
        for dep in si.knowledge.dependencies:
            parts.append(f"- {dep.source} -> {dep.target}: {dep.reason}\n")

        # 读者画像
        parts.append(f"\n## Target Audience: {si.audience.display_name}\n")
        parts.append(f"- Math: {si.audience.math_level}\n")
        parts.append(f"- Coding: {si.audience.coding_level}\n")
        parts.append(f"- Domain: {si.audience.domain_knowledge}\n")

        # 对话历史
        parts.append(f"\n## Conversation ({len(si.conversation)} messages)\n")

        for entry in si.conversation:
            if entry.role == "teacher":
                tr = entry.teaching_response
                meta = f"[Level {tr.level}, concept: {tr.concept_id}]" if tr else ""
                parts.append(f"\n### Teacher {meta}\n\n{entry.raw_text}\n")
            else:
                fb = entry.feedback
                if fb:
                    meta = f"[{fb.type}, confidence={fb.confidence:.2f}, concept: {fb.concept_id}]"
                else:
                    meta = ""
                parts.append(f"\n### Learner {meta}\n\n{entry.raw_text}\n")

        # 语言指令
        lang_names = {"zh": "Chinese (中文)", "en": "English", "ja": "Japanese"}
        lang_name = lang_names.get(language, language)
        parts.append(f"\n## OUTPUT LANGUAGE\n\nWrite the entire tutorial in **{lang_name}**. "
                     f"Keep code snippets, file paths, and technical terms (API names, library names) in English.\n")

        return "\n".join(parts)

    async def _call_llm(self, user_message: str) -> str:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=16384,
            system=self._system_prompt,
            messages=[{"role": "user", "content": user_message}],
            timeout=600.0,
        )

        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)
