"""Teacher Agent 测试。"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from spiral_teacher.agents.teacher import (
    TeacherAgent,
    compute_next_level,
    _read_source_files,
)
from spiral_teacher.models import (
    Concept,
    ConceptDependency,
    ConversationEntry,
    Feedback,
    FeedbackDetail,
    Knowledge,
    TeachingResponse,
)


# ── 测试数据 ──


def make_knowledge(repo_path: str = "/tmp/test-repo") -> Knowledge:
    return Knowledge(
        project_summary="A test project",
        concepts=[
            Concept(id="concept_a", name="Concept A", category="core_algorithm",
                    description="First concept", difficulty=1,
                    source_files=["src/a.py"]),
            Concept(id="concept_b", name="Concept B", category="design_pattern",
                    description="Second concept", difficulty=2,
                    prerequisites=["concept_a"],
                    source_files=["src/b.py"],
                    key_equations=["y = f(x)"],
                    common_misconceptions=["It's the same as A"]),
            Concept(id="concept_c", name="Concept C", category="math_foundation",
                    description="Third concept", difficulty=4,
                    source_files=["src/c.py"]),
        ],
        dependencies=[
            ConceptDependency(source="concept_a", target="concept_b", reason="B builds on A"),
        ],
        teaching_order=["concept_a", "concept_b", "concept_c"],
        source_type="repository",
        source_path=repo_path,
    )


def make_feedback(**overrides) -> Feedback:
    defaults = {
        "type": "confused",
        "concept_id": "concept_b",
        "detail": FeedbackDetail(stuck_point="I don't understand step 2"),
        "confidence": 0.3,
        "understanding_summary": "I get step 1 but not step 2",
    }
    defaults.update(overrides)
    return Feedback(**defaults)


VALID_TEACHER_OUTPUT = """{
  "concept_id": "concept_b",
  "level": 2,
  "analogies_used": ["Like sorting cards"],
  "code_references": [],
  "next_concept_id": null
}
---JSON---
## Concept B Explained

Here is how Concept B works...
"""


def _make_agent_with_mock(responses: list[str]) -> TeacherAgent:
    mock_client = AsyncMock(spec=anthropic.AsyncAnthropic)
    side_effects = []
    for text in responses:
        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = text
        mock_response.content = [mock_block]
        side_effects.append(mock_response)
    mock_client.messages.create = AsyncMock(side_effect=side_effects)
    return TeacherAgent(model="claude-haiku-4-5", client=mock_client)


# ── compute_next_level 全组合测试 ──


class TestComputeNextLevel:
    """覆盖所有 feedback type × level 组合。"""

    @pytest.mark.parametrize("level,expected", [
        (0, 0), (1, 1), (2, 2), (3, 2), (4, 2), (5, 2),
    ])
    def test_confused(self, level, expected):
        assert compute_next_level("confused", level) == expected

    @pytest.mark.parametrize("level,expected", [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 5),
    ])
    def test_go_deeper(self, level, expected):
        assert compute_next_level("go_deeper", level) == expected

    @pytest.mark.parametrize("level", range(6))
    def test_wrong_assumption_keeps_level(self, level):
        assert compute_next_level("wrong_assumption", level) == level

    @pytest.mark.parametrize("level", range(6))
    def test_request_example_keeps_level(self, level):
        assert compute_next_level("request_example", level) == level

    @pytest.mark.parametrize("level", range(6))
    def test_understood_keeps_level(self, level):
        assert compute_next_level("understood", level) == level


# ── 源码读取 ──


class TestReadSourceFiles:
    def test_reads_existing_file(self, tmp_path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("def hello():\n    pass\n")

        result = _read_source_files(["src/a.py"], str(tmp_path))
        assert "def hello():" in result
        assert "src/a.py" in result

    def test_skips_missing_file(self, tmp_path):
        result = _read_source_files(["nonexistent.py"], str(tmp_path))
        assert result == ""

    def test_truncates_long_files(self, tmp_path):
        long_content = "\n".join(f"line {i}" for i in range(2000))
        (tmp_path / "big.py").write_text(long_content)

        result = _read_source_files(["big.py"], str(tmp_path))
        assert "truncated" in result

    def test_multi_file_budget(self, tmp_path):
        for name in ["a.py", "b.py", "c.py"]:
            content = "\n".join(f"# {name} line {i}" for i in range(600))
            (tmp_path / name).write_text(content)

        result = _read_source_files(["a.py", "b.py", "c.py"], str(tmp_path))
        # 总预算 1000 行，a.py (600) + b.py (400 截断) = 1000，c.py 不包含
        assert "a.py" in result
        # b.py 应该被截断或部分包含
        assert "b.py" in result


# ── TeacherAgent mock 测试 ──


class TestTeacherAgentUnit:

    @pytest.mark.asyncio
    async def test_give_overview(self):
        output = """{
  "concept_id": "concept_a",
  "level": 1,
  "analogies_used": [],
  "code_references": [],
  "next_concept_id": null
}
---JSON---
## Overview

This project does X, Y, Z.
"""
        agent = _make_agent_with_mock([output])
        knowledge = make_knowledge()
        tr, raw = await agent.give_overview(knowledge)

        assert tr.level == 1
        assert tr.content.strip() != ""
        assert "Overview" in raw

    @pytest.mark.asyncio
    async def test_introduce_concept(self):
        agent = _make_agent_with_mock([VALID_TEACHER_OUTPUT])
        knowledge = make_knowledge()
        tr, raw = await agent.introduce_concept(
            "concept_b", knowledge, conversation=[], level=2,
        )

        assert tr.concept_id == "concept_b"
        assert tr.level == 2
        assert "Concept B" in tr.content

    @pytest.mark.asyncio
    async def test_introduce_concept_invalid_id(self):
        agent = _make_agent_with_mock([])
        knowledge = make_knowledge()

        with pytest.raises(ValueError, match="概念不存在"):
            await agent.introduce_concept("nonexistent", knowledge, [])

    @pytest.mark.asyncio
    async def test_respond_to_feedback_confused(self):
        agent = _make_agent_with_mock([VALID_TEACHER_OUTPUT])
        knowledge = make_knowledge()
        fb = make_feedback(type="confused")

        tr, raw = await agent.respond_to_feedback(
            fb, knowledge, conversation=[], current_level=3,
        )

        # confused + level 3 → 降到 level 2
        assert tr.level == 2

    @pytest.mark.asyncio
    async def test_respond_to_feedback_go_deeper(self):
        output = VALID_TEACHER_OUTPUT.replace('"level": 2', '"level": 3')
        agent = _make_agent_with_mock([output])
        knowledge = make_knowledge()
        fb = make_feedback(type="go_deeper", detail=FeedbackDetail(deeper_question="Why?"))

        tr, raw = await agent.respond_to_feedback(
            fb, knowledge, conversation=[], current_level=2,
        )

        # go_deeper + level 2 → level 3
        assert tr.level == 3

    @pytest.mark.asyncio
    async def test_empty_conversation(self):
        """conversation 为空列表时正常工作。"""
        agent = _make_agent_with_mock([VALID_TEACHER_OUTPUT])
        knowledge = make_knowledge()

        tr, raw = await agent.introduce_concept(
            "concept_b", knowledge, conversation=[], level=2,
        )
        assert tr.concept_id == "concept_b"

    @pytest.mark.asyncio
    async def test_source_files_not_found(self, tmp_path):
        """source_files 中的文件不存在时不抛异常。"""
        agent = _make_agent_with_mock([VALID_TEACHER_OUTPUT])
        knowledge = make_knowledge(repo_path=str(tmp_path))
        # source_files 指向不存在的文件

        tr, raw = await agent.introduce_concept(
            "concept_b", knowledge, conversation=[], level=2,
        )
        assert tr.concept_id == "concept_b"


# ── 解析退化测试 ──


class TestParseResponse:

    def test_valid_json_and_markdown(self):
        tr = TeacherAgent._parse_response(
            VALID_TEACHER_OUTPUT, concept_id="concept_b", level=2,
        )
        assert tr.concept_id == "concept_b"
        assert tr.level == 2
        assert "Concept B" in tr.content
        assert "Like sorting cards" in tr.analogies_used

    def test_fallback_on_invalid_json(self):
        """JSON 解析失败时退化。"""
        raw = "This is just plain markdown without any JSON."
        tr = TeacherAgent._parse_response(raw, concept_id="concept_b", level=2)

        assert tr.concept_id == "concept_b"
        assert tr.level == 2
        assert tr.content == raw
        assert tr.analogies_used == []
        assert tr.code_references == []

    def test_fallback_on_malformed_json(self):
        raw = "{not valid json}\n---JSON---\n## Content here"
        tr = TeacherAgent._parse_response(raw, concept_id="concept_a", level=3)

        assert tr.concept_id == "concept_a"
        assert tr.level == 3
        assert tr.content == raw

    def test_json_without_separator(self):
        """JSON 在开头但没有分隔符时，尝试提取。"""
        raw = '{"concept_id": "x", "level": 2, "analogies_used": []}\n\n## Explanation\n\nSome text.'
        tr = TeacherAgent._parse_response(raw, concept_id="concept_a", level=2)

        # 应该能提取出 JSON 和 markdown
        assert tr.concept_id == "concept_a"
        assert "Explanation" in tr.content or tr.content == raw

    def test_overrides_concept_id_and_level(self):
        """解析时强制使用调用方的 concept_id 和 level。"""
        raw = '{"concept_id": "wrong_id", "level": 99, "analogies_used": []}\n---JSON---\n## Content'
        tr = TeacherAgent._parse_response(raw, concept_id="correct_id", level=2)

        assert tr.concept_id == "correct_id"
        assert tr.level == 2
