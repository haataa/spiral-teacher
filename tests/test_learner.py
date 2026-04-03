"""Learner Agent 测试。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from spiral_teacher.agents.learner import LearnerAgent, validate_concept_feedback, validate_feedback
from spiral_teacher.models import (
    AudienceProfile,
    Concept,
    ConceptDependency,
    ConversationEntry,
    Feedback,
    FeedbackDetail,
    Knowledge,
    TeachingResponse,
)


# ── 测试数据 ──


def make_audience() -> AudienceProfile:
    return AudienceProfile(
        name="ml_engineer",
        display_name="ML 工程师",
        math_level="本科数学完整，熟悉优化和概率",
        coding_level="Python/C++ 熟练",
        domain_knowledge="熟悉 Transformer、KV 缓存",
        confusion_triggers=[
            "理论为什么在工程上成立的 gap",
            "多个近似叠加后误差是否可控",
        ],
    )


def make_knowledge() -> Knowledge:
    return Knowledge(
        project_summary="A test project",
        concepts=[
            Concept(id="concept_a", name="Concept A", category="core",
                    description="First concept", difficulty=1),
            Concept(id="concept_b", name="Concept B", category="core",
                    description="Second concept", difficulty=3,
                    prerequisites=["concept_a"],
                    common_misconceptions=["It's the same as A"]),
        ],
        dependencies=[
            ConceptDependency(source="concept_a", target="concept_b", reason="B depends on A"),
        ],
        teaching_order=["concept_a", "concept_b"],
        source_type="repository",
        source_path="/tmp/test",
    )


VALID_LEARNER_OUTPUT = """{
  "type": "confused",
  "concept_id": "concept_b",
  "detail": {
    "stuck_point": "I don't understand why step 2 follows from step 1"
  },
  "confidence": 0.4,
  "understanding_summary": "I understand that we need to transform the distribution, but I don't see why the Hadamard matrix specifically achieves uniformity."
}
---JSON---
Hmm, I get that we're trying to make the distribution uniform, but I'm stuck on why the Hadamard matrix is the right tool here. Could you explain what property of the Hadamard matrix makes it work for this purpose?
"""


def _make_agent_with_mock(responses: list[str]) -> LearnerAgent:
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
    return LearnerAgent(model="claude-haiku-4-5", client=mock_client)


# ── validate_feedback 测试（核心：防手下留情）──


class TestValidateFeedback:
    """覆盖所有降级/修正场景。"""

    def test_understood_low_confidence_downgrades(self):
        """understood + confidence < 0.85 → go_deeper"""
        fb = Feedback(
            type="understood", concept_id="c", confidence=0.5,
            detail=FeedbackDetail(),
            understanding_summary="I think I get it" * 10,
        )
        result = validate_feedback(fb)
        assert result.type == "go_deeper"
        assert result.confidence == 0.5

    def test_understood_high_confidence_passes(self):
        """understood + confidence >= 0.85 + long summary → 通过"""
        fb = Feedback(
            type="understood", concept_id="c", confidence=0.9,
            detail=FeedbackDetail(),
            understanding_summary="This concept works by transforming the input distribution through an orthogonal matrix, which preserves norms while spreading energy across dimensions.",
        )
        result = validate_feedback(fb)
        assert result.type == "understood"
        assert result.confidence == 0.9

    def test_understood_short_summary_downgrades(self):
        """understood + summary < 50 chars → go_deeper"""
        fb = Feedback(
            type="understood", concept_id="c", confidence=0.95,
            detail=FeedbackDetail(),
            understanding_summary="I get it.",
        )
        result = validate_feedback(fb)
        assert result.type == "go_deeper"
        assert result.confidence <= 0.7

    def test_confused_high_confidence_corrected(self):
        """confused + confidence > 0.6 → confidence 修正为 0.5"""
        fb = Feedback(
            type="confused", concept_id="c", confidence=0.8,
            detail=FeedbackDetail(stuck_point="step 2"),
            understanding_summary="I'm confused about step 2",
        )
        result = validate_feedback(fb)
        assert result.type == "confused"
        assert result.confidence == 0.5

    def test_confused_normal_confidence_passes(self):
        """confused + confidence <= 0.6 → 不修改"""
        fb = Feedback(
            type="confused", concept_id="c", confidence=0.3,
            detail=FeedbackDetail(stuck_point="step 2"),
            understanding_summary="Lost",
        )
        result = validate_feedback(fb)
        assert result.type == "confused"
        assert result.confidence == 0.3

    def test_go_deeper_passes_through(self):
        """go_deeper 不受影响。"""
        fb = Feedback(
            type="go_deeper", concept_id="c", confidence=0.7,
            detail=FeedbackDetail(deeper_question="Why?"),
            understanding_summary="I get the intuition, want to see the math",
        )
        result = validate_feedback(fb)
        assert result.type == "go_deeper"
        assert result.confidence == 0.7

    def test_wrong_assumption_passes_through(self):
        """wrong_assumption 不受影响。"""
        fb = Feedback(
            type="wrong_assumption", concept_id="c", confidence=0.6,
            detail=FeedbackDetail(assumption="I think X is true"),
            understanding_summary="My assumption is X",
        )
        result = validate_feedback(fb)
        assert result.type == "wrong_assumption"

    def test_request_example_passes_through(self):
        """request_example 不受影响。"""
        fb = Feedback(
            type="request_example", concept_id="c", confidence=0.5,
            detail=FeedbackDetail(request="A numerical example"),
            understanding_summary="Need to see it with numbers",
        )
        result = validate_feedback(fb)
        assert result.type == "request_example"


# ── validate_concept_feedback 测试（概念级校验）──


class TestValidateConceptFeedback:
    """覆盖概念级降级场景。"""

    def _make_understood(self, confidence=0.9):
        return Feedback(
            type="understood", concept_id="c", confidence=confidence,
            detail=FeedbackDetail(),
            understanding_summary="This concept works by transforming the input through an orthogonal matrix, preserving norms while redistributing energy across dimensions.",
        )

    def test_hard_concept_first_round_downgrades_to_go_deeper(self):
        """difficulty >= 4 首轮 understood → go_deeper"""
        fb = self._make_understood()
        result = validate_concept_feedback(fb, concept_difficulty=4, rounds_on_current=1, past_feedback_types=[])
        assert result.type == "go_deeper"
        assert result.confidence <= 0.75

    def test_medium_concept_first_round_passes(self):
        """difficulty=3 首轮 understood → 不拦截（Rule 4 仅对 >= 4 生效）"""
        fb = self._make_understood()
        # Rule 5 仍会拦截（无 request_example），所以降级为 request_example 而非 understood
        result = validate_concept_feedback(fb, concept_difficulty=3, rounds_on_current=1, past_feedback_types=[])
        assert result.type == "request_example"  # Rule 5 kicks in

    def test_medium_concept_with_example_passes(self):
        """difficulty=3 有 request_example 历史时 understood → 通过"""
        fb = self._make_understood()
        result = validate_concept_feedback(
            fb, concept_difficulty=3, rounds_on_current=2,
            past_feedback_types=["request_example"],
        )
        assert result.type == "understood"

    def test_hard_concept_no_example_downgrades_to_request_example(self):
        """高难度概念未经 request_example 就 understood → request_example"""
        fb = self._make_understood()
        result = validate_concept_feedback(
            fb, concept_difficulty=3, rounds_on_current=2,
            past_feedback_types=["go_deeper"],
        )
        assert result.type == "request_example"
        assert result.confidence <= 0.80

    def test_easy_concept_first_round_passes(self):
        """低难度概念首轮 understood → 通过"""
        fb = self._make_understood()
        result = validate_concept_feedback(fb, concept_difficulty=2, rounds_on_current=1, past_feedback_types=[])
        assert result.type == "understood"
        assert result.confidence == 0.9

    def test_hard_concept_with_example_history_passes(self):
        """高难度概念有 request_example 历史时 understood → 通过"""
        fb = self._make_understood()
        result = validate_concept_feedback(
            fb, concept_difficulty=4, rounds_on_current=3,
            past_feedback_types=["go_deeper", "request_example"],
        )
        assert result.type == "understood"
        assert result.confidence == 0.9

    def test_non_understood_passes_through(self):
        """非 understood 类型不受影响。"""
        fb = Feedback(
            type="confused", concept_id="c", confidence=0.3,
            detail=FeedbackDetail(stuck_point="step 2"),
            understanding_summary="Lost on step 2",
        )
        result = validate_concept_feedback(fb, concept_difficulty=5, rounds_on_current=1, past_feedback_types=[])
        assert result.type == "confused"
        assert result.confidence == 0.3

    def test_rule4_takes_priority_over_rule5(self):
        """首轮 + 无 example → Rule 4 (go_deeper) 优先于 Rule 5 (request_example)"""
        fb = self._make_understood()
        result = validate_concept_feedback(fb, concept_difficulty=4, rounds_on_current=1, past_feedback_types=[])
        assert result.type == "go_deeper"


# ── LearnerAgent mock 测试 ──


class TestLearnerAgentUnit:

    @pytest.mark.asyncio
    async def test_react_valid_output(self):
        agent = _make_agent_with_mock([VALID_LEARNER_OUTPUT])
        fb, raw = await agent.react(
            teacher_response="Here is how concept B works...",
            concept_id="concept_b",
            teaching_level=2,
            audience=make_audience(),
            knowledge=make_knowledge(),
            conversation=[],
        )

        assert isinstance(fb, Feedback)
        assert fb.type == "confused"
        assert fb.concept_id == "concept_b"
        assert fb.confidence == 0.4
        assert len(fb.understanding_summary) > 0

    @pytest.mark.asyncio
    async def test_react_empty_conversation(self):
        """conversation 为空时正常工作。"""
        agent = _make_agent_with_mock([VALID_LEARNER_OUTPUT])
        fb, raw = await agent.react(
            teacher_response="Explanation...",
            concept_id="concept_a",
            teaching_level=1,
            audience=make_audience(),
            knowledge=make_knowledge(),
            conversation=[],
        )
        assert isinstance(fb, Feedback)

    @pytest.mark.asyncio
    async def test_react_with_history(self):
        """有对话历史时正常工作。"""
        history = [
            ConversationEntry(
                role="teacher",
                teaching_response=TeachingResponse(
                    concept_id="concept_a", level=2,
                    content="Here is concept A...",
                ),
                raw_text="Here is concept A...",
            ),
            ConversationEntry(
                role="learner",
                feedback=Feedback(
                    type="understood", concept_id="concept_a",
                    confidence=0.9,
                    understanding_summary="Concept A is about X and works by doing Y.",
                ),
                raw_text="I understand concept A.",
            ),
        ]
        agent = _make_agent_with_mock([VALID_LEARNER_OUTPUT])
        fb, raw = await agent.react(
            teacher_response="Now let's look at concept B...",
            concept_id="concept_b",
            teaching_level=2,
            audience=make_audience(),
            knowledge=make_knowledge(),
            conversation=history,
        )
        assert isinstance(fb, Feedback)

    @pytest.mark.asyncio
    async def test_react_overrides_concept_id(self):
        """LLM 输出的 concept_id 被调用方覆盖。"""
        output_with_wrong_id = VALID_LEARNER_OUTPUT.replace(
            '"concept_id": "concept_b"', '"concept_id": "wrong_id"'
        )
        agent = _make_agent_with_mock([output_with_wrong_id])
        fb, raw = await agent.react(
            teacher_response="Explanation...",
            concept_id="concept_b",
            teaching_level=2,
            audience=make_audience(),
            knowledge=make_knowledge(),
            conversation=[],
        )
        assert fb.concept_id == "concept_b"


# ── 解析退化测试 ──


class TestParseResponse:

    def test_fallback_on_invalid_json(self):
        """解析失败退化为 confused + confidence 0.3。"""
        fb = LearnerAgent._parse_response("This is not JSON at all.", "concept_x")
        assert fb.type == "confused"
        assert fb.concept_id == "concept_x"
        assert fb.confidence == 0.3
        assert len(fb.understanding_summary) > 0

    def test_fallback_on_empty_response(self):
        fb = LearnerAgent._parse_response("", "concept_x")
        assert fb.type == "confused"
        assert fb.confidence == 0.3

    def test_parse_understood_output(self):
        output = """{
  "type": "understood",
  "concept_id": "concept_a",
  "detail": {},
  "confidence": 0.95,
  "understanding_summary": "This concept transforms input data using an orthogonal matrix to achieve uniform distribution across all dimensions."
}
---JSON---
I understand this now! The key insight is that the orthogonal matrix preserves norms while redistributing energy.
"""
        fb = LearnerAgent._parse_response(output, "concept_a")
        assert fb.type == "understood"
        assert fb.confidence == 0.95

    def test_parse_json_in_markdown_fence(self):
        output = """```json
{
  "type": "go_deeper",
  "concept_id": "c",
  "detail": {"deeper_question": "Why not PCA?"},
  "confidence": 0.7,
  "understanding_summary": "I understand the rotation but want to know why not PCA"
}
```

I want to understand why PCA wouldn't work here.
"""
        fb = LearnerAgent._parse_response(output, "c")
        assert fb.type == "go_deeper"
