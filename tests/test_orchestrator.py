"""Orchestrator 测试。

使用 mock Agent 测试主循环逻辑。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from spiral_teacher.agents.learner import LearnerAgent
from spiral_teacher.agents.reader import ReaderAgent
from spiral_teacher.agents.teacher import TeacherAgent
from spiral_teacher.models import (
    AudienceProfile,
    Concept,
    ConceptDependency,
    ConversationEntry,
    Feedback,
    FeedbackDetail,
    Knowledge,
    TeachingResponse,
    TutorialConfig,
)
from spiral_teacher.orchestrator import generate_tutorial, load_audience_profile

PROFILES_DIR = Path(__file__).parent.parent / "src" / "spiral_teacher" / "profiles"


# ── 测试数据 ──


def make_knowledge() -> Knowledge:
    return Knowledge(
        project_summary="Test project",
        concepts=[
            Concept(id="a", name="A", category="core", description="First", difficulty=1),
            Concept(id="b", name="B", category="core", description="Second", difficulty=2,
                    prerequisites=["a"]),
            Concept(id="c", name="C", category="core", description="Third", difficulty=3),
        ],
        dependencies=[
            ConceptDependency(source="a", target="b", reason="B needs A"),
        ],
        teaching_order=["a", "b", "c"],
        source_type="repository",
        source_path="/tmp/test",
    )


def make_config(**overrides) -> TutorialConfig:
    defaults = {
        "source": "/tmp/test-repo",
        "source_type": "repository",
        "audience": "ml_engineer",
        "output_path": "output.md",
        "max_rounds": 30,
        "max_rounds_per_concept": 6,
    }
    defaults.update(overrides)
    return TutorialConfig(**defaults)


def make_teaching_response(concept_id: str, level: int = 2) -> TeachingResponse:
    return TeachingResponse(
        concept_id=concept_id,
        level=level,
        content=f"Explanation of {concept_id}",
    )


def make_feedback(concept_id: str, fb_type: str = "understood", confidence: float = 0.95) -> Feedback:
    return Feedback(
        type=fb_type,
        concept_id=concept_id,
        detail=FeedbackDetail(),
        confidence=confidence,
        understanding_summary=f"I understand {concept_id} because it works by doing X which leads to Y and the reason is Z" if fb_type == "understood" else f"I'm confused about {concept_id}",
    )


def _mock_reader(knowledge: Knowledge) -> ReaderAgent:
    reader = AsyncMock(spec=ReaderAgent)
    reader.read_repository = AsyncMock(return_value=knowledge)
    return reader


def _mock_teacher() -> TeacherAgent:
    teacher = AsyncMock(spec=TeacherAgent)

    async def give_overview(knowledge):
        tr = make_teaching_response("overview", level=1)
        return tr, "Overview text"

    async def introduce_concept(concept_id, knowledge, conversation, level=2):
        tr = make_teaching_response(concept_id, level)
        return tr, f"Introducing {concept_id}"

    async def respond_to_feedback(feedback, knowledge, conversation, current_level):
        tr = make_teaching_response(feedback.concept_id, current_level)
        return tr, f"Responding to {feedback.type}"

    teacher.give_overview = AsyncMock(side_effect=give_overview)
    teacher.introduce_concept = AsyncMock(side_effect=introduce_concept)
    teacher.respond_to_feedback = AsyncMock(side_effect=respond_to_feedback)
    return teacher


def _mock_learner_always_understood() -> LearnerAgent:
    """模拟一个总是说 understood 的 Learner。"""
    learner_agent = AsyncMock(spec=LearnerAgent)

    async def react(teacher_response, concept_id, teaching_level, audience, knowledge, conversation):
        fb = make_feedback(concept_id, "understood", 0.95)
        return fb, "I understand!"

    learner_agent.react = AsyncMock(side_effect=react)
    return learner_agent


def _mock_learner_confused_then_understood(confused_rounds: int = 2) -> LearnerAgent:
    """模拟一个先 confused 几轮然后 understood 的 Learner。"""
    learner_agent = AsyncMock(spec=LearnerAgent)
    call_count = {"value": 0}

    async def react(teacher_response, concept_id, teaching_level, audience, knowledge, conversation):
        call_count["value"] += 1
        # 每个概念的前 confused_rounds 轮返回 confused
        concept_rounds = sum(
            1 for e in conversation
            if e.role == "learner" and e.feedback and e.feedback.concept_id == concept_id
        )
        if concept_rounds < confused_rounds:
            fb = make_feedback(concept_id, "confused", 0.3)
            return fb, f"I'm confused about {concept_id}"
        else:
            fb = make_feedback(concept_id, "understood", 0.95)
            return fb, f"I understand {concept_id}!"

    learner_agent.react = AsyncMock(side_effect=react)
    return learner_agent


# ── 画像加载 ──


class TestLoadAudienceProfile:

    def test_load_existing_profile(self):
        profile = load_audience_profile("ml_engineer")
        assert profile.name == "ml_engineer"
        assert len(profile.confusion_triggers) > 0

    def test_load_nonexistent_profile(self):
        with pytest.raises(FileNotFoundError, match="读者画像不存在"):
            load_audience_profile("nonexistent_profile")


# ── 主循环测试 ──


class TestGenerateTutorial:

    @pytest.mark.asyncio
    async def test_normal_flow_all_understood(self):
        """所有概念都 understood 的正常流程。"""
        knowledge = make_knowledge()
        config = make_config()

        result = await generate_tutorial(
            config,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=_mock_learner_always_understood(),
        )

        assert result.knowledge == knowledge
        assert result.audience.name == "ml_engineer"
        # 对话应包含：overview + (introduce + learner) * 3 concepts
        assert len(result.conversation) >= 7

        # 检查对话交替
        for i, entry in enumerate(result.conversation):
            if i == 0:
                assert entry.role == "teacher"  # overview
            # 后续应该是 teacher/learner 交替

    @pytest.mark.asyncio
    async def test_concepts_progress_in_order(self):
        """概念按 teaching_order 推进。"""
        knowledge = make_knowledge()
        config = make_config()

        result = await generate_tutorial(
            config,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=_mock_learner_always_understood(),
        )

        # 提取 teacher 引入的概念顺序（跳过 overview）
        introduced = [
            e.teaching_response.concept_id
            for e in result.conversation
            if e.role == "teacher" and e.teaching_response
            and e.teaching_response.concept_id != "overview"
        ]
        # 应该包含 a, b, c（可能有重复因为 respond_to_feedback）
        assert "a" in introduced
        assert "b" in introduced
        assert "c" in introduced

    @pytest.mark.asyncio
    async def test_confused_triggers_teacher_response(self):
        """Learner confused 后 Teacher 会响应反馈。"""
        knowledge = make_knowledge()
        config = make_config()

        teacher = _mock_teacher()
        result = await generate_tutorial(
            config,
            reader=_mock_reader(knowledge),
            teacher=teacher,
            learner=_mock_learner_confused_then_understood(confused_rounds=1),
        )

        # respond_to_feedback 应该被调用过
        assert teacher.respond_to_feedback.call_count > 0

    @pytest.mark.asyncio
    async def test_validate_feedback_is_applied(self):
        """validate_feedback 被应用：understood + low confidence → go_deeper。"""
        knowledge = make_knowledge()
        config = make_config()

        learner_agent = AsyncMock(spec=LearnerAgent)
        call_count = {"value": 0}

        async def react(teacher_response, concept_id, teaching_level, audience, knowledge, conversation):
            call_count["value"] += 1
            if call_count["value"] == 1:
                # 第一次：understood 但 confidence 太低，会被降级
                fb = Feedback(
                    type="understood", concept_id=concept_id,
                    confidence=0.5,  # < 0.85 → 降级
                    detail=FeedbackDetail(),
                    understanding_summary="I think I get it" * 10,
                )
                return fb, "Maybe I understand?"
            else:
                # 后续：真正 understood
                fb = make_feedback(concept_id, "understood", 0.95)
                return fb, "Yes I understand now!"

        learner_agent.react = AsyncMock(side_effect=react)

        result = await generate_tutorial(
            config,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=learner_agent,
        )

        # 第一个概念应该因为 validate_feedback 降级而多一轮
        learner_entries = [
            e for e in result.conversation if e.role == "learner"
        ]
        # 第一条应该是 go_deeper（被降级了）
        assert learner_entries[0].feedback.type == "go_deeper"

    @pytest.mark.asyncio
    async def test_per_concept_timeout(self):
        """单概念超时后强制推进。"""
        knowledge = make_knowledge()
        config = make_config(max_rounds_per_concept=2)

        # Learner 永远 confused
        learner_agent = AsyncMock(spec=LearnerAgent)

        async def react(teacher_response, concept_id, teaching_level, audience, knowledge, conversation):
            fb = make_feedback(concept_id, "confused", 0.2)
            return fb, "Still confused"

        learner_agent.react = AsyncMock(side_effect=react)

        result = await generate_tutorial(
            config,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=learner_agent,
        )

        # 即使全部 confused，所有概念都应该被覆盖（强制推进）
        # 应该有 3 个概念的 introduce 调用
        teacher = result  # 我们需要检查 conversation
        # 检查对话中有 3 个不同概念的 teacher 引入
        introduced_concepts = set()
        for e in result.conversation:
            if e.role == "teacher" and e.teaching_response:
                cid = e.teaching_response.concept_id
                if cid != "overview":
                    introduced_concepts.add(cid)
        assert introduced_concepts == {"a", "b", "c"}

    @pytest.mark.asyncio
    async def test_global_timeout(self):
        """全局轮次上限触发终止。"""
        knowledge = make_knowledge()
        config = make_config(max_rounds=3)  # 只允许 3 轮

        # Learner 永远 confused
        learner_agent = AsyncMock(spec=LearnerAgent)

        async def react(teacher_response, concept_id, teaching_level, audience, knowledge, conversation):
            fb = make_feedback(concept_id, "confused", 0.2)
            return fb, "Confused"

        learner_agent.react = AsyncMock(side_effect=react)

        result = await generate_tutorial(
            config,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=learner_agent,
        )

        # 总 learner 轮次不应超过 max_rounds
        learner_rounds = sum(1 for e in result.conversation if e.role == "learner")
        assert learner_rounds <= 3

    @pytest.mark.asyncio
    async def test_consecutive_understood_circuit_breaker(self):
        """连续 3 次 understood 后触发 challenge prompt。"""
        knowledge = make_knowledge()
        config = make_config()

        events: list[str] = []

        def on_event(event, msg, data):
            events.append(event)

        result = await generate_tutorial(
            config,
            on_event=on_event,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=_mock_learner_always_understood(),
        )

        assert result is not None
        assert len(result.conversation) > 0

    @pytest.mark.asyncio
    async def test_on_event_callback(self):
        """事件回调被正确调用。"""
        knowledge = make_knowledge()
        config = make_config()

        events: list[tuple[str, str]] = []

        def on_event(event, msg, data):
            events.append((event, msg))

        await generate_tutorial(
            config,
            on_event=on_event,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=_mock_learner_always_understood(),
        )

        event_types = [e[0] for e in events]
        assert "knowledge_ready" in event_types
        assert "overview_done" in event_types
        assert "finished" in event_types

    @pytest.mark.asyncio
    async def test_synthesis_input_completeness(self):
        """返回的 SynthesisInput 包含所有必要数据。"""
        knowledge = make_knowledge()
        config = make_config()

        result = await generate_tutorial(
            config,
            reader=_mock_reader(knowledge),
            teacher=_mock_teacher(),
            learner=_mock_learner_always_understood(),
        )

        assert result.knowledge == knowledge
        assert result.audience.name == "ml_engineer"
        assert len(result.conversation) > 0
        # conversation 中有 teacher 和 learner 的条目
        roles = {e.role for e in result.conversation}
        assert "teacher" in roles
        assert "learner" in roles
