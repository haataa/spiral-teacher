"""models.py 合约验证测试。

验证 docs/contracts/models.md 中的 5 条成功标准。
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from spiral_teacher.models import (
    AudienceProfile,
    CodeReference,
    Concept,
    ConceptDependency,
    ConversationEntry,
    Feedback,
    FeedbackDetail,
    Knowledge,
    SynthesisInput,
    TeachingResponse,
    TutorialConfig,
)

PROFILES_DIR = Path(__file__).parent.parent / "src" / "spiral_teacher" / "profiles"


# ── 测试数据工厂 ──


def make_concept(**overrides) -> Concept:
    defaults = {
        "id": "test_concept",
        "name": "Test Concept",
        "category": "core_algorithm",
        "description": "A test concept",
        "difficulty": 3,
    }
    defaults.update(overrides)
    return Concept(**defaults)


def make_knowledge(**overrides) -> Knowledge:
    c1 = make_concept(id="concept_a", name="Concept A", difficulty=1)
    c2 = make_concept(id="concept_b", name="Concept B", difficulty=2)
    defaults = {
        "project_summary": "A test project",
        "concepts": [c1, c2],
        "dependencies": [
            ConceptDependency(
                source="concept_a",
                target="concept_b",
                reason="B depends on A",
            )
        ],
        "teaching_order": ["concept_a", "concept_b"],
        "source_type": "repository",
        "source_path": "/tmp/test-repo",
    }
    defaults.update(overrides)
    return Knowledge(**defaults)


def make_feedback(**overrides) -> Feedback:
    defaults = {
        "type": "confused",
        "concept_id": "test_concept",
        "detail": FeedbackDetail(stuck_point="I don't understand step 2"),
        "confidence": 0.3,
        "understanding_summary": "I get step 1 but not step 2",
    }
    defaults.update(overrides)
    return Feedback(**defaults)


def make_teaching_response(**overrides) -> TeachingResponse:
    defaults = {
        "concept_id": "test_concept",
        "level": 2,
        "content": "## Explanation\n\nHere is how it works...",
    }
    defaults.update(overrides)
    return TeachingResponse(**defaults)


# ── 标准 1: JSON round-trip ──


class TestRoundTrip:
    """所有模型可序列化为 JSON 并反序列化回来。"""

    def test_concept_round_trip(self):
        original = make_concept(
            source_files=["src/rotation.py"],
            key_equations=["y = H @ x"],
            common_misconceptions=["It's the same as PCA"],
        )
        json_str = original.model_dump_json()
        restored = Concept.model_validate_json(json_str)
        assert restored == original

    def test_knowledge_round_trip(self):
        original = make_knowledge()
        json_str = original.model_dump_json()
        restored = Knowledge.model_validate_json(json_str)
        assert restored == original

    def test_feedback_round_trip(self):
        original = make_feedback()
        json_str = original.model_dump_json()
        restored = Feedback.model_validate_json(json_str)
        assert restored == original

    def test_teaching_response_round_trip(self):
        original = make_teaching_response(
            analogies_used=["Like sorting cards"],
            code_references=[
                CodeReference(
                    file_path="src/rotation.py",
                    start_line=10,
                    end_line=20,
                    snippet="def rotate(x): ...",
                    explanation="The rotation function",
                )
            ],
        )
        json_str = original.model_dump_json()
        restored = TeachingResponse.model_validate_json(json_str)
        assert restored == original

    def test_conversation_entry_round_trip_teacher(self):
        original = ConversationEntry(
            role="teacher",
            teaching_response=make_teaching_response(),
            raw_text="Here is how it works...",
        )
        json_str = original.model_dump_json()
        restored = ConversationEntry.model_validate_json(json_str)
        assert restored == original

    def test_conversation_entry_round_trip_learner(self):
        original = ConversationEntry(
            role="learner",
            feedback=make_feedback(),
            raw_text="I don't understand step 2",
        )
        json_str = original.model_dump_json()
        restored = ConversationEntry.model_validate_json(json_str)
        assert restored == original

    def test_tutorial_config_round_trip(self):
        original = TutorialConfig(
            source="/tmp/repo",
            source_type="repository",
            audience="ml_engineer",
            output_path="output.md",
        )
        json_str = original.model_dump_json()
        restored = TutorialConfig.model_validate_json(json_str)
        assert restored == original

    def test_synthesis_input_round_trip(self):
        entry = ConversationEntry(
            role="teacher",
            teaching_response=make_teaching_response(),
            raw_text="Explanation text",
        )
        profile = AudienceProfile(
            name="ml_engineer",
            display_name="ML 工程师",
            math_level="本科数学",
            coding_level="Python 熟练",
            domain_knowledge="熟悉 Transformer",
            confusion_triggers=["理论与工程的 gap"],
        )
        original = SynthesisInput(
            conversation=[entry],
            knowledge=make_knowledge(),
            audience=profile,
        )
        json_str = original.model_dump_json()
        restored = SynthesisInput.model_validate_json(json_str)
        assert restored == original


# ── 标准 2: 硬校验规则 ──


class TestValidation:
    """验证规则通过 Pydantic validator 实现，非法输入抛出 ValidationError。"""

    # Concept.difficulty 范围
    def test_concept_difficulty_too_low(self):
        with pytest.raises(ValidationError):
            make_concept(difficulty=0)

    def test_concept_difficulty_too_high(self):
        with pytest.raises(ValidationError):
            make_concept(difficulty=6)

    # Feedback.confidence 范围
    def test_feedback_confidence_too_low(self):
        with pytest.raises(ValidationError):
            make_feedback(confidence=-0.1)

    def test_feedback_confidence_too_high(self):
        with pytest.raises(ValidationError):
            make_feedback(confidence=1.1)

    # TeachingResponse.level 范围
    def test_teaching_level_too_low(self):
        with pytest.raises(ValidationError):
            make_teaching_response(level=-1)

    def test_teaching_level_too_high(self):
        with pytest.raises(ValidationError):
            make_teaching_response(level=6)

    # Knowledge: teaching_order 引用不存在的概念
    def test_knowledge_invalid_teaching_order(self):
        with pytest.raises(ValidationError, match="不存在的概念"):
            make_knowledge(teaching_order=["concept_a", "nonexistent"])

    # Knowledge: dependency 引用不存在的概念
    def test_knowledge_invalid_dependency_source(self):
        with pytest.raises(ValidationError, match="不存在的概念"):
            make_knowledge(
                dependencies=[
                    ConceptDependency(
                        source="nonexistent",
                        target="concept_a",
                        reason="test",
                    )
                ]
            )

    def test_knowledge_invalid_dependency_target(self):
        with pytest.raises(ValidationError, match="不存在的概念"):
            make_knowledge(
                dependencies=[
                    ConceptDependency(
                        source="concept_a",
                        target="nonexistent",
                        reason="test",
                    )
                ]
            )

    # ConversationEntry: role 与内容不匹配
    def test_conversation_entry_teacher_with_feedback(self):
        with pytest.raises(ValidationError):
            ConversationEntry(
                role="teacher",
                feedback=make_feedback(),
                raw_text="text",
            )

    def test_conversation_entry_learner_with_teaching(self):
        with pytest.raises(ValidationError):
            ConversationEntry(
                role="learner",
                teaching_response=make_teaching_response(),
                raw_text="text",
            )

    def test_conversation_entry_both_filled(self):
        with pytest.raises(ValidationError):
            ConversationEntry(
                role="teacher",
                teaching_response=make_teaching_response(),
                feedback=make_feedback(),
                raw_text="text",
            )

    def test_conversation_entry_neither_filled(self):
        with pytest.raises(ValidationError):
            ConversationEntry(
                role="teacher",
                raw_text="text",
            )


# ── 标准 3: 从 PLAN.md 示例 JSON 构造模型 ──


class TestPlanExamples:
    """可从 PLAN.md 中的示例 JSON 构造出对应模型实例。

    注：PLAN.md 中 from/to 需映射为 source/target，topic 需映射为 concept_id。
    """

    def test_knowledge_from_plan_example(self):
        """对应 PLAN.md 中 Reader 输出示例。"""
        k = Knowledge(
            project_summary="一句话描述项目做什么",
            concepts=[
                Concept(
                    id="wht_rotation",
                    name="Walsh-Hadamard 旋转",
                    category="core_algorithm",
                    description="通过正交变换使 KV 向量各维度分布均匀化",
                    prerequisites=["linear_algebra_basics", "orthogonal_matrix"],
                    difficulty=3,
                    source_files=["turboquant/rotation.py"],
                    key_equations=["y = D2 @ H @ D1 @ x"],
                    related_concepts=["polar_quant", "fwht"],
                ),
                Concept(
                    id="polar_quant",
                    name="Polar Quantization",
                    category="core_algorithm",
                    description="极坐标量化",
                    difficulty=4,
                ),
                # 补充 teaching_order 中引用的其他概念（最小定义）
                Concept(id="kv_cache_problem", name="KV Cache Problem", category="core_algorithm", description="KV 缓存内存问题", difficulty=1),
                Concept(id="distribution_problem", name="Distribution Problem", category="math_foundation", description="分布不均匀问题", difficulty=2),
                Concept(id="kv_asymmetry", name="KV Asymmetry", category="core_algorithm", description="K/V 不对称性", difficulty=3),
                Concept(id="sparse_v", name="Sparse V", category="engineering_detail", description="稀疏 V 缓存", difficulty=3),
            ],
            dependencies=[
                ConceptDependency(
                    source="wht_rotation",
                    target="polar_quant",
                    reason="PolarQuant 依赖旋转后的高斯分布假设",
                )
            ],
            teaching_order=[
                "kv_cache_problem",
                "distribution_problem",
                "wht_rotation",
                "polar_quant",
                "kv_asymmetry",
                "sparse_v",
            ],
            source_type="repository",
            source_path="./turboquant",
        )
        assert len(k.concepts) == 6
        assert k.teaching_order[0] == "kv_cache_problem"

    def test_feedback_from_plan_example(self):
        """对应 PLAN.md 中 Learner 输出示例（topic → concept_id）。"""
        f = Feedback(
            type="confused",
            concept_id="wht_rotation",
            detail=FeedbackDetail(
                stuck_point="为什么 Hadamard 矩阵能让分布变均匀",
            ),
            confidence=0.7,
            understanding_summary="我目前的理解：旋转是为了让分布变均匀，但我不确定为什么 Hadamard 矩阵能做到这一点",
        )
        assert f.type == "confused"
        assert f.concept_id == "wht_rotation"


# ── 标准 4: YAML 加载 ──


class TestYamlLoading:
    """AudienceProfile 可从 YAML 文件加载。"""

    def test_load_ml_engineer_profile(self):
        yaml_path = PROFILES_DIR / "ml_engineer.yaml"
        assert yaml_path.exists(), f"Profile not found: {yaml_path}"

        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        profile = AudienceProfile(**data)
        assert profile.name == "ml_engineer"
        assert profile.display_name == "ML 工程师"
        assert len(profile.confusion_triggers) > 0


# ── 标准 5: ConversationEntry 序列化 ──


class TestConversationEntrySerialization:
    """ConversationEntry 的 round-trip 序列化正确。"""

    def test_teacher_entry_preserves_type(self):
        entry = ConversationEntry(
            role="teacher",
            teaching_response=make_teaching_response(),
            raw_text="explanation",
        )
        json_str = entry.model_dump_json()
        restored = ConversationEntry.model_validate_json(json_str)

        assert restored.role == "teacher"
        assert restored.teaching_response is not None
        assert restored.feedback is None
        assert restored.teaching_response.concept_id == "test_concept"

    def test_learner_entry_preserves_type(self):
        entry = ConversationEntry(
            role="learner",
            feedback=make_feedback(),
            raw_text="question",
        )
        json_str = entry.model_dump_json()
        restored = ConversationEntry.model_validate_json(json_str)

        assert restored.role == "learner"
        assert restored.feedback is not None
        assert restored.teaching_response is None
        assert restored.feedback.type == "confused"

    def test_list_of_entries_round_trip(self):
        """模拟完整对话历史的序列化。"""
        entries = [
            ConversationEntry(
                role="teacher",
                teaching_response=make_teaching_response(),
                raw_text="Let me explain...",
            ),
            ConversationEntry(
                role="learner",
                feedback=make_feedback(),
                raw_text="I'm confused about...",
            ),
            ConversationEntry(
                role="teacher",
                teaching_response=make_teaching_response(level=3),
                raw_text="Let me go deeper...",
            ),
        ]
        # 通过 SynthesisInput 间接验证列表序列化
        si = SynthesisInput(
            conversation=entries,
            knowledge=make_knowledge(),
            audience=AudienceProfile(
                name="test",
                display_name="Test",
                math_level="basic",
                coding_level="basic",
                domain_knowledge="none",
                confusion_triggers=["everything"],
            ),
        )
        json_str = si.model_dump_json()
        restored = SynthesisInput.model_validate_json(json_str)
        assert len(restored.conversation) == 3
        assert restored.conversation[0].role == "teacher"
        assert restored.conversation[1].role == "learner"


# ── 软校验行为测试 ──


class TestSoftValidation:
    """验证不做硬校验的场景确实不会抛异常。"""

    def test_feedback_detail_mismatch_type_and_field(self):
        """type=confused 但填了 deeper_question 而非 stuck_point，不应报错。"""
        f = make_feedback(
            type="confused",
            detail=FeedbackDetail(deeper_question="Why does this work?"),
        )
        assert f.type == "confused"
        assert f.detail.stuck_point is None

    def test_understood_with_low_confidence(self):
        """type=understood 但 confidence=0.8，不应报错。"""
        f = make_feedback(type="understood", confidence=0.8)
        assert f.type == "understood"

    def test_knowledge_topological_violation_warns(self, caplog):
        """teaching_order 违反拓扑排序时只 warning 不报错。"""
        with caplog.at_level("WARNING"):
            # concept_b depends on concept_a, but order is reversed
            k = make_knowledge(teaching_order=["concept_b", "concept_a"])
            assert k is not None
        assert "出现在" in caplog.text or len(caplog.records) > 0
