"""Orchestrator: 编排 Reader → Teacher ↔ Learner 的完整对话流程。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from spiral_teacher.agents.learner import LearnerAgent, validate_feedback
from spiral_teacher.agents.reader import ReaderAgent
from spiral_teacher.agents.teacher import TeacherAgent
from spiral_teacher.models import (
    AudienceProfile,
    ConversationEntry,
    Knowledge,
    SynthesisInput,
    TutorialConfig,
)

logger = logging.getLogger(__name__)

PROFILES_DIR = Path(__file__).parent / "profiles"

# 事件类型
EVENT_KNOWLEDGE_READY = "knowledge_ready"
EVENT_OVERVIEW_DONE = "overview_done"
EVENT_CONCEPT_START = "concept_start"
EVENT_TEACHER_RESPONSE = "teacher_response"
EVENT_LEARNER_RESPONSE = "learner_response"
EVENT_CONCEPT_DONE = "concept_done"
EVENT_FINISHED = "finished"

OnEventCallback = Callable[[str, str, dict[str, Any]], None]
"""回调签名: (event_type, message, data)"""

CHALLENGE_PROMPT = (
    "The learner has been understanding concepts quickly. "
    "For this concept, dig deeper into potential misconceptions and edge cases. "
    "Don't accept surface-level understanding."
)


def load_audience_profile(audience_key: str) -> AudienceProfile:
    """从 YAML 文件加载读者画像。"""
    yaml_path = PROFILES_DIR / f"{audience_key}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"读者画像不存在: {yaml_path}. "
            f"可用画像: {[p.stem for p in PROFILES_DIR.glob('*.yaml')]}"
        )

    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return AudienceProfile(**data)


async def generate_tutorial(
    config: TutorialConfig,
    on_event: OnEventCallback | None = None,
    reader: ReaderAgent | None = None,
    teacher: TeacherAgent | None = None,
    learner: LearnerAgent | None = None,
) -> SynthesisInput:
    """运行完整的教学对话生成流程。"""
    lang = config.language
    reader = reader or ReaderAgent()
    teacher = teacher or TeacherAgent(language=lang)
    learner = learner or LearnerAgent(language=lang)

    def _emit(event: str, msg: str, **data):
        logger.info(msg)
        if on_event:
            on_event(event, msg, data)

    # ── Phase 1: 初始化 ──

    _emit("init", "加载读者画像...")
    audience = load_audience_profile(config.audience)

    _emit("init", f"扫描仓库: {config.source}")
    knowledge = await reader.read_repository(
        config.source,
        topic=config.topic,
    )
    _emit(EVENT_KNOWLEDGE_READY, f"知识图谱提取完成: {len(knowledge.concepts)} 个概念",
          knowledge=knowledge)

    # ── Phase 2: 对话主循环 ──

    # 按 importance 过滤概念
    concepts_map = {c.id: c for c in knowledge.concepts}
    teaching_order = [
        cid for cid in knowledge.teaching_order
        if concepts_map.get(cid) and concepts_map[cid].importance >= config.min_importance
    ]
    skipped = len(knowledge.teaching_order) - len(teaching_order)
    if skipped:
        _emit("init", f"跳过 {skipped} 个低重要性概念 (importance < {config.min_importance})")

    conversation: list[ConversationEntry] = []
    covered_concepts: set[str] = set()
    consecutive_understood = 0
    total_rounds = 0

    # 鸟瞰概述
    _emit("init", "生成项目鸟瞰概述...")
    tr, raw = await teacher.give_overview(knowledge)
    overview_entry = ConversationEntry(
        role="teacher", teaching_response=tr, raw_text=raw,
    )
    conversation.append(overview_entry)
    _emit(EVENT_OVERVIEW_DONE, "鸟瞰概述完成", entry=overview_entry)

    # 逐概念对话
    for concept_idx, concept_id in enumerate(teaching_order):
        if total_rounds >= config.max_rounds:
            _emit("timeout", f"达到全局轮次上限 ({config.max_rounds})，终止")
            break

        _emit(EVENT_CONCEPT_START, f"概念 [{concept_idx + 1}/{len(teaching_order)}]: {concept_id}",
              concept_id=concept_id, concept_idx=concept_idx)

        # 引入概念
        challenge = CHALLENGE_PROMPT if consecutive_understood >= 3 else None
        tr, raw = await teacher.introduce_concept(
            concept_id, knowledge, conversation, level=2,
        )
        if challenge:
            _emit("circuit_breaker", "连续 understood 熔断触发")
            raw = raw + "\n\n" + challenge

        teacher_entry = ConversationEntry(
            role="teacher", teaching_response=tr, raw_text=raw,
        )
        conversation.append(teacher_entry)
        _emit(EVENT_TEACHER_RESPONSE, f"Teacher 引入 {concept_id} (Level {tr.level})",
              entry=teacher_entry)

        current_level = tr.level
        rounds_on_current = 0

        # 当前概念的对话循环
        while rounds_on_current < config.max_rounds_per_concept:
            if total_rounds >= config.max_rounds:
                break

            total_rounds += 1
            rounds_on_current += 1

            fb, learner_raw = await learner.react(
                teacher_response=raw,
                concept_id=concept_id,
                teaching_level=current_level,
                audience=audience,
                knowledge=knowledge,
                conversation=conversation,
            )
            fb = validate_feedback(fb)

            learner_entry = ConversationEntry(
                role="learner", feedback=fb, raw_text=learner_raw,
            )
            conversation.append(learner_entry)
            _emit(EVENT_LEARNER_RESPONSE,
                  f"Learner: {fb.type} (confidence={fb.confidence:.2f})",
                  entry=learner_entry)

            if fb.type == "understood":
                covered_concepts.add(concept_id)
                consecutive_understood += 1
                _emit(EVENT_CONCEPT_DONE,
                      f"概念 {concept_id} 已理解 (连续 understood: {consecutive_understood})",
                      concept_id=concept_id, result="understood",
                      conversation=conversation)
                break
            else:
                consecutive_understood = 0

            tr, raw = await teacher.respond_to_feedback(
                fb, knowledge, conversation, current_level,
            )
            teacher_entry = ConversationEntry(
                role="teacher", teaching_response=tr, raw_text=raw,
            )
            conversation.append(teacher_entry)
            current_level = tr.level
            _emit(EVENT_TEACHER_RESPONSE, f"Teacher 响应 (Level {tr.level})",
                  entry=teacher_entry)

        else:
            if concept_id not in covered_concepts:
                covered_concepts.add(concept_id)
                consecutive_understood = 0
                _emit(EVENT_CONCEPT_DONE,
                      f"概念 {concept_id} 达到轮次上限，强制推进",
                      concept_id=concept_id, result="timeout",
                      conversation=conversation)

    # ── Phase 3: 组装输出 ──

    result = SynthesisInput(
        conversation=conversation,
        knowledge=knowledge,
        audience=audience,
    )

    _emit(EVENT_FINISHED,
          f"对话完成: {len(conversation)} 条消息, "
          f"{len(covered_concepts)}/{len(knowledge.concepts)} 概念覆盖",
          result=result)

    return result
