"""端到端测试脚本：渐进式输出，每个阶段即时写文件。"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spiral_teacher.models import TutorialConfig, ConversationEntry, Knowledge
from spiral_teacher.orchestrator import (
    generate_tutorial,
    EVENT_KNOWLEDGE_READY,
    EVENT_OVERVIEW_DONE,
    EVENT_CONCEPT_START,
    EVENT_TEACHER_RESPONSE,
    EVENT_LEARNER_RESPONSE,
    EVENT_CONCEPT_DONE,
    EVENT_FINISHED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def on_event(event: str, msg: str, data: dict[str, Any]):
    """渐进式输出：每个关键节点即时写文件。"""

    # ── 知识图谱提取完成：立即写文件 ──
    if event == EVENT_KNOWLEDGE_READY:
        knowledge: Knowledge = data["knowledge"]
        kg_path = OUTPUT_DIR / "knowledge.json"
        kg_path.write_text(knowledge.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"知识图谱已保存: {kg_path}")
        print(f"概念数: {len(knowledge.concepts)}")
        print(f"教学顺序:")
        for i, cid in enumerate(knowledge.teaching_order):
            concept = next((c for c in knowledge.concepts if c.id == cid), None)
            name = concept.name if concept else cid
            diff = f" (difficulty {concept.difficulty})" if concept else ""
            print(f"  {i+1:2d}. {name}{diff}")
        print(f"{'='*60}\n")
        return

    # ── 鸟瞰概述完成：写到对话文件 ──
    if event == EVENT_OVERVIEW_DONE:
        entry: ConversationEntry = data["entry"]
        _append_to_conversation_log(entry)
        print(f">>> {msg}")
        print(f"    {entry.raw_text[:200]}...")
        return

    # ── 新概念开始 ──
    if event == EVENT_CONCEPT_START:
        print(f"\n{'─'*40}")
        print(f">>> {msg}")
        return

    # ── Teacher 响应 ──
    if event == EVENT_TEACHER_RESPONSE:
        entry = data.get("entry")
        if entry:
            _append_to_conversation_log(entry)
        print(f"  [T] {msg}")
        return

    # ── Learner 响应 ──
    if event == EVENT_LEARNER_RESPONSE:
        entry: ConversationEntry = data.get("entry")
        if entry:
            _append_to_conversation_log(entry)
        if entry and entry.feedback:
            fb = entry.feedback
            icon = {"confused": "?", "go_deeper": ">>", "wrong_assumption": "!",
                    "understood": "OK", "request_example": "eg"}.get(fb.type, "?")
            print(f"  [{icon}] Learner: {fb.type} (confidence={fb.confidence:.2f})")
            if fb.detail.stuck_point:
                print(f"     stuck: {fb.detail.stuck_point[:120]}")
            if fb.detail.deeper_question:
                print(f"     question: {fb.detail.deeper_question[:120]}")
            if fb.detail.assumption:
                print(f"     assumption: {fb.detail.assumption[:120]}")
            print(f"     summary: {fb.understanding_summary[:150]}")
        return

    # ── 概念完成：保存当前对话快照 ──
    if event == EVENT_CONCEPT_DONE:
        conversation = data.get("conversation", [])
        _save_conversation_snapshot(conversation)
        result_icon = "[OK]" if data.get("result") == "understood" else "[TIMEOUT]"
        print(f"  {result_icon} {msg}")
        return

    # ── 全部完成 ──
    if event == EVENT_FINISHED:
        result = data.get("result")
        if result:
            _save_final_output(result)
        print(f"\n{'='*60}")
        print(f">>> {msg}")
        print(f"{'='*60}")
        return

    # ── 其他事件 ──
    print(f">>> {msg}")


def _append_to_conversation_log(entry: ConversationEntry):
    """追加一条对话到 Markdown 日志文件（实时可读）。"""
    log_path = OUTPUT_DIR / "conversation.md"
    role = "**Teacher**" if entry.role == "teacher" else "**Learner**"

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n---\n\n### {role}\n\n")
        if entry.role == "learner" and entry.feedback:
            fb = entry.feedback
            f.write(f"*[{fb.type}, confidence={fb.confidence:.2f}, concept={fb.concept_id}]*\n\n")
        elif entry.role == "teacher" and entry.teaching_response:
            tr = entry.teaching_response
            f.write(f"*[Level {tr.level}, concept={tr.concept_id}]*\n\n")
        f.write(entry.raw_text)
        f.write("\n")


def _save_conversation_snapshot(conversation: list[ConversationEntry]):
    """保存当前对话的 JSON 快照。"""
    snap_path = OUTPUT_DIR / "conversation.json"
    data = [e.model_dump() for e in conversation]
    snap_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_final_output(result):
    """保存最终输出。"""
    # JSON 快照
    _save_conversation_snapshot(result.conversation)

    # 统计摘要
    summary_path = OUTPUT_DIR / "summary.txt"
    learner_entries = [e for e in result.conversation if e.role == "learner"]
    teacher_entries = [e for e in result.conversation if e.role == "teacher"]

    understood = [e for e in learner_entries if e.feedback and e.feedback.type == "understood"]
    confused = [e for e in learner_entries if e.feedback and e.feedback.type == "confused"]
    go_deeper = [e for e in learner_entries if e.feedback and e.feedback.type == "go_deeper"]
    wrong_assumption = [e for e in learner_entries if e.feedback and e.feedback.type == "wrong_assumption"]
    request_example = [e for e in learner_entries if e.feedback and e.feedback.type == "request_example"]

    lines = [
        f"概念数: {len(result.knowledge.concepts)}",
        f"对话总条数: {len(result.conversation)}",
        f"  Teacher: {len(teacher_entries)}",
        f"  Learner: {len(learner_entries)}",
        f"",
        f"Learner 反馈分布:",
        f"  understood: {len(understood)}",
        f"  confused: {len(confused)}",
        f"  go_deeper: {len(go_deeper)}",
        f"  wrong_assumption: {len(wrong_assumption)}",
        f"  request_example: {len(request_example)}",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")


async def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "D:/workspace/agent-world-model"
    topic = sys.argv[2] if len(sys.argv) > 2 else None

    config = TutorialConfig(
        source=repo_path,
        source_type="repository",
        topic=topic,
        audience="ml_engineer",
        output_path="output/tutorial.md",
        max_rounds=20,
        max_rounds_per_concept=4,
    )

    print(f"=== Spiral Teacher E2E ===")
    print(f"Repo: {repo_path}")
    print(f"Topic: {topic or '(all)'}")
    print(f"Max rounds: {config.max_rounds}")

    # 清理上次输出
    for f in OUTPUT_DIR.glob("conversation.*"):
        f.unlink()

    await generate_tutorial(config, on_event=on_event)


if __name__ == "__main__":
    asyncio.run(main())
