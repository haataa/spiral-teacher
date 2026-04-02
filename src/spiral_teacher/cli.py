"""Spiral Teacher CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from spiral_teacher.models import (
    ConversationEntry,
    Knowledge,
    SynthesisInput,
    TutorialConfig,
)
from spiral_teacher.orchestrator import (
    EVENT_CONCEPT_DONE,
    EVENT_CONCEPT_START,
    EVENT_FINISHED,
    EVENT_KNOWLEDGE_READY,
    EVENT_LEARNER_RESPONSE,
    EVENT_OVERVIEW_DONE,
    EVENT_TEACHER_RESPONSE,
    generate_tutorial,
    load_audience_profile,
)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


# ── 渐进式输出 ──


class ProgressWriter:
    """渐进式输出：每个关键节点即时写文件。"""

    def __init__(self, output_dir: Path, clean: bool = True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if clean:
            for f in ["conversation.md", "conversation.json", "summary.txt"]:
                (self.output_dir / f).unlink(missing_ok=True)

    def on_event(self, event: str, msg: str, data: dict[str, Any]):
        if event == EVENT_KNOWLEDGE_READY:
            self._on_knowledge_ready(data["knowledge"])
        elif event == EVENT_OVERVIEW_DONE:
            self._append_conversation(data["entry"])
            print(f">>> {msg}")
        elif event == EVENT_CONCEPT_START:
            print(f"\n{'─'*40}")
            print(f">>> {msg}")
        elif event == EVENT_TEACHER_RESPONSE:
            entry = data.get("entry")
            if entry:
                self._append_conversation(entry)
            print(f"  [T] {msg}")
        elif event == EVENT_LEARNER_RESPONSE:
            entry = data.get("entry")
            if entry:
                self._append_conversation(entry)
                self._print_learner(entry)
        elif event == EVENT_CONCEPT_DONE:
            conversation = data.get("conversation", [])
            self._save_conversation_snapshot(conversation)
            tag = "[OK]" if data.get("result") == "understood" else "[TIMEOUT]"
            print(f"  {tag} {msg}")
        elif event == EVENT_FINISHED:
            result = data.get("result")
            if result:
                self._save_final(result)
            print(f"\n{'='*60}")
            print(f">>> {msg}")
            print(f"{'='*60}")
        else:
            print(f">>> {msg}")

    def _on_knowledge_ready(self, knowledge: Knowledge):
        path = self.output_dir / "knowledge.json"
        path.write_text(knowledge.model_dump_json(indent=2), encoding="utf-8")
        print(f"\n{'='*60}")
        print(f"Knowledge graph saved: {path}")
        print(f"Concepts: {len(knowledge.concepts)}")
        print(f"Teaching order:")
        concepts = {c.id: c for c in knowledge.concepts}
        for i, cid in enumerate(knowledge.teaching_order):
            c = concepts.get(cid)
            name = c.name if c else cid
            diff = f" (difficulty {c.difficulty}, importance {c.importance})" if c else ""
            print(f"  {i+1:2d}. {name}{diff}")
        print(f"{'='*60}\n")

    @staticmethod
    def _print_learner(entry: ConversationEntry):
        fb = entry.feedback
        if not fb:
            return
        icons = {"confused": "?", "go_deeper": ">>", "wrong_assumption": "!",
                 "understood": "OK", "request_example": "eg"}
        icon = icons.get(fb.type, "?")
        print(f"  [{icon}] Learner: {fb.type} (confidence={fb.confidence:.2f})")
        if fb.detail.stuck_point:
            print(f"     stuck: {fb.detail.stuck_point[:120]}")
        if fb.detail.deeper_question:
            print(f"     question: {fb.detail.deeper_question[:120]}")
        if fb.detail.assumption:
            print(f"     assumption: {fb.detail.assumption[:120]}")
        summary = fb.understanding_summary[:150]
        print(f"     summary: {summary}")

    def _append_conversation(self, entry: ConversationEntry):
        path = self.output_dir / "conversation.md"
        role = "**Teacher**" if entry.role == "teacher" else "**Learner**"
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n---\n\n### {role}\n\n")
            if entry.role == "learner" and entry.feedback:
                fb = entry.feedback
                f.write(f"*[{fb.type}, confidence={fb.confidence:.2f}, concept={fb.concept_id}]*\n\n")
            elif entry.role == "teacher" and entry.teaching_response:
                tr = entry.teaching_response
                f.write(f"*[Level {tr.level}, concept={tr.concept_id}]*\n\n")
            f.write(entry.raw_text)
            f.write("\n")

    def _save_conversation_snapshot(self, conversation: list[ConversationEntry]):
        path = self.output_dir / "conversation.json"
        data = [e.model_dump() for e in conversation]
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_final(self, result: SynthesisInput):
        self._save_conversation_snapshot(result.conversation)
        learner_entries = [e for e in result.conversation if e.role == "learner"]
        teacher_entries = [e for e in result.conversation if e.role == "teacher"]
        types = {}
        for e in learner_entries:
            if e.feedback:
                types[e.feedback.type] = types.get(e.feedback.type, 0) + 1

        lines = [
            f"Concepts: {len(result.knowledge.concepts)}",
            f"Messages: {len(result.conversation)}",
            f"  Teacher: {len(teacher_entries)}",
            f"  Learner: {len(learner_entries)}",
            "",
            "Learner feedback:",
        ]
        for t, count in sorted(types.items()):
            lines.append(f"  {t}: {count}")

        (self.output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


# ── generate 命令 ──


async def _cmd_generate(args):
    output_dir = Path(args.output)

    config = TutorialConfig(
        source=args.repo,
        source_type="repository",
        topic=args.topic,
        audience=args.audience,
        output_path=str(output_dir / "tutorial.md"),
        language=args.language,
        max_rounds=args.max_rounds,
        max_rounds_per_concept=args.max_rounds_per_concept,
    )

    print(f"=== Spiral Teacher ===")
    print(f"Repo:     {args.repo}")
    print(f"Topic:    {args.topic or '(all)'}")
    print(f"Audience: {args.audience}")
    print(f"Language: {args.language}")
    print(f"Rounds:   {args.max_rounds} (per concept: {args.max_rounds_per_concept})")

    # 检查是否 resume（在清理文件之前！）
    resume_args = _check_resume(output_dir, args)
    writer = ProgressWriter(output_dir, clean=resume_args is None)
    if resume_args:
        result = await _run_with_resume(config, writer, resume_args)
    else:
        result = await generate_tutorial(config, on_event=writer.on_event)

    # 合成教程
    if not args.no_synthesize:
        print(f"\nSynthesizing tutorial...")
        from spiral_teacher.agents.synthesizer import SynthesizerAgent
        synthesizer = SynthesizerAgent()
        tutorial = await synthesizer.compile(result, language=args.language)
        tutorial_path = output_dir / "tutorial.md"
        tutorial_path.write_text(tutorial, encoding="utf-8")
        print(f"Tutorial saved: {tutorial_path} ({len(tutorial)} chars)")


def _check_resume(output_dir: Path, args) -> dict | None:
    """检查是否可以从上次的结果继续。"""
    kg_path = output_dir / "knowledge.json"
    conv_path = output_dir / "conversation.json"

    if not args.resume:
        return None

    if not kg_path.exists() or not conv_path.exists():
        print("No previous data found, starting fresh.")
        return None

    kg_data = json.loads(kg_path.read_text(encoding="utf-8"))
    knowledge = Knowledge.model_validate(kg_data)

    conv_data = json.loads(conv_path.read_text(encoding="utf-8"))
    conversation = [ConversationEntry.model_validate(e) for e in conv_data]

    # 找出已覆盖的概念
    covered = set()
    for e in conversation:
        if e.feedback and e.feedback.type == "understood":
            covered.add(e.feedback.concept_id)

    remaining = [cid for cid in knowledge.teaching_order if cid not in covered]
    print(f"\nResuming: {len(covered)} concepts covered, {len(remaining)} remaining")

    return {
        "knowledge": knowledge,
        "conversation": conversation,
        "covered": covered,
        "remaining": remaining,
    }


async def _run_with_resume(config: TutorialConfig, writer: ProgressWriter, resume: dict) -> SynthesisInput:
    """从上次停下的地方继续。"""
    from spiral_teacher.agents.learner import LearnerAgent, validate_feedback
    from spiral_teacher.agents.teacher import TeacherAgent

    lang = config.language
    teacher = TeacherAgent(language=lang)
    learner = LearnerAgent(language=lang)
    audience = load_audience_profile(config.audience)
    knowledge = resume["knowledge"]
    conversation = list(resume["conversation"])
    covered = set(resume["covered"])
    total_rounds = 0
    consecutive_understood = 0

    for concept_idx, concept_id in enumerate(resume["remaining"]):
        if total_rounds >= config.max_rounds:
            print(f">>> Reached max rounds ({config.max_rounds}), stopping")
            break

        print(f"\n{'─'*40}")
        print(f">>> Concept [{len(covered)+1}/{len(knowledge.concepts)}]: {concept_id}")

        tr, raw = await teacher.introduce_concept(
            concept_id, knowledge, conversation, level=2,
        )
        teacher_entry = ConversationEntry(
            role="teacher", teaching_response=tr, raw_text=raw,
        )
        conversation.append(teacher_entry)
        writer._append_conversation(teacher_entry)
        print(f"  [T] Teacher introduced {concept_id} (Level {tr.level})")

        current_level = tr.level
        rounds_on_current = 0

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
            writer._append_conversation(learner_entry)
            writer._print_learner(learner_entry)

            if fb.type == "understood":
                covered.add(concept_id)
                consecutive_understood += 1
                print(f"  [OK] {concept_id} understood")
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
            writer._append_conversation(teacher_entry)
            current_level = tr.level
            print(f"  [T] Teacher responded (Level {tr.level})")
        else:
            if concept_id not in covered:
                covered.add(concept_id)
                consecutive_understood = 0
                print(f"  [TIMEOUT] {concept_id} forced advance")

    writer._save_conversation_snapshot(conversation)
    print(f"\n{'='*60}")
    print(f"Done: {len(conversation)} messages, {len(covered)}/{len(knowledge.concepts)} concepts covered")
    print(f"{'='*60}")

    return SynthesisInput(
        conversation=conversation,
        knowledge=knowledge,
        audience=audience,
    )


# ── synthesize 命令 ──


async def _cmd_synthesize(args):
    output_dir = Path(args.output)
    kg_path = output_dir / "knowledge.json"
    conv_path = output_dir / "conversation.json"

    if not kg_path.exists():
        print(f"Error: {kg_path} not found. Run 'generate' first.", file=sys.stderr)
        sys.exit(1)
    if not conv_path.exists():
        print(f"Error: {conv_path} not found. Run 'generate' first.", file=sys.stderr)
        sys.exit(1)

    print("Loading data...")
    knowledge = Knowledge.model_validate_json(kg_path.read_text(encoding="utf-8"))
    conv_data = json.loads(conv_path.read_text(encoding="utf-8"))
    conversation = [ConversationEntry.model_validate(e) for e in conv_data]
    audience = load_audience_profile(args.audience)

    print(f"  Knowledge: {len(knowledge.concepts)} concepts")
    print(f"  Conversation: {len(conversation)} entries")
    print(f"  Language: {args.language}")

    si = SynthesisInput(conversation=conversation, knowledge=knowledge, audience=audience)

    from spiral_teacher.agents.synthesizer import SynthesizerAgent
    synthesizer = SynthesizerAgent()
    tutorial = await synthesizer.compile(si, language=args.language)

    tutorial_path = output_dir / "tutorial.md"
    tutorial_path.write_text(tutorial, encoding="utf-8")
    print(f"\nTutorial saved: {tutorial_path} ({len(tutorial)} chars)")


# ── Main ──


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(
        prog="spiral-teacher",
        description="Multi-agent adaptive teaching content generator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate
    gen = subparsers.add_parser("generate", help="Generate tutorial from a code repository")
    gen.add_argument("--repo", required=True, help="Path to the code repository")
    gen.add_argument("--topic", default=None, help="Focus topic (optional)")
    gen.add_argument("--audience", default="ml_engineer", help="Audience profile key (default: ml_engineer)")
    gen.add_argument("--language", default="zh", help="Output language (default: zh)")
    gen.add_argument("--output", default="output", help="Output directory (default: output)")
    gen.add_argument("--max-rounds", type=int, default=20, help="Max total dialogue rounds (default: 20)")
    gen.add_argument("--max-rounds-per-concept", type=int, default=4, help="Max rounds per concept (default: 4)")
    gen.add_argument("--resume", action="store_true", help="Resume from previous run")
    gen.add_argument("--no-synthesize", action="store_true", help="Skip tutorial synthesis step")

    # synthesize
    syn = subparsers.add_parser("synthesize", help="Synthesize tutorial from existing conversation data")
    syn.add_argument("--output", default="output", help="Output directory containing conversation.json and knowledge.json")
    syn.add_argument("--audience", default="ml_engineer", help="Audience profile key")
    syn.add_argument("--language", default="zh", help="Output language (default: zh)")

    args = parser.parse_args()

    if args.command == "generate":
        asyncio.run(_cmd_generate(args))
    elif args.command == "synthesize":
        asyncio.run(_cmd_synthesize(args))


if __name__ == "__main__":
    main()
