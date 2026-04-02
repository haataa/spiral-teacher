"""从已有对话数据合成教学文档。"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spiral_teacher.agents.synthesizer import SynthesizerAgent
from spiral_teacher.models import (
    AudienceProfile,
    ConversationEntry,
    Knowledge,
    SynthesisInput,
)
from spiral_teacher.orchestrator import load_audience_profile

OUTPUT_DIR = Path("output")


async def main():
    print("Loading data...")

    # 加载知识图谱
    kg_data = json.loads((OUTPUT_DIR / "knowledge.json").read_text(encoding="utf-8"))
    knowledge = Knowledge.model_validate(kg_data)
    print(f"  Knowledge: {len(knowledge.concepts)} concepts")

    # 加载对话历史
    conv_data = json.loads((OUTPUT_DIR / "conversation.json").read_text(encoding="utf-8"))
    conversation = [ConversationEntry.model_validate(e) for e in conv_data]
    print(f"  Conversation: {len(conversation)} entries")

    # 加载画像
    audience = load_audience_profile("ml_engineer")
    print(f"  Audience: {audience.display_name}")

    si = SynthesisInput(
        conversation=conversation,
        knowledge=knowledge,
        audience=audience,
    )

    language = sys.argv[1] if len(sys.argv) > 1 else "zh"
    print(f"\nSynthesizing tutorial (language={language}, this may take a few minutes)...")
    synthesizer = SynthesizerAgent()
    tutorial = await synthesizer.compile(si, language=language)

    output_path = OUTPUT_DIR / "tutorial.md"
    output_path.write_text(tutorial, encoding="utf-8")
    print(f"\nTutorial saved: {output_path} ({len(tutorial)} chars)")


if __name__ == "__main__":
    asyncio.run(main())
