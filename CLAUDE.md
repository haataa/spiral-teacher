# Spiral Teacher - Development Guide

## Project Overview

Multi-agent system that generates teaching tutorials from code repositories. Simulates a Teacher-Learner dialogue, then synthesizes the conversation into a structured tutorial document.

## Architecture

```
Reader (Opus) → Knowledge Graph
                    ↓
Orchestrator: Teacher (Opus) ↔ Learner (Sonnet) loop
                    ↓
Synthesizer (Opus) → Tutorial Markdown
```

- **Reader** (`agents/reader.py`): Scans repo, extracts structured knowledge via LLM. Single API call.
- **Teacher** (`agents/teacher.py`): Generates multi-level explanations. Level 0-5 system. Level switching logic in `compute_next_level()` is deterministic code, not LLM.
- **Learner** (`agents/learner.py`): Simulates learner feedback. Has `validate_feedback()` hard checks (confidence/type consistency, summary length) to prevent "going easy" on Teacher.
- **Orchestrator** (`orchestrator.py`): Runs the dialogue loop. Manages concept progression, timeouts, circuit breaker for consecutive `understood`.
- **Synthesizer** (`agents/synthesizer.py`): Compiles dialogue history into coherent tutorial document.
- **CLI** (`cli.py`): Entry point. `spiral-teacher generate` and `spiral-teacher synthesize` commands. Supports `--resume`.

## Key Design Decisions

1. **Importance-first teaching order** (not difficulty-first). Reader prompt ranks concepts by importance to project understanding. CLI architecture goes last, core algorithms first. Concepts below `min_importance` threshold (default 3) are skipped entirely.

2. **Triple-layer anti-leniency for Learner**:
   - Prompt layer: confusion_triggers checklist (Step 4), first encounter rule (Step 6), example requirement (Step 7), forced wrong_assumption for hard concepts, confidence anchors
   - Code layer (stateless): `validate_feedback()` — understood + confidence < 0.85 → go_deeper; understood + short summary → go_deeper; confused + high confidence → correct
   - Code layer (stateful): `validate_concept_feedback()` — hard concept first-round understood → go_deeper; hard concept without prior request_example → request_example

3. **Graceful degradation for parsing**: Teacher/Learner outputs may not follow JSON+separator format. `utils.extract_json_from_text()` tries 3 strategies (separator → ```json block → brute-force { } matching). If all fail, Teacher degrades to raw text as content, Learner degrades to "confused".

4. **File-based communication**: All agent outputs are structured Pydantic models. Conversation history saved incrementally to `output/conversation.json` and `output/conversation.md`.

5. **Chinese by default**: `TutorialConfig.language` defaults to "zh". Teacher/Learner/Synthesizer all receive language instructions.

## Running

```bash
# Install
uv sync --extra dev

# Generate tutorial
spiral-teacher generate --repo /path/to/repo

# Resume from previous run
spiral-teacher generate --repo /path/to/repo --resume --max-rounds 10

# Re-synthesize from existing conversation data
spiral-teacher synthesize --output output --language zh

# Run tests (127 unit tests)
uv run pytest tests/ -m "not integration"
```

## Development Approach

Following the "Harness Design" methodology (from Anthropic's engineering blog):
- **Contract-first**: Each agent has a contract in `docs/contracts/` defining interface, validation criteria, and "don't do" list. Write contract → agent review → implement → test.
- **Read traces, tune prompts**: After E2E runs, read `output/conversation.md` to find judgment biases, then update prompts in `src/spiral_teacher/prompts/`.
- **Pressure-test assumptions**: Each harness component encodes assumptions about model limitations. Re-evaluate when models improve.

## File Layout

```
src/spiral_teacher/
├── cli.py              # CLI entry point (generate, synthesize, --resume)
├── orchestrator.py     # Main dialogue loop
├── models.py           # Pydantic data models (Knowledge, Feedback, TeachingResponse, etc.)
├── utils.py            # Repo scanning, JSON extraction
├── agents/
│   ├── reader.py       # ReaderAgent (knowledge extraction)
│   ├── teacher.py      # TeacherAgent (multi-level explanation) + compute_next_level()
│   ├── learner.py      # LearnerAgent (simulated learner) + validate_feedback()
│   └── synthesizer.py  # SynthesizerAgent (dialogue → tutorial)
├── prompts/            # System prompts for each agent (editable .md files)
│   ├── reader.md
│   ├── teacher.md
│   ├── learner.md
│   └── synthesizer.md
└── profiles/           # Audience profiles (YAML)
    └── ml_engineer.yaml
docs/contracts/         # Agent contracts (interface specs)
tests/                  # 127 unit tests (mock LLM, no API needed)
scripts/                # Legacy E2E scripts (prefer CLI instead)
```

## Current Status (Phase 2 in progress)

All core agents implemented and tested. Learner prompt tuned with stateful concept-level validation. E2E validated on `agent-world-model`, `Eureka`, and `RF-Agent` repos.
See PLAN.md for Phase 2/3 roadmap.

### Known issues
- Learner (Sonnet) never produces `confused` feedback — prefers `go_deeper` instead. May need further prompt tuning or a code-level rule.
- Per-concept depth (~5 rounds) vs coverage trade-off: 20 rounds covers only ~4 concepts. Consider increasing `max_rounds` for repos with many concepts.

## Cost Estimate

~$5-8 per full tutorial (20-40 rounds). Reader ~$1, Teacher ~$3-5, Learner ~$0.50, Synthesizer ~$1.
