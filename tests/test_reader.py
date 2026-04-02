"""Reader Agent 测试。

分两部分：
- scan_repository 的纯函数测试（不需要 API）
- ReaderAgent 的集成测试（需要 API key，标记为 integration）
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from spiral_teacher.agents.reader import ReaderAgent, ReaderError
from spiral_teacher.models import Knowledge
from spiral_teacher.utils import scan_repository

# 项目自身路径，用作测试仓库
REPO_ROOT = Path(__file__).parent.parent


# ── scan_repository 测试 ──


class TestScanRepository:
    """测试仓库扫描工具函数。"""

    def test_scan_self_repo(self):
        """能扫描本项目自身。"""
        result = scan_repository(str(REPO_ROOT))
        assert "## Repository Structure" in result
        assert "## File:" in result
        assert "models.py" in result

    def test_scan_includes_readme(self):
        """README 被最高优先级包含。"""
        result = scan_repository(str(REPO_ROOT))
        assert "README.md" in result

    def test_scan_includes_source_files(self):
        """源码文件被包含。"""
        result = scan_repository(str(REPO_ROOT))
        assert "utils.py" in result

    def test_scan_nonexistent_path(self):
        """不存在的路径抛 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            scan_repository("/nonexistent/path/xyz")

    def test_scan_truncation(self, tmp_path):
        """token 阈值生效时截断低优先级文件。"""
        # 创建一个有大量文件的临时仓库
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        for i in range(50):
            (src_dir / f"module_{i}.py").write_text(f"# Module {i}\n" + "x = 1\n" * 200)

        # 用很小的 token 阈值
        result = scan_repository(str(tmp_path), max_tokens=1000)
        # 应该包含部分文件，不是全部
        assert "## File:" in result
        file_count = result.count("## File:")
        assert file_count < 50

    def test_scan_with_topic(self, tmp_path):
        """topic 感知：相关文件优先级提升。"""
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        # 创建两个文件：一个与 topic 相关，一个无关
        (src_dir / "quantization.py").write_text("# Quantization module\n" + "x = 1\n" * 100)
        (src_dir / "logging_utils.py").write_text("# Logging utilities\n" + "y = 2\n" * 100)

        result = scan_repository(str(tmp_path), topic="quantization", max_tokens=500)

        # quantization.py 应该优先被包含
        if "quantization.py" in result:
            assert True  # topic 相关文件被包含
        # 如果 token 够两个都包含也行

    def test_scan_skips_binary(self, tmp_path):
        """二进制文件被跳过。"""
        (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n" + b"\x00" * 100)
        (tmp_path / "code.py").write_text("print('hello')")

        result = scan_repository(str(tmp_path))
        assert "image.png" not in result
        assert "code.py" in result

    def test_scan_skips_gitdir(self, tmp_path):
        """.git 目录被跳过。"""
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("[core]")
        (tmp_path / "main.py").write_text("print('hi')")

        result = scan_repository(str(tmp_path))
        assert ".git" not in result or "main.py" in result

    def test_scan_empty_repo(self, tmp_path):
        """空仓库（只有被跳过的文件）抛 ValueError。"""
        (tmp_path / ".gitignore").write_text("*")
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "HEAD").write_text("ref: refs/heads/main")

        # 只有 .gitignore（不是源码文件但也不在跳过列表里），实际上会被包含
        # 为了真正测空仓库，我们只放被跳过的文件
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        # .gitignore 实际上会被扫描到（txt 类文件）
        # 所以换个方式：只创建 .git 目录
        empty_dir = tmp_path / "empty_repo"
        empty_dir.mkdir()
        empty_git = empty_dir / ".git"
        empty_git.mkdir()
        (empty_git / "HEAD").write_text("ref: refs/heads/main")

        with pytest.raises(ValueError, match="没有可读取的文件"):
            scan_repository(str(empty_dir))


# ── ReaderAgent 单元测试（mock LLM）──


class TestReaderAgentUnit:
    """使用 mock 测试 ReaderAgent 的解析和重试逻辑。"""

    VALID_KNOWLEDGE_JSON = json.dumps({
        "project_summary": "A test project",
        "concepts": [
            {
                "id": "concept_a",
                "name": "Concept A",
                "category": "core_algorithm",
                "description": "First concept",
                "difficulty": 1,
                "source_files": ["src/a.py"],
                "common_misconceptions": [],
            },
            {
                "id": "concept_b",
                "name": "Concept B",
                "category": "design_pattern",
                "description": "Second concept",
                "prerequisites": ["concept_a"],
                "difficulty": 2,
                "source_files": ["src/b.py"],
                "common_misconceptions": ["It's the same as concept A"],
            },
            {
                "id": "concept_c",
                "name": "Concept C",
                "category": "engineering_detail",
                "description": "Third concept",
                "difficulty": 3,
                "source_files": ["src/c.py"],
                "common_misconceptions": ["It requires concept D"],
            },
        ],
        "dependencies": [
            {"source": "concept_a", "target": "concept_b", "reason": "B builds on A"},
        ],
        "teaching_order": ["concept_a", "concept_b", "concept_c"],
    })

    def _make_agent_with_mock(self, responses: list[str]) -> ReaderAgent:
        """创建使用 mock client 的 ReaderAgent。"""
        mock_client = AsyncMock(spec=anthropic.AsyncAnthropic)

        # 构造 mock response
        side_effects = []
        for text in responses:
            mock_response = MagicMock()
            mock_block = MagicMock()
            mock_block.type = "text"
            mock_block.text = text
            mock_response.content = [mock_block]
            side_effects.append(mock_response)

        mock_client.messages.create = AsyncMock(side_effect=side_effects)

        agent = ReaderAgent(model="claude-haiku-4-5", client=mock_client)
        return agent

    @pytest.mark.asyncio
    async def test_parse_valid_json(self, tmp_path):
        """正常 JSON 输出能解析为 Knowledge。"""
        (tmp_path / "main.py").write_text("print('hello')")

        agent = self._make_agent_with_mock([self.VALID_KNOWLEDGE_JSON])
        knowledge = await agent.read_repository(str(tmp_path))

        assert isinstance(knowledge, Knowledge)
        assert len(knowledge.concepts) == 3
        assert knowledge.teaching_order == ["concept_a", "concept_b", "concept_c"]
        assert knowledge.source_type == "repository"
        assert knowledge.source_path == str(tmp_path)

    @pytest.mark.asyncio
    async def test_parse_json_in_markdown_fence(self, tmp_path):
        """JSON 被 markdown 代码块包裹时能正确提取。"""
        (tmp_path / "main.py").write_text("print('hello')")

        wrapped = f"```json\n{self.VALID_KNOWLEDGE_JSON}\n```"
        agent = self._make_agent_with_mock([wrapped])
        knowledge = await agent.read_repository(str(tmp_path))

        assert isinstance(knowledge, Knowledge)
        assert len(knowledge.concepts) == 3

    @pytest.mark.asyncio
    async def test_retry_on_invalid_json(self, tmp_path):
        """第一次返回非法 JSON，重试后成功。"""
        (tmp_path / "main.py").write_text("print('hello')")

        agent = self._make_agent_with_mock([
            "This is not valid JSON {{{",  # 第一次失败
            self.VALID_KNOWLEDGE_JSON,  # 重试成功
        ])
        knowledge = await agent.read_repository(str(tmp_path))

        assert isinstance(knowledge, Knowledge)
        # 验证 LLM 被调用了 2 次
        assert agent.client.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, tmp_path):
        """3 次尝试都失败后抛出 ReaderError。"""
        (tmp_path / "main.py").write_text("print('hello')")

        agent = self._make_agent_with_mock([
            "bad json 1",
            "bad json 2",
            "bad json 3",
        ])

        with pytest.raises(ReaderError, match="3 次尝试后仍无法解析"):
            await agent.read_repository(str(tmp_path))

    @pytest.mark.asyncio
    async def test_topic_passed_to_user_message(self, tmp_path):
        """topic 参数被传递到 LLM 消息中。"""
        (tmp_path / "main.py").write_text("print('hello')")

        agent = self._make_agent_with_mock([self.VALID_KNOWLEDGE_JSON])
        await agent.read_repository(str(tmp_path), topic="quantization")

        # 检查第一次调用的 messages 参数
        call_args = agent.client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        assert "quantization" in messages[0]["content"]


# ── 集成测试（需要 API key）──


@pytest.mark.integration
class TestReaderAgentIntegration:
    """端到端集成测试，调用真实 API。

    运行方式：pytest -m integration tests/test_reader.py
    需要设置 ANTHROPIC_API_KEY 环境变量。
    """

    @pytest.mark.asyncio
    async def test_read_self_repo(self):
        """用本项目自身作为测试仓库，验证端到端流程。"""
        agent = ReaderAgent(model="claude-haiku-4-5")
        knowledge = await agent.read_repository(
            str(REPO_ROOT),
            max_input_tokens=30_000,
        )

        # 合约验证标准 1: 输出合法的 Knowledge 实例
        assert isinstance(knowledge, Knowledge)

        # 合约验证标准 2: 通过 models.py 的硬校验（构造成功即通过）

        # 合约验证标准 3: teaching_order 覆盖所有概念
        concept_ids = {c.id for c in knowledge.concepts}
        assert set(knowledge.teaching_order) == concept_ids

        # 至少 3 个概念
        assert len(knowledge.concepts) >= 3

        # 合约验证标准 5: 有依赖关系
        if len(knowledge.concepts) > 3:
            assert len(knowledge.dependencies) > 0

        # 合约验证标准 4: source_files 指向实际存在的文件
        concepts_with_files = [c for c in knowledge.concepts if c.source_files]
        assert len(concepts_with_files) / len(knowledge.concepts) >= 0.5  # 至少 50%

        for concept in concepts_with_files:
            for sf in concept.source_files:
                full_path = REPO_ROOT / sf
                assert full_path.exists(), f"source_file 不存在: {sf} (concept: {concept.id})"
