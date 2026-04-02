"""Reader Agent: 从代码仓库提取结构化知识图谱。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic
from pydantic import ValidationError

from spiral_teacher.models import Knowledge
from spiral_teacher.utils import scan_repository

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "reader.md"


class ReaderError(Exception):
    """Reader Agent 错误。"""

    def __init__(self, message: str, raw_response: str | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.cause = cause


class ReaderAgent:
    """从代码仓库提取结构化知识图谱。

    使用 Anthropic API 直接调用（非 Agent SDK），因为 Reader 是单次调用。
    """

    def __init__(
        self,
        model: str = "claude-opus-4-6",
        client: anthropic.AsyncAnthropic | None = None,
    ):
        self.model = model
        self.client = client or anthropic.AsyncAnthropic()
        self._system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    async def read_repository(
        self,
        repo_path: str,
        topic: str | None = None,
        max_input_tokens: int = 60_000,
    ) -> Knowledge:
        """扫描代码仓库，提取结构化知识图谱。

        Args:
            repo_path: 本地代码仓库路径
            topic: 可选，聚焦主题
            max_input_tokens: 仓库内容截断阈值

        Returns:
            Knowledge 模型实例

        Raises:
            FileNotFoundError: repo_path 不存在
            ValueError: 仓库中没有可读取的源码文件
            ReaderError: LLM 调用失败或多次重试后仍无法解析
        """
        # 阶段 1：仓库扫描
        repo_context = scan_repository(repo_path, topic=topic, max_tokens=max_input_tokens)

        # 阶段 2：知识提取
        user_message = self._build_user_message(repo_context, topic)
        knowledge = await self._extract_with_retry(user_message, repo_path)
        return knowledge

    def _build_user_message(self, repo_context: str, topic: str | None) -> str:
        """构建发送给 LLM 的用户消息。"""
        parts = ["Below is the content of a code repository. Analyze it and extract a structured knowledge graph.\n"]

        if topic:
            parts.append(f"Focus on the topic: **{topic}**. Still cover prerequisite concepts needed to understand this topic.\n")

        parts.append(repo_context)
        return "\n".join(parts)

    async def _extract_with_retry(
        self,
        user_message: str,
        repo_path: str,
        max_attempts: int = 3,
    ) -> Knowledge:
        """调用 LLM 提取知识，失败时重试。"""
        messages: list[dict] = [{"role": "user", "content": user_message}]
        last_raw_response: str | None = None
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            try:
                raw_response = await self._call_llm(messages)
                last_raw_response = raw_response

                knowledge = self._parse_response(raw_response, repo_path)
                if attempt > 0:
                    logger.info("第 %d 次尝试成功解析", attempt + 1)
                return knowledge

            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                logger.warning(
                    "第 %d/%d 次尝试解析失败: %s",
                    attempt + 1, max_attempts, e,
                )

                if attempt < max_attempts - 1:
                    # 将错误信息追加到对话中，让模型修正
                    messages.append({"role": "assistant", "content": raw_response})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Your JSON output failed validation:\n\n{e}\n\n"
                            "Please fix the issues and output the corrected JSON. "
                            "Output ONLY the JSON object, nothing else."
                        ),
                    })

            except anthropic.APIError as e:
                last_error = e
                logger.error("API 调用错误: %s", e)
                # 认证错误不重试
                if isinstance(e, anthropic.AuthenticationError):
                    raise ReaderError(
                        f"API 认证失败: {e}",
                        cause=e,
                    ) from e
                # 其他 API 错误（rate limit 等），anthropic SDK 自带重试
                raise ReaderError(
                    f"API 调用失败: {e}",
                    raw_response=last_raw_response,
                    cause=e,
                ) from e

        raise ReaderError(
            f"{max_attempts} 次尝试后仍无法解析 LLM 输出",
            raw_response=last_raw_response,
            cause=last_error,
        )

    async def _call_llm(self, messages: list[dict]) -> str:
        """调用 Anthropic API。"""
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=16384,
            system=self._system_prompt,
            messages=messages,
            timeout=300.0,
        )

        # 提取文本内容
        text_parts = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        return "\n".join(text_parts)

    def _parse_response(self, raw_response: str, repo_path: str) -> Knowledge:
        """解析 LLM 返回的 JSON 为 Knowledge 模型。"""
        # 尝试提取 JSON（LLM 可能在 JSON 前后加了文本）
        json_str = self._extract_json(raw_response)
        data = json.loads(json_str)

        # 补充代码层面硬编码的字段
        data["source_type"] = "repository"
        data["source_path"] = repo_path

        return Knowledge.model_validate(data)

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 输出中提取 JSON 对象。

        处理 LLM 可能在 JSON 前后添加文本或 markdown 代码块的情况。
        """
        text = text.strip()

        # 如果被 markdown 代码块包裹
        if text.startswith("```"):
            # 找到第一个换行后开始
            first_newline = text.index("\n")
            # 找到最后一个 ```
            last_fence = text.rfind("```")
            if last_fence > first_newline:
                text = text[first_newline + 1:last_fence].strip()

        # 如果整体就是 JSON
        if text.startswith("{"):
            return text

        # 尝试找到第一个 { 和最后一个 }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            return text[first_brace:last_brace + 1]

        return text
