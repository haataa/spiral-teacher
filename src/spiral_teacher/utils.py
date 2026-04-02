"""仓库扫描和文件处理工具。"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 常量 ──

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".env",
    "dist", "build", ".next", ".nuxt", "target", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "egg-info", ".eggs",
}

SKIP_EXTENSIONS = {
    # 二进制 / 图片
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".pdf", ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".jar",
    # 数据文件
    ".csv", ".tsv", ".parquet", ".arrow", ".h5", ".hdf5",
    ".sqlite", ".db",
    # 锁文件 / 生成文件
    ".lock",
}

SKIP_FILENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "Pipfile.lock", "poetry.lock", "uv.lock",
    ".DS_Store", "Thumbs.db",
}

SOURCE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".rs", ".go", ".java", ".kt", ".scala",
    ".c", ".cpp", ".h", ".hpp",
    ".rb", ".php", ".swift", ".m",
    ".lua", ".zig", ".nim", ".ex", ".exs",
    ".sh", ".bash", ".zsh",
}

CONFIG_FILENAMES = {
    "pyproject.toml", "setup.py", "setup.cfg",
    "package.json", "tsconfig.json",
    "Cargo.toml", "go.mod",
    "Makefile", "CMakeLists.txt",
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env.example",
}

README_PATTERNS = {"readme", "readme.md", "readme.rst", "readme.txt"}

MAX_SINGLE_FILE_BYTES = 50 * 1024  # 50KB
MAX_SINGLE_FILE_LINES = 500


# ── 文件优先级 ──


def _file_priority(
    rel_path: Path,
    topic_keywords: list[str] | None = None,
) -> int:
    """返回文件优先级，数字越小越优先。

    优先级:
      0 - README / 项目配置
      1 - src/lib 下的源码（topic 匹配时从 2 提升）
      2 - 其他源码
      3 - 测试文件
      4 - 其他文档
      5 - 其他文件
    """
    name_lower = rel_path.name.lower()
    parts_lower = [p.lower() for p in rel_path.parts]

    # README
    if name_lower in README_PATTERNS:
        return 0

    # 项目配置
    if rel_path.name in CONFIG_FILENAMES:
        return 0

    suffix = rel_path.suffix.lower()

    # 源码文件
    if suffix in SOURCE_EXTENSIONS:
        # 测试文件
        if any(p in ("test", "tests", "spec", "specs") for p in parts_lower) or \
           name_lower.startswith("test_") or name_lower.endswith("_test.py"):
            base_priority = 3
        # src/lib 下的源码
        elif any(p in ("src", "lib", "pkg", "internal", "cmd") for p in parts_lower):
            base_priority = 1
        else:
            base_priority = 2

        # topic 感知提升
        if topic_keywords:
            path_str = str(rel_path).lower()
            if any(kw in path_str for kw in topic_keywords):
                base_priority = max(0, base_priority - 1)

        return base_priority

    # Markdown / 文档
    if suffix in (".md", ".rst", ".txt", ".adoc"):
        return 4

    return 5


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数：字符数 / 4。"""
    return len(text) // 4


def _truncate_file_content(content: str, file_path: str) -> str:
    """对超大文件截断，保留前 N 行。"""
    lines = content.splitlines(keepends=True)
    if len(lines) <= MAX_SINGLE_FILE_LINES:
        return content

    truncated = "".join(lines[:MAX_SINGLE_FILE_LINES])
    truncated += f"\n[... truncated, total {len(lines)} lines ...]\n"
    logger.info("截断文件 %s: %d -> %d 行", file_path, len(lines), MAX_SINGLE_FILE_LINES)
    return truncated


def _should_skip_dir(name: str) -> bool:
    """判断目录是否应跳过。"""
    if name in SKIP_DIRS:
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _should_skip_file(path: Path) -> bool:
    """判断文件是否应跳过。"""
    if path.name in SKIP_FILENAMES:
        return True
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    return False


# ── 文件树生成 ──


def _build_file_tree(repo_path: Path, files: list[Path]) -> str:
    """从文件列表生成简洁的目录树。"""
    # 收集所有目录和文件的相对路径
    entries: set[str] = set()
    for f in files:
        rel = f.relative_to(repo_path)
        # 添加所有父目录
        for parent in rel.parents:
            if parent != Path("."):
                entries.add(str(parent) + "/")
        entries.add(str(rel))

    sorted_entries = sorted(entries)

    lines = []
    for entry in sorted_entries:
        depth = entry.count("/")
        if entry.endswith("/"):
            depth -= 1
        name = entry.rstrip("/").split("/")[-1]
        if entry.endswith("/"):
            name += "/"
        indent = "    " * depth
        lines.append(f"{indent}{name}")

    return "\n".join(lines)


# ── 主函数 ──


def scan_repository(
    repo_path: str,
    topic: str | None = None,
    max_tokens: int = 60_000,
) -> str:
    """扫描代码仓库，组装为 LLM 可消化的上下文字符串。

    Args:
        repo_path: 仓库根目录路径
        topic: 可选聚焦主题，用于优先保留相关文件
        max_tokens: token 截断阈值（估算值）

    Returns:
        包含文件树 + 文件内容的格式化字符串

    Raises:
        FileNotFoundError: repo_path 不存在
        ValueError: 仓库中没有可读取的文件
    """
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"仓库路径不存在: {repo_path}")
    if not root.is_dir():
        raise FileNotFoundError(f"路径不是目录: {repo_path}")

    # topic 分词
    topic_keywords: list[str] | None = None
    if topic:
        topic_keywords = [w.lower() for w in topic.replace("-", " ").replace("_", " ").split() if len(w) > 1]

    # 收集所有文件
    all_files: list[tuple[int, int, Path]] = []  # (priority, depth, path)

    for path in root.rglob("*"):
        # 跳过特殊目录
        if any(_should_skip_dir(p) for p in path.relative_to(root).parts):
            continue

        if not path.is_file():
            continue

        if _should_skip_file(path):
            continue

        # 跳过超大二进制文件
        try:
            if path.stat().st_size > 1_000_000:  # 1MB
                continue
        except OSError:
            continue

        rel_path = path.relative_to(root)
        priority = _file_priority(rel_path, topic_keywords)
        depth = len(rel_path.parts)
        all_files.append((priority, depth, path))

    if not all_files:
        raise ValueError(f"仓库中没有可读取的文件: {repo_path}")

    # 按优先级排序，同优先级按深度排序（浅层优先）
    all_files.sort(key=lambda x: (x[0], x[1]))

    # 读取文件内容，按优先级累加直到达到 token 阈值
    included_files: list[tuple[Path, str]] = []  # (path, content)
    total_tokens = 0
    skipped_files: list[str] = []
    truncated_files: list[str] = []

    for _priority, _depth, path in all_files:
        try:
            raw = path.read_bytes()
            # 超大单文件检查
            if len(raw) > MAX_SINGLE_FILE_BYTES:
                content = raw.decode("utf-8", errors="replace")
                content = _truncate_file_content(content, str(path.relative_to(root)))
                truncated_files.append(str(path.relative_to(root)))
            else:
                content = raw.decode("utf-8", errors="replace")
        except OSError as e:
            logger.warning("无法读取文件 %s: %s", path, e)
            skipped_files.append(str(path.relative_to(root)))
            continue

        file_tokens = _estimate_tokens(content)

        if total_tokens + file_tokens > max_tokens:
            skipped_files.append(str(path.relative_to(root)))
            continue

        included_files.append((path, content))
        total_tokens += file_tokens

    if not included_files:
        raise ValueError(f"仓库中没有可读取的文件（全部被截断或跳过）: {repo_path}")

    # 组装输出
    included_paths = [p for p, _ in included_files]
    tree = _build_file_tree(root, included_paths)

    sections = [f"## Repository Structure\n\n{tree}\n"]

    for path, content in included_files:
        rel = path.relative_to(root)
        sections.append(f"## File: {rel}\n\n```\n{content}\n```\n")

    result = "\n".join(sections)

    # 日志输出
    logger.info(
        "仓库扫描完成: %d 个文件, 约 %d tokens, %d 个文件被截断, %d 个文件被跳过",
        len(included_files), total_tokens, len(truncated_files), len(skipped_files),
    )
    if truncated_files:
        logger.info("截断的文件: %s", truncated_files)
    if skipped_files and len(skipped_files) <= 20:
        logger.info("跳过的文件: %s", skipped_files)

    return result


# ── JSON 提取 ──


def extract_json_from_text(text: str) -> tuple[dict | None, str]:
    """从 LLM 输出中提取 JSON 对象和剩余 Markdown 文本。

    尝试多种策略：
    1. ---JSON--- 分隔符
    2. ```json 代码块
    3. 文本中第一个 {...} 对象
    4. 全部失败返回 (None, 原文)

    Returns:
        (parsed_dict_or_None, remaining_markdown_text)
    """
    text = text.strip()

    # 策略 1: ---JSON--- 分隔符
    separator = "---JSON---"
    if separator in text:
        json_part, md_part = text.split(separator, 1)
        parsed = _try_parse_json(json_part.strip())
        if parsed is not None:
            return parsed, md_part.strip()

    # 策略 2: ```json 代码块
    json_block_match = re.search(r"```json\s*\n(.*?)```", text, re.DOTALL)
    if json_block_match:
        parsed = _try_parse_json(json_block_match.group(1).strip())
        if parsed is not None:
            # markdown 是代码块之外的文本
            md_part = text[:json_block_match.start()] + text[json_block_match.end():]
            return parsed, md_part.strip()

    # 策略 3: 找文本中第一个 {...} (顶层 JSON 对象)
    first_brace = text.find("{")
    if first_brace != -1:
        json_str = _find_matching_brace(text, first_brace)
        if json_str:
            parsed = _try_parse_json(json_str)
            if parsed is not None:
                md_part = text[:first_brace] + text[first_brace + len(json_str):]
                return parsed, md_part.strip()

    # 全部失败
    return None, text


def _try_parse_json(text: str) -> dict | None:
    """尝试解析 JSON，失败返回 None。"""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _find_matching_brace(text: str, start: int) -> str | None:
    """从 start 位置找到匹配的 } ，返回完整的 JSON 字符串。"""
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None
