"""YamlSkillLoader - 解析 YAML frontmatter 格式的 .md Skill 文件

使用 YAML frontmatter（--- ... ---）格式。
支持 browser-act 等外部 Skill 使用此格式。

文件格式示例：
```markdown
---
name: browser-act
description: Browser automation CLI for AI agents
version: "2.0.2"
author: BrowserAct
---

# browser-act

正文内容作为指令...
```
"""

import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import SkillParameter
from .skill_loader import SkillLoader


class YamlSkillLoader(SkillLoader):
    """YAML frontmatter 格式 Markdown Skill 文件加载器

    检测文件是否以 --- ... --- 包围的 YAML frontmatter 开头，
    若是则使用此加载器解析。
    """

    FRONTMATTER_PATTERN = re.compile(
        r"^---\s*\n(.*?)\n---\s*(?:\n|$)",
        re.DOTALL | re.MULTILINE,
    )

    @classmethod
    def is_frontmatter_format(cls, path: str | Path) -> bool:
        """判断文件是否采用 YAML frontmatter 格式"""
        path = Path(path)
        try:
            content = path.read_text(encoding="utf-8")
            return cls.FRONTMATTER_PATTERN.match(content) is not None
        except Exception:
            return False

    @classmethod
    def load(cls, path: str | Path) -> Optional[Dict[str, Any]]:
        """加载并解析一个 YAML frontmatter 格式的 .md Skill 文件

        Args:
            path: .md 文件路径

        Returns:
            解析结果字典：
            {
                "name": str,
                "description": str,
                "version": str,
                "author": str | None,
                "parameters": list[SkillParameter],
                "tags": list[str],
                "instructions": str,   # frontmatter 之后的 Markdown 正文作为指令
                "file_path": str,
            }
            解析失败返回 None
        """
        path = Path(path)
        if not path.exists() or path.suffix.lower() not in (".md", ".MD"):
            return None

        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        fm_match = cls.FRONTMATTER_PATTERN.match(content)
        if not fm_match:
            return None  # 非 frontmatter 格式，不处理

        try:
            fm = yaml.safe_load(fm_match.group(1))
        except yaml.YAMLError:
            return None

        if not isinstance(fm, dict):
            return None

        # 支持嵌套在 metadata 下的字段（如 browser-act 的 metadata.author）
        metadata = fm.get("metadata", {}) if isinstance(fm.get("metadata"), dict) else {}

        def _get(key: str, default: Any = "") -> Any:
            return fm.get(key) or metadata.get(key) or default

        # 提取元数据
        name = str(_get("name", ""))
        if not name:
            # fallback：尝试从 frontmatter 的 title 字段或文件名
            name = str(_get("title", "")) or path.stem.lower().replace("-", "_")

        description = str(_get("description", ""))
        version = str(_get("version", "0.1.0"))
        author_val = _get("author")
        author = str(author_val) if author_val else None

        # frontmatter 之后的正文作为指令
        body = content[fm_match.end():].strip()
        if not body:
            return None  # 缺少正文指令

        # 清理 Markdown 标题（# Skill: xxx）但保留其余内容作为指令
        instructions = cls._clean_markdown(body)

        if not instructions:
            return None

        # parameters 和 tags 暂不支持（外部 Skill 通常不提供这些字段）
        return {
            "name": name,
            "description": description or f"Skill: {name}",
            "version": version,
            "author": author,
            "parameters": [],
            "tags": [],
            "instructions": instructions.strip(),
            "file_path": str(path),
        }

    @classmethod
    def _clean_markdown(cls, text: str) -> str:
        """清理 Markdown 格式，保留正文结构"""
        lines = text.splitlines()
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # 跳过顶层 # 标题（# Skill: xxx 是 Skill 名字）
            if stripped.startswith("# "):
                continue
            # ## 区块标题保留，只清理标题标记（去掉 ### 前缀）
            if stripped.startswith("## ") or stripped.startswith("### "):
                cleaned.append(stripped)
                continue
            cleaned.append(line)
        result = "\n".join(cleaned).strip()
        # 清理残留的单个 # 标题标记
        result = re.sub(r"^#\s+", "", result, flags=re.MULTILINE)
        return result

    @classmethod
    def scan_directory(cls, directory: str | Path) -> List[Dict[str, Any]]:
        """扫描目录下所有 YAML frontmatter 格式的 .md 文件"""
        directory = Path(directory)
        if not directory.is_dir():
            return []

        results = []
        for md_file in directory.glob("*.md"):
            if cls.is_frontmatter_format(md_file):
                skill_data = cls.load(md_file)
                if skill_data:
                    results.append(skill_data)
        return results
