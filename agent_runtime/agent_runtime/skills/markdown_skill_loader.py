"""BlockMarkdownSkillLoader - 解析 ## 区块格式的 .md Skill 文件

使用 ## 元信息 / ## 指令 区块结构，不支持 YAML frontmatter。

文件格式示例：
```markdown
# Skill: office-docx

## 元信息
name: office-docx
description: 创建 Word 文档
version: 0.1.0
author: Travel-Assistant

## 参数规范
- title: string, 必填, 文档标题
- content: string, 必填, 文档正文

## 指令
你是一个 Word 文档创建助手。当用户要求创建文档时：
1. 使用 python-docx 库创建 Document 对象
2. 添加标题和段落
3. 保存到用户指定路径

## 使用场景
当用户说"帮我创建一份报告"时调用此 Skill。
```
"""

import re
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import SkillMetadata, SkillParameter
from .skill_loader import SkillLoader


class MarkdownSkillLoader(SkillLoader):
    """## 区块格式 Markdown Skill 文件加载器

    支持 ## 元信息 / ## 指令 区块格式。
    不支持 YAML frontmatter 格式（--- ... ---）。
    """

    # 提取元信息块（存在于 ## 元信息 和下一个 ## 之间）
    META_PATTERN = re.compile(
        r"(?:^|\n)(?:##\s*元信息|##\s*Meta|##\s*Metadata)(.*?)(?=\n##\s|\n#\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # 提取指令块（存在于 ## 指令 和下一个 ## 之间）
    INSTRUCTION_PATTERN = re.compile(
        r"(?:^|\n)(?:##\s*指令|##\s*Instruction)(.*?)(?=\n##\s|\n#\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # 提取参数块（存在于 ## 参数规范 和下一个 ## 之间）
    PARAM_PATTERN = re.compile(
        r"(?:^|\n)(?:##\s*参数规范|##\s*Parameters?|##\s*Params)(.*?)(?=\n##\s|\n#\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # 提取标签块
    TAG_PATTERN = re.compile(
        r"(?:^|\n)(?:##\s*标签|##\s*Tags?)(.*?)(?=\n##\s|\n#\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # 提取 skill 名字（# Skill: xxx）
    NAME_PATTERN = re.compile(r"^#\s*Skill:\s*(\S+)", re.IGNORECASE | re.MULTILINE)

    @classmethod
    def load(cls, path: str | Path) -> Optional[Dict[str, Any]]:
        """加载并解析一个 .md Skill 文件

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
                "instructions": str,
                "file_path": str,
            }
            解析失败返回 None
        """
        path = Path(path)
        if not path.exists() or path.suffix.lower() != ".md":
            return None

        try:
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None

        # 提取 name（优先从元信息 YAML 中取，其次从标题取）
        name = cls._extract_name_from_meta(content) or cls._extract_name_from_title(content)
        if not name:
            # 用文件名作为 fallback
            name = path.stem.lower().replace("-", "_")

        description = cls._extract_field(content, "description") or ""
        version = cls._extract_field(content, "version") or "0.1.0"
        author = cls._extract_field(content, "author")
        instructions = cls._extract_instructions(content)
        parameters = cls._extract_parameters(content)
        tags = cls._extract_tags(content)

        if not instructions:
            return None  # 缺少指令，视为无效 Skill

        return {
            "name": name,
            "description": description or f"Markdown Skill: {name}",
            "version": version,
            "author": author,
            "parameters": parameters,
            "tags": tags,
            "instructions": instructions.strip(),
            "file_path": str(path),
        }

    @classmethod
    def _extract_name_from_title(cls, content: str) -> Optional[str]:
        m = cls.NAME_PATTERN.search(content)
        return m.group(1).strip() if m else None

    @classmethod
    def _extract_name_from_meta(cls, content: str) -> Optional[str]:
        meta = cls._extract_yaml_block(content, "元信息", "Meta", "Metadata")
        if meta and "name" in meta:
            return str(meta["name"])
        return None

    @classmethod
    def _extract_field(cls, content: str, field: str) -> Optional[str]:
        """从 YAML 元信息块中提取单个字段"""
        meta = cls._extract_yaml_block(content, "元信息", "Meta", "Metadata")
        if meta and field in meta:
            val = meta[field]
            return str(val) if val else None
        return None

    @classmethod
    def _extract_yaml_block(cls, content: str, *section_names: str) -> Optional[Dict[str, Any]]:
        """提取并解析 YAML 元信息块"""
        for name in section_names:
            pattern = re.compile(
                rf"(?:^|\n)(?:##\s*{re.escape(name)}.*?)(.*?)(?=\n##\s|\n#\s|\Z)",
                re.DOTALL | re.IGNORECASE,
            )
            m = pattern.search(content)
            if m:
                yaml_text = m.group(1).strip()
                # 简单的 YAML 解析：支持 key: value 和 key: | 格式
                result = {}
                for line in yaml_text.split("\n"):
                    line = line.rstrip()
                    if ":|" in line:
                        # 多行值
                        key, val = line.split(":", 1)
                        result[key.strip()] = val.strip()
                    elif re.match(r"^\s*[\w\-\_]+:\s*", line):
                        key, val = line.split(":", 1)
                        result[key.strip()] = val.strip()
                if result:
                    return result
        return None

    @classmethod
    def _extract_instructions(cls, content: str) -> Optional[str]:
        """提取指令块"""
        for section in ["指令", "Instruction"]:
            m = cls.INSTRUCTION_PATTERN.search(content)
            if m:
                text = m.group(1).strip()
                # 清理 Markdown 格式（移除 ### 标记等）
                text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
                return text
        return None

    @classmethod
    def _extract_parameters(cls, content: str) -> List[SkillParameter]:
        """提取参数规范"""
        params = []
        seen_names: set[str] = set()  # 去重，防止同一参数被多次解析

        # 按各 section 名称逐一搜索，各自独立匹配
        for section in ["参数规范", "Parameters", "Params"]:
            pattern = re.compile(
                rf"(?:^|\n)(?:##\s*{re.escape(section)}.*?)(.*?)(?=\n##\s|\n#\s|\Z)",
                re.DOTALL,
            )
            m = pattern.search(content)
            if not m:
                continue

            block = m.group(1)
            lines = block.strip().split("\n")
            for line in lines:
                line = line.strip().strip("|").strip()
                if not line or line.startswith("---") or line.startswith("| name"):
                    continue

                # 表格格式：| name | type | required | description |
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        name = parts[0]
                        type_desc = parts[1]
                        required = "必填" in parts[2] or "required" in parts[2].lower()
                        desc = parts[3] if len(parts) > 3 else ""
                        if name and name != "name" and name not in seen_names:
                            seen_names.add(name)
                            params.append(SkillParameter(
                                name=name,
                                type=type_desc.lower(),
                                description=desc,
                                required=required,
                            ))

                # 列表格式：- name: type, 必填, 描述
                elif line.startswith("-") and ":" in line:
                    m2 = re.match(r"-\s*(\w+):\s*(\w+),?\s*(必填|可选|required|optional)?,?\s*(.*)", line)
                    if m2:
                        name, ptype, req, desc = m2.groups()
                        if name not in seen_names:
                            seen_names.add(name)
                            params.append(SkillParameter(
                                name=name,
                                type=ptype.lower(),
                                description=desc or "",
                                required=("必填" in (req or "") or "required" in (req or "").lower()),
                            ))
        return params

    @classmethod
    def _extract_tags(cls, content: str) -> List[str]:
        """提取标签"""
        for section in ["标签", "Tags"]:
            m = cls.TAG_PATTERN.search(content)
            if m:
                block = m.group(1)
                tags = []
                for line in block.strip().split("\n"):
                    line = line.strip().strip(",").strip()
                    if line.startswith("-"):
                        tags.append(line.lstrip("-").strip())
                    elif line and not line.startswith("#"):
                        tags.append(line)
                return [t for t in tags if t]
        return []

    @classmethod
    def scan_directory(cls, directory: str | Path) -> List[Dict[str, Any]]:
        """扫描目录下所有 .md 文件，返回解析结果列表"""
        directory = Path(directory)
        if not directory.is_dir():
            return []

        results = []
        for md_file in directory.glob("*.md"):
            skill_data = cls.load(md_file)
            if skill_data:
                results.append(skill_data)
        return results
