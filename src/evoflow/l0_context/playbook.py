"""L0 上下文进化 —— Playbook（类 ACE 的增量 delta 条目手册）。

对应 ACE 的做法（详见 docs/research-to-code.md）：不重写整个 system prompt，
而是以增量 DeltaEntry 的形式
持续累积"可执行的教训"，Curator 负责去重合并，render 时按 token 预算
拼进 system prompt。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _normalize(text: str) -> str:
    """规范化文本用于去重比较。"""
    return " ".join(text.split()).lower()


def _estimate_tokens(text: str) -> int:
    """粗略 token 估算：中文按 1 字 ≈ 1 token、英文按 4 字符 ≈ 1 token 折中。"""
    return max(1, len(text) // 2)


@dataclass
class DeltaEntry:
    """一条增量 playbook 条目。

    :param id: 条目唯一标识。
    :param content: 条目内容（一条可执行的教训/规则）。
    :param tags: 标签（用于分类与检索）。
    :param created_at: 创建时间戳。
    :param hits: 命中次数（被使用时累加，用于排序与淘汰）。
    """

    id: str
    content: str
    tags: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    hits: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DeltaEntry:
        return cls(**d)


class Playbook:
    """增量 playbook：条目的集合与渲染器。

    条目按内容去重；渲染按 hits 降序并受 token 预算约束。
    """

    def __init__(self, entries: Iterable[DeltaEntry] | None = None) -> None:
        self.entries: list[DeltaEntry] = list(entries or [])

    def curate(self, deltas: Iterable[DeltaEntry]) -> list[DeltaEntry]:
        """Curator：合并一批 delta 条目，去重后追加。

        去重键为规范化后的 content；重复条目合并 tags（并集）、累加 hits、
        保留最早的 created_at 与原 id。

        :return: 本次真正新增的条目列表。
        """
        index = {_normalize(e.content): e for e in self.entries}
        added: list[DeltaEntry] = []
        for d in deltas:
            key = _normalize(d.content)
            if key in index:
                existing = index[key]
                existing.tags = sorted(set(existing.tags) | set(d.tags))
                existing.hits += d.hits
                existing.created_at = min(existing.created_at, d.created_at)
            else:
                self.entries.append(d)
                index[key] = d
                added.append(d)
        return added

    def render(self, max_tokens: int = 2000) -> str:
        """把 playbook 渲染为可拼进 system prompt 的文本。

        按 hits 降序排列（高频经验优先），超出 token 预算即截断。
        """
        lines = ["## Playbook（进化中的经验手册）"]
        budget = max_tokens - _estimate_tokens(lines[0])
        for e in sorted(self.entries, key=lambda x: (-x.hits, x.created_at)):
            tag = f"[{','.join(e.tags)}] " if e.tags else ""
            line = f"- {tag}{e.content}"
            cost = _estimate_tokens(line)
            if cost > budget:
                break
            lines.append(line)
            budget -= cost
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """保存为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump([e.to_dict() for e in self.entries], f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Playbook:
        """从 JSON 文件加载；文件不存在时返回空 playbook。"""
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(DeltaEntry.from_dict(d) for d in data)

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[DeltaEntry]:
        return iter(self.entries)
