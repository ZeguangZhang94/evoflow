"""L1 经验/记忆库 —— 用户画像记忆（事实型，可合规删除）。

对应「知识放外面 = 可删除」的运维设计：用户事实（偏好、约束、背景）
不蒸馏进权重，而是放在外部记忆里，支持按 key 精确删除（``forget``），
满足隐私合规场景；``render`` 输出可直接注入 system prompt 的文本。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evoflow.l1_memory.store import VectorStore


@dataclass
class UserProfile:
    """一个用户的事实画像。

    :param user_id: 用户标识。
    :param facts: 事实键值对（如 {"饮食偏好": "不吃辣"}）。
    :param updated_at: 最近更新时间戳。
    """

    user_id: str
    facts: dict[str, str] = field(default_factory=dict)
    updated_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> UserProfile:
        return cls(**d)


class ProfileMemory:
    """用户画像记忆：增删查 + prompt 渲染 + 按用户持久化。

    :param store: 可选向量存储；提供时事实会同步嵌入入库，可用
        :meth:`recall` 按语义检索相关事实（事实仍以 key 为准存储，
        store 仅作语义索引）。
    """

    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store
        self._profiles: dict[str, UserProfile] = {}

    # -------------------------------------------------------------- mutate

    def update(self, user_id: str, key: str, value: str) -> UserProfile:
        """写入/覆盖一条事实，并同步语义索引。"""
        profile = self._profiles.setdefault(
            user_id, UserProfile(user_id=user_id)
        )
        profile.facts[key] = value
        profile.updated_at = time.time()
        if self.store is not None:
            self._remove_from_store(user_id, key)
            self.store.add(
                item_id=f"p-{uuid.uuid5(uuid.NAMESPACE_URL, f'evoflow:{user_id}:{key}').hex[:12]}",
                text=f"{key}: {value}",
                payload={"user_id": user_id, "key": key, "value": value},
            )
        return profile

    def forget(self, user_id: str, key: str) -> bool:
        """删除一条事实（合规可删除）。返回是否确实删除了一条记录。"""
        profile = self._profiles.get(user_id)
        removed = False
        if profile is not None and key in profile.facts:
            del profile.facts[key]
            profile.updated_at = time.time()
            removed = True
        removed = self._remove_from_store(user_id, key) or removed
        return removed

    def _remove_from_store(self, user_id: str, key: str) -> bool:
        """从语义索引删除对应记录（仅对暴露 _records 的本地实现生效）。"""
        if self.store is None:
            return False
        records = getattr(self.store, "_records", None)
        if not isinstance(records, list):
            return False
        before = len(records)
        records[:] = [
            r
            for r in records
            if not (
                r.get("payload", {}).get("user_id") == user_id
                and r.get("payload", {}).get("key") == key
            )
        ]
        return len(records) != before

    # --------------------------------------------------------------- query

    def get(self, user_id: str) -> UserProfile:
        """取用户画像；不存在时返回空画像（不落盘）。"""
        return self._profiles.get(user_id) or UserProfile(user_id=user_id)

    def recall(self, user_id: str, query: str, k: int = 3) -> list[tuple[str, str]]:
        """按语义检索该用户最相关的 k 条事实，返回 [(key, value)]。

        需要构造时提供 ``store``；未提供时返回空列表。
        """
        if self.store is None:
            return []
        out: list[tuple[str, str]] = []
        for payload, _score in self.store.search(query, k=k * 3):
            if payload.get("user_id") != user_id:
                continue
            out.append((payload.get("key", ""), payload.get("value", "")))
            if len(out) >= k:
                break
        return out

    def render(self, user_id: str, max_items: int = 20) -> str:
        """渲染为可注入 system prompt 的文本（事实按插入序，最多 max_items 条）。"""
        profile = self.get(user_id)
        lines = [f"[用户画像 {user_id}]"]
        for i, (key, value) in enumerate(profile.facts.items()):
            if i >= max_items:
                lines.append(f"- ...（其余 {len(profile.facts) - max_items} 条略）")
                break
            lines.append(f"- {key}: {value}")
        if not profile.facts:
            lines.append("- （暂无记录）")
        return "\n".join(lines)

    # ------------------------------------------------------------- persist

    def save(self, path: str | Path) -> None:
        """落盘全部画像（jsonl，每行一个用户）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for profile in self._profiles.values():
                f.write(json.dumps(profile.to_dict(), ensure_ascii=False) + "\n")

    def load(self, path: str | Path) -> None:
        """从 jsonl 加载画像；语义索引不重建（store 有自己的 save/load）。"""
        path = Path(path)
        if not path.exists():
            return
        self._profiles.clear()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    profile = UserProfile.from_dict(json.loads(line))
                    self._profiles[profile.user_id] = profile

    def __len__(self) -> int:
        return len(self._profiles)
