"""L1 经验/记忆库 —— A-MEM 式情节记忆（Zettelkasten 卡片笔记）。

对应 A-MEM（arXiv:2502.12110）：每条记忆是一张结构化「卡片」（关键词/标签/
上下文），入库时自动与 top-k 相似笔记互相建链，并支持把高访问、互链的笔记簇
**演化合并**为更抽象的笔记——模拟人类记忆从具体情节到抽象图式的固化过程。

注意：论文与厂商自测的增益数字（如检索准确率提升百分比）均基于其自有基准，
不可全信；落地前请用**自有语料**复测建链密度、演化阈值等超参。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from evoflow.l1_memory.store import VectorStore
from evoflow.llm.client import LLMClient, Message

_EN_WORD = re.compile(r"[A-Za-z0-9_]{2,}")
_CJK = re.compile(r"[一-鿿]")

# 极简停用表：仅过滤最常见虚词，保持零依赖与可预期性。
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "with", "it", "this", "that", "be", "as",
    "的", "了", "和", "是", "在", "我", "你", "他", "她", "它", "们",
}

_LLM_PROMPT = """请为下面这条记忆生成结构化的卡片属性，严格按以下三行格式输出，不要输出其他内容：
关键词: 至多5个，用逗号分隔
标签: 至多3个，用逗号分隔
上下文: 一句话说明这条记忆产生的背景/适用场景

记忆内容：
{text}"""


def _tokenize(text: str) -> list[str]:
    """零依赖分词：英文/数字按词，中文按单字 + 相邻双字 bigram。"""
    tokens = [w.lower() for w in _EN_WORD.findall(text)]
    cjk = _CJK.findall(text)
    tokens.extend(cjk)
    tokens.extend(a + b for a, b in zip(cjk, cjk[1:], strict=False))
    return [t for t in tokens if t not in _STOPWORDS]


def _heuristic_attrs(text: str) -> tuple[list[str], list[str], str]:
    """无 LLM 时的启发式属性提取：高频词做关键词/标签，截断做上下文。"""
    freq: dict[str, int] = {}
    for tok in _tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq, key=lambda t: (-freq[t], t))
    keywords = ranked[:5]
    tags = ranked[:3]
    context = text.strip().replace("\n", " ")[:80]
    return keywords, tags, context


def _parse_llm_attrs(raw: str) -> tuple[list[str], list[str], str] | None:
    """解析 LLM 固定格式输出；格式不符返回 None（调用方回退启发式）。"""
    keywords: list[str] = []
    tags: list[str] = []
    context = ""
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("关键词"):
            keywords = [
                w.strip() for w in re.split(r"[,，、]", line.split(":", 1)[-1]) if w.strip()
            ]
        elif line.startswith("标签"):
            tags = [w.strip() for w in re.split(r"[,，、]", line.split(":", 1)[-1]) if w.strip()]
        elif line.startswith("上下文"):
            context = line.split(":", 1)[-1].strip()
    if not keywords and not tags and not context:
        return None
    return keywords[:5], tags[:3], context


@dataclass
class MemoryNote:
    """一张 Zettelkasten 记忆卡片。

    :param id: 唯一标识。
    :param content: 原始记忆文本。
    :param keywords: 关键词（LLM 生成或启发式提取）。
    :param tags: 标签。
    :param context: 上下文（背景/适用场景一句话）。
    :param links: 互相链接的笔记 id 列表。
    :param created_at: 创建时间戳。
    :param last_accessed: 最近访问时间戳。
    :param access_count: 被检索命中次数。
    :param archived: 是否已被演化合并（归档笔记不参与检索）。
    """

    id: str
    content: str
    keywords: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    context: str = ""
    links: list[str] = field(default_factory=list)
    created_at: float = 0.0
    last_accessed: float = 0.0
    access_count: int = 0
    archived: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> MemoryNote:
        return cls(**d)

    def searchable_text(self) -> str:
        """用于嵌入检索的拼接文本。"""
        parts = [self.content, " ".join(self.keywords), " ".join(self.tags), self.context]
        return "\n".join(p for p in parts if p)


class EpisodicMemory:
    """A-MEM 式情节记忆：卡片化 + 自动建链 + 演化合并。

    :param store: 向量存储（实现 :class:`VectorStore` 协议），作为检索索引；
        卡片本体由本类维护，store 的 payload 只放 ``note_id``。
    :param client: 可选 LLM 客户端；缺省时用零依赖启发式生成卡片属性。
    :param link_k: 新笔记入库时自动建链的相似笔记数量（top-k）。
    """

    def __init__(
        self,
        store: VectorStore,
        client: LLMClient | None = None,
        link_k: int = 3,
    ) -> None:
        self.store = store
        self.client = client
        self.link_k = link_k
        self._notes: dict[str, MemoryNote] = {}

    # ------------------------------------------------------------------ add

    def add(self, text: str) -> MemoryNote:
        """新增一条记忆：生成卡片属性 → 入库 → 与 top-k 相似笔记互相建链。"""
        keywords, tags, context = self._gen_attrs(text)
        now = time.time()
        note = MemoryNote(
            id=f"n-{uuid.uuid4().hex[:12]}",
            content=text,
            keywords=keywords,
            tags=tags,
            context=context,
            created_at=now,
            last_accessed=now,
        )
        self._notes[note.id] = note
        self.store.add(item_id=note.id, text=note.searchable_text(), payload={"note_id": note.id})
        self._auto_link(note)
        return note

    def _gen_attrs(self, text: str) -> tuple[list[str], list[str], str]:
        """有 client 走 LLM（固定格式 + 解析失败回退），否则启发式。"""
        if self.client is not None:
            raw = self.client.chat([Message(role="user", content=_LLM_PROMPT.format(text=text))])
            parsed = _parse_llm_attrs(raw)
            if parsed is not None:
                return parsed
        return _heuristic_attrs(text)

    def _auto_link(self, note: MemoryNote) -> None:
        """检索 top-k 相似笔记并互相写入 links。"""
        if self.link_k <= 0:
            return
        for payload, _score in self.store.search(note.searchable_text(), k=self.link_k + 1):
            other = self._notes.get(payload.get("note_id", ""))
            if other is None or other.id == note.id or other.archived:
                continue
            if other.id not in note.links:
                note.links.append(other.id)
            if note.id not in other.links:
                other.links.append(note.id)

    # -------------------------------------------------------------- retrieve

    def retrieve(
        self,
        query: str,
        k: int = 1,
        include_neighbors: bool = False,
    ) -> list[MemoryNote]:
        """按查询检索最相关的笔记（默认 k=1：相关性 > 数量）。

        命中后更新 ``access_count`` / ``last_accessed``；``include_neighbors``
        为 True 时把命中笔记的链接邻居一并返回（去重，排在命中之后）。
        """
        # 归档笔记仍在索引里，放大检索量再过滤。
        candidates: list[MemoryNote] = []
        for payload, _score in self.store.search(query, k=max(k * 3, k + 5)):
            note = self._notes.get(payload.get("note_id", ""))
            if note is not None and not note.archived:
                candidates.append(note)
            if len(candidates) >= k:
                break
        now = time.time()
        for note in candidates:
            note.access_count += 1
            note.last_accessed = now
        if include_neighbors:
            seen = {n.id for n in candidates}
            for note in list(candidates):
                for link_id in note.links:
                    neighbor = self._notes.get(link_id)
                    if neighbor is not None and not neighbor.archived and neighbor.id not in seen:
                        candidates.append(neighbor)
                        seen.add(neighbor.id)
        return candidates

    # ---------------------------------------------------------------- evolve

    def evolve(self, min_access: int = 2, min_cluster: int = 2) -> list[MemoryNote]:
        """记忆演化：把高访问且互相链接的笔记簇合并成一条更抽象的笔记。

        :param min_access: 参与演化的最低访问次数。
        :param min_cluster: 参与合并的最小簇大小。
        :return: 新生成的抽象笔记列表；被合并的旧笔记标记为 ``archived``，
            其余笔记指向旧笔记的链接会重定向到新笔记。
        """
        hot = {
            n.id
            for n in self._notes.values()
            if not n.archived and n.access_count >= min_access
        }
        clusters = self._connected_clusters(hot)
        created: list[MemoryNote] = []
        for cluster in clusters:
            if len(cluster) < min_cluster:
                continue
            members = [self._notes[i] for i in sorted(cluster)]
            created.append(self._merge(members))
        return created

    def _connected_clusters(self, ids: set[str]) -> list[set[str]]:
        """在 links 图上求 ids 子集的连通分量（无向）。"""
        clusters: list[set[str]] = []
        visited: set[str] = set()
        for start in sorted(ids):
            if start in visited:
                continue
            stack, comp = [start], set()
            while stack:
                cur = stack.pop()
                if cur in comp:
                    continue
                comp.add(cur)
                for nxt in self._notes[cur].links:
                    if nxt in ids and nxt not in comp:
                        stack.append(nxt)
            visited |= comp
            clusters.append(comp)
        return clusters

    def _merge(self, members: Sequence[MemoryNote]) -> MemoryNote:
        """合并一个笔记簇为一条抽象笔记，并重定向外部链接。"""
        keywords = list(dict.fromkeys(kw for m in members for kw in m.keywords))[:5]
        tags = list(dict.fromkeys(t for m in members for t in m.tags))[:3]
        external = {
            link
            for m in members
            for link in m.links
            if link not in {m.id for m in members}
        }
        member_ids = {m.id for m in members}
        now = time.time()
        merged = MemoryNote(
            id=f"n-{uuid.uuid4().hex[:12]}",
            content=self._merge_content(members, keywords),
            keywords=keywords,
            tags=tags,
            context=f"由 {len(members)} 条高访问记忆演化合并而来",
            links=sorted(external),
            created_at=now,
            last_accessed=now,
            access_count=sum(m.access_count for m in members),
        )
        self._notes[merged.id] = merged
        self.store.add(
            item_id=merged.id, text=merged.searchable_text(), payload={"note_id": merged.id}
        )
        for m in members:
            m.archived = True
        # 外部邻居的链接重定向：旧 id → 新 id。
        for other in self._notes.values():
            if other.id in member_ids or other.id == merged.id:
                continue
            if any(link in member_ids for link in other.links):
                redirected = [
                    merged.id if link in member_ids else link for link in other.links
                ]
                other.links = list(dict.fromkeys(redirected))
        return merged

    def _merge_content(self, members: Sequence[MemoryNote], keywords: list[str]) -> str:
        """合并正文：有 client 时让 LLM 抽象总结，否则启发式拼接。"""
        joined = "\n".join(f"- {m.content}" for m in members)
        if self.client is not None:
            raw = self.client.chat(
                [
                    Message(
                        role="user",
                        content=(
                            "下面是一组相关记忆，请用一句话抽象概括它们的共同规律，"
                            "不要输出其他内容：\n" + joined
                        ),
                    )
                ]
            ).strip()
            if raw:
                return raw
        digest = "；".join(m.content.strip().replace("\n", " ")[:40] for m in members)
        return f"共性主题[{', '.join(keywords)}]：{digest}"

    # ------------------------------------------------------------- persist

    def save(self, path: str | Path) -> None:
        """落盘全部卡片（jsonl，每行一张卡片，含归档标记）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for note in self._notes.values():
                f.write(json.dumps(note.to_dict(), ensure_ascii=False) + "\n")

    def load(self, path: str | Path) -> None:
        """从 jsonl 加载卡片并重建检索索引（清空 store 后重嵌未归档卡片）。"""
        path = Path(path)
        if not path.exists():
            return
        self._notes.clear()
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    note = MemoryNote.from_dict(json.loads(line))
                    self._notes[note.id] = note
        records = getattr(self.store, "_records", None)
        if isinstance(records, list):
            records.clear()
        for note in self._notes.values():
            if not note.archived:
                self.store.add(
                    item_id=note.id,
                    text=note.searchable_text(),
                    payload={"note_id": note.id},
                )

    def __len__(self) -> int:
        return sum(1 for n in self._notes.values() if not n.archived)
