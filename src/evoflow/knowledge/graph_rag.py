"""轻量图 RAG：LightRAG（arXiv:2410.05779）双层检索思想的零依赖简化实现。

**简化说明（docstring 承诺）**：本模块是教学级简化实现——实体提取靠
LLM 一次性出词或启发式规则，关系图退化为「实体 → chunk」倒排索引，
不做子图遍历与社区摘要。生产环境建议替换为真 LightRAG / LazyGraphRAG。

双层检索：
- 低层（精确事实）：查询实体命中倒排索引，拿到包含该实体的 chunk；
- 高层（宽泛主题）：查询走向量语义检索；
- 融合排序：先按实体命中数、再按向量得分降序。

增量合并：新文档只需切 chunk、抽实体、更新倒排 dict 并追加 jsonl，
O(1) 增量入库，无需重建全图。
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from evoflow.knowledge.base import Chunk, Document, format_context
from evoflow.knowledge.chunker import chunk_text
from evoflow.l1_memory.store import VectorStore
from evoflow.llm.client import LLMClient, Message

_ENTITY_PROMPT = (
    "从以下文本中抽取关键实体（专有名词、产品、公司、人名、技术术语等），"
    "用逗号分隔输出，只输出实体词本身。\n\n{text}"
)

_CAP_WORD_RE = re.compile(r"[A-Z][A-Za-z0-9]*(?:\.[A-Z][A-Za-z0-9]*)*|[A-Z]{2,}")
_CJK_RUN_RE = re.compile(r"[一-鿿]{2,}")


class LightGraphRAG:
    """双层检索的轻量图知识层。

    :param store: 向量存储（高层语义检索底座）。
    :param client: 可选 LLM 客户端；提供时用 LLM 抽实体，否则用
        大写词 / 中文连续词启发式。
    :param chunk_size: 切分窗口字符数。
    :param index_path: 可选 jsonl 路径；提供时 add 会追加实体索引记录，
        可用 :meth:`load_index` 恢复（倒排索引可持久化，chunk 正文经
        store 持久化）。
    """

    def __init__(
        self,
        store: VectorStore,
        client: LLMClient | None = None,
        chunk_size: int = 500,
        index_path: str | Path | None = None,
    ) -> None:
        self.store = store
        self.client = client
        self.chunk_size = chunk_size
        self.index_path = Path(index_path) if index_path is not None else None
        self._entity_index: dict[str, set[str]] = {}  # entity -> {chunk_id}
        self._chunks: dict[str, Chunk] = {}  # chunk_id -> Chunk

    # ---------------------------------------------------------------- add

    def add(self, documents: Iterable[Document]) -> int:
        """入库一批文档（O(1) 增量合并），返回产生的 chunk 数。"""
        n = 0
        for doc in documents:
            for i, ctext in enumerate(chunk_text(doc.text, self.chunk_size)):
                chunk = Chunk(
                    id=f"{doc.id}-c{i}",
                    doc_id=doc.id,
                    text=ctext,
                    metadata=dict(doc.metadata),
                )
                self.store.add(item_id=chunk.id, text=ctext, payload=chunk.to_dict())
                self._chunks[chunk.id] = chunk
                for entity in self._extract_entities(ctext):
                    self._entity_index.setdefault(entity, set()).add(chunk.id)
                    self._persist_index_record(entity, chunk.id, doc.id)
                n += 1
        return n

    # -------------------------------------------------------------- query

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """双层检索：实体命中（低层）+ 向量语义（高层）融合排序。"""
        # 低层：实体命中
        hit_counts: dict[str, int] = {}
        for e in self._extract_entities(text):
            for ent, cids in self._entity_index.items():
                if e == ent or (len(e) >= 2 and (e in ent or ent in e)):
                    for cid in cids:
                        hit_counts[cid] = hit_counts.get(cid, 0) + 1

        # 高层：向量语义
        vec_hits = self.store.search(text, k=max(k * 2, 10))
        payload_by_id = {p["id"]: p for p, _ in vec_hits}
        vec_score = {p["id"]: s for p, s in vec_hits}

        candidates = set(hit_counts) | set(vec_score)
        ranked = sorted(
            candidates,
            key=lambda cid: (hit_counts.get(cid, 0), vec_score.get(cid, 0.0)),
            reverse=True,
        )
        result: list[Chunk] = []
        for cid in ranked[: max(0, k)]:
            if cid in self._chunks:
                result.append(self._chunks[cid])
            elif cid in payload_by_id:
                result.append(Chunk.from_dict(payload_by_id[cid]))
        return result

    def augment(self, query: str, k: int = 4, max_tokens: int = 2000) -> str:
        """检索并拼成带 ``[来源:{doc_id}]`` 引用标记的注入文本。"""
        return format_context(self.query(query, k=k), max_tokens=max_tokens)

    # ----------------------------------------------------------- entities

    def _extract_entities(self, text: str) -> list[str]:
        """实体抽取：有 client 走 LLM，否则启发式（大写词 + 中文连续词）。"""
        if self.client is not None:
            messages: list[Message] = [
                {"role": "user", "content": _ENTITY_PROMPT.format(text=text)}
            ]
            resp = self.client.chat(messages)
            seen: dict[str, None] = {}
            for e in re.split(r"[,，、;；\n/]", resp):
                e = e.strip()
                if e:
                    seen[e] = None
            return list(seen)
        ents = _CAP_WORD_RE.findall(text)
        ents.extend(_CJK_RUN_RE.findall(text))
        return list(dict.fromkeys(ents))

    # --------------------------------------------------------- persistence

    def _persist_index_record(self, entity: str, chunk_id: str, doc_id: str) -> None:
        if self.index_path is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"entity": entity, "chunk_id": chunk_id, "doc_id": doc_id},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def load_index(self, path: str | Path | None = None) -> int:
        """从 jsonl 恢复实体倒排索引，返回加载的记录数。

        注意：只恢复「实体 → chunk_id」映射；chunk 正文需经 store 加载后
        由向量检索路径返回（简化实现的已知边界）。
        """
        path = Path(path) if path is not None else self.index_path
        if path is None or not path.exists():
            return 0
        n = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._entity_index.setdefault(rec["entity"], set()).add(rec["chunk_id"])
                n += 1
        return n

    def __len__(self) -> int:
        return len(self._chunks)
