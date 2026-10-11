"""向量 RAG：chunk →（可选 contextual prefix）→ embed → 入库。

对应公开研究中的混合检索共识（向量 + 词频双路召回可显著提升召回率，详见
docs/research-to-code.md）：``hybrid=True``（默认）时，把简单
词频匹配（BM25 的零依赖替身，不引外部库）与向量相似度两套排序做
RRF（Reciprocal Rank Fusion, k=60）融合；``hybrid=False`` 时纯向量检索。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from evoflow.knowledge.base import Chunk, Document, format_context
from evoflow.knowledge.chunker import ContextualPrefixer, chunk_text
from evoflow.l1_memory.store import VectorStore
from evoflow.llm.client import LLMClient

RRF_K = 60  # RRF 平滑常数（论文常用值）

_CJK_RE = re.compile(r"[一-鿿]")
_WORD_RE = re.compile(r"[a-z0-9]+")


def rrf_fuse(
    rankings: Sequence[Sequence[str]], k: int = RRF_K
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion：多套 id 排序融合为一套。

    每个 id 得分为各排序中 ``1 / (k + rank)`` 之和（rank 从 1 起）。
    返回 ``[(id, score)]`` 降序列表。
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _terms(text: str) -> list[str]:
    """查询词项：英文小写词 + 中文 bigram（零依赖替代分词器）。"""
    t = text.lower()
    terms = _WORD_RE.findall(t)
    cjk_runs = re.findall(r"[一-鿿]+", t)
    for run in cjk_runs:
        if len(run) == 1:
            terms.append(run)
        else:
            terms.extend(run[i : i + 2] for i in range(len(run) - 1))
    return terms


def _tf_score(terms: Sequence[str], text: str) -> float:
    """简单词频得分：查询词项在文本中的命中次数之和。"""
    t = text.lower()
    return float(sum(t.count(term) for term in terms))


class VectorRAG:
    """基于 :class:`VectorStore` 的知识层实现。

    :param store: 向量存储（实现 :class:`VectorStore` 协议）。
    :param client: 可选 LLM 客户端；提供时用 LLM 生成 contextual prefix，
        否则用启发式前缀。
    :param chunk_size: 切分窗口字符数。
    :param overlap: 相邻 chunk 重叠字符数。
    :param hybrid: 是否启用词频 + 向量的 RRF 混合检索。
    """

    def __init__(
        self,
        store: VectorStore,
        client: LLMClient | None = None,
        chunk_size: int = 500,
        overlap: int = 50,
        hybrid: bool = True,
    ) -> None:
        self.store = store
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.hybrid = hybrid
        self._prefixer = ContextualPrefixer(client)
        self._chunks: dict[str, Chunk] = {}  # chunk_id -> Chunk（本会话索引）

    def add(self, documents: Iterable[Document]) -> int:
        """入库一批文档，返回产生的 chunk 数。"""
        n = 0
        for doc in documents:
            for i, ctext in enumerate(chunk_text(doc.text, self.chunk_size, self.overlap)):
                prefix = self._prefixer.prefix(doc, ctext)
                chunk = Chunk(
                    id=f"{doc.id}-c{i}",
                    doc_id=doc.id,
                    text=ctext,
                    context_prefix=prefix,
                    metadata=dict(doc.metadata),
                )
                embed_text = f"{prefix}\n{ctext}" if prefix else ctext
                self.store.add(item_id=chunk.id, text=embed_text, payload=chunk.to_dict())
                self._chunks[chunk.id] = chunk
                n += 1
        return n

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """检索最相关的 chunk。hybrid 模式下为 RRF 融合排序。"""
        if not self.hybrid:
            return [Chunk.from_dict(p) for p, _ in self.store.search(text, k=k)]

        vec_hits = self.store.search(text, k=max(k * 4, 20))
        payload_by_id = {p["id"]: p for p, _ in vec_hits}
        vec_rank = [p["id"] for p, _ in vec_hits]

        terms = _terms(text)
        scored = [
            (cid, _tf_score(terms, c.text))
            for cid, c in self._chunks.items()
        ]
        tf_rank = [cid for cid, s in sorted(scored, key=lambda x: x[1], reverse=True) if s > 0]

        fused = rrf_fuse([vec_rank, tf_rank])
        result: list[Chunk] = []
        for cid, _ in fused[: max(0, k)]:
            if cid in self._chunks:
                result.append(self._chunks[cid])
            elif cid in payload_by_id:
                result.append(Chunk.from_dict(payload_by_id[cid]))
        return result

    def augment(self, query: str, k: int = 4, max_tokens: int = 2000) -> str:
        """检索并拼成带 ``[来源:{doc_id}]`` 引用标记的注入文本。"""
        return format_context(self.query(query, k=k), max_tokens=max_tokens)

    def __len__(self) -> int:
        return len(self._chunks)
