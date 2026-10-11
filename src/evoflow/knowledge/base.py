"""知识层基础类型：Document / Chunk / KnowledgeLayer 协议。

核心设计原则：知识放在参数之外（RAG 而非微调进权重）。
所有实现必须支持 ``augment`` 输出带 ``[来源:{doc_id}]`` 引用标记的文本——
引用溯源是合规硬需求。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Protocol


@dataclass
class Document:
    """一篇待入库的原始文档。

    :param id: 文档唯一标识（引用溯源的锚点）。
    :param text: 文档全文。
    :param metadata: 任意元数据（如 title、source、url）。
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """文档切分后的检索单元。

    :param id: chunk 唯一标识（通常为 ``{doc_id}-c{序号}``）。
    :param doc_id: 所属文档 id。
    :param text: chunk 正文。
    :param context_prefix: 上下文前缀（Contextual Retrieval），仅参与嵌入，
        不作为正文返回。
    :param metadata: 继承自文档的元数据。
    """

    id: str
    doc_id: str
    text: str
    context_prefix: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Chunk:
        return cls(**d)


class KnowledgeLayer(Protocol):
    """知识层协议：入库 / 检索 / 注入。"""

    def add(self, documents: Iterable[Document]) -> int:
        """入库一批文档，返回产生的 chunk 数。"""
        ...

    def query(self, text: str, k: int = 4) -> list[Chunk]:
        """按查询检索最相关的 chunk，最多 k 条。"""
        ...

    def augment(self, query: str, k: int = 4, max_tokens: int = 2000) -> str:
        """检索并拼成可注入 prompt 的带引用标记文本。"""
        ...


def format_context(chunks: Sequence[Chunk], max_tokens: int = 2000) -> str:
    """把 chunk 列表拼成带引用标记的注入文本。

    每行格式：``[来源:{doc_id}] 内容``。
    ``max_tokens`` 按字符数近似控制预算（中英文混合下 1 token ≈ 1–2 字符，
    这里取保守的按字符截断）；预算用尽即停止追加。
    """
    lines: list[str] = []
    used = 0
    for c in chunks:
        line = f"[来源:{c.doc_id}] {c.text}"
        if used + len(line) > max_tokens:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
