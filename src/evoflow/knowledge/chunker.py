"""文本切分与上下文前缀（Contextual Retrieval 轻量实现）。

- :func:`chunk_text`：纯标准库滑窗切分，中英文混合按字符计。
- :class:`ContextualPrefixer`：对应 Anthropic Contextual Retrieval
  （官方博客 2024-09，无 arXiv）——为每个 chunk 生成一段"它在原文中的
  位置/背景"前缀，拼在 chunk 前参与嵌入，显著降低检索时的语境丢失。
  有 LLM client 时让模型生成 50–100 token 前缀；无 client 时退化为
  启发式（metadata 标题 + 文档首句）。
"""

from __future__ import annotations

import re

from evoflow.knowledge.base import Document
from evoflow.llm.client import LLMClient, Message

_PREFIX_PROMPT = (
    "以下是完整文档和该文档中的一个片段。请用 50–100 个 token 写出该片段"
    "在文档中的上下文背景（属于哪个主题、承接什么内容），只输出背景文字本身，"
    "不要输出解释。\n\n<文档>\n{document}\n</文档>\n\n<片段>\n{chunk}\n</片段>"
)


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[str]:
    """按字符滑窗切分文本（中英文混合按字符计）。

    :param text: 输入文本。
    :param size: 每个 chunk 的最大字符数。
    :param overlap: 相邻 chunk 的重叠字符数，缓解语义在边界被切断。
    :returns: chunk 字符串列表；空文本返回空列表。
    :raises ValueError: ``size <= 0`` 或 ``overlap`` 不在 ``[0, size)`` 内。
    """
    if size <= 0:
        raise ValueError("size 必须为正整数")
    if not 0 <= overlap < size:
        raise ValueError("overlap 必须满足 0 <= overlap < size")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    chunks = [text[i : i + size] for i in range(0, len(text), step)]
    # 最后一个窗口若只是上一窗口的重叠尾巴（不足一步新内容），丢弃
    if len(chunks) > 1 and len(chunks[-1]) <= overlap:
        chunks.pop()
    return chunks


class ContextualPrefixer:
    """为 chunk 生成上下文前缀。

    :param client: 可选 LLM 客户端；为 None 时使用启发式前缀
        （metadata 标题 + 文档首句）。
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client

    def prefix(self, document: Document, chunk: str) -> str:
        """生成 ``chunk`` 在 ``document`` 中的上下文前缀。"""
        if self.client is not None:
            return self._llm_prefix(document, chunk)
        return self._heuristic_prefix(document)

    def _llm_prefix(self, document: Document, chunk: str) -> str:
        if self.client is None:  # 防御：无客户端时走启发式前缀
            return self._heuristic_prefix(document)
        messages: list[Message] = [
            {
                "role": "user",
                "content": _PREFIX_PROMPT.format(document=document.text, chunk=chunk),
            }
        ]
        return self.client.chat(messages).strip()

    @staticmethod
    def _heuristic_prefix(document: Document) -> str:
        title = str(document.metadata.get("title", "")).strip()
        first_sentence = re.split(r"[。！？!?\n.]", document.text.strip(), maxsplit=1)[
            0
        ][:80]
        parts = [p for p in (title, first_sentence) if p]
        return " | ".join(parts)
