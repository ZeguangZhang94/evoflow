"""知识层（Knowledge Layer）：知识放在参数之外。

对应公开研究（ACE / ReasoningBank / QLoRA 等）的三层栈「知识层」——以 RAG 而非微调承载知识，
所有注入文本带 ``[来源:{doc_id}]`` 引用标记（合规硬需求）。

公开 API：

- :class:`Document` / :class:`Chunk` / :class:`KnowledgeLayer`（协议）
- :func:`chunk_text` / :class:`ContextualPrefixer`（Contextual Retrieval）
- :class:`VectorRAG`（向量 + 词频 RRF 混合检索）
- :class:`LightGraphRAG`（LightRAG 双层检索思想的零依赖简化实现）
- :func:`recommend`（路由决策线：语料 <200K token 且 <500 查询/天 →
  长上下文；多跳/全局 → graph；否则 vector）
"""

from evoflow.knowledge.base import Chunk, Document, KnowledgeLayer, format_context
from evoflow.knowledge.chunker import ContextualPrefixer, chunk_text
from evoflow.knowledge.graph_rag import LightGraphRAG
from evoflow.knowledge.router import recommend
from evoflow.knowledge.vector_rag import VectorRAG, rrf_fuse

__all__ = [
    "Chunk",
    "ContextualPrefixer",
    "Document",
    "KnowledgeLayer",
    "LightGraphRAG",
    "VectorRAG",
    "chunk_text",
    "format_context",
    "recommend",
    "rrf_fuse",
]
