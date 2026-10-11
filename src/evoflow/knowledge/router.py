"""知识层路由：按语料规模与查询形态选择检索方案的决策线。

决策线（按优先级）：
1. 语料 < 200K token 且 < 500 查询/天 → ``long_context``
   （直接塞进长上下文，省掉检索基础设施）；
2. 多跳 / 全局性查询 → ``graph``（LightGraphRAG）；
3. 其余 → ``vector``（VectorRAG）。
"""

from __future__ import annotations

LONG_CONTEXT_CORPUS_TOKENS = 200_000
LONG_CONTEXT_QUERIES_PER_DAY = 500


def recommend(
    corpus_tokens: int,
    queries_per_day: int,
    multi_hop: bool = False,
) -> str:
    """按语料规模、查询量与查询形态推荐知识层方案。

    :param corpus_tokens: 语料总 token 数（近似值即可）。
    :param queries_per_day: 日均查询量。
    :param multi_hop: 查询是否以多跳 / 全局性为主。
    :returns: ``"long_context"`` / ``"graph"`` / ``"vector"``。
    """
    if (
        corpus_tokens < LONG_CONTEXT_CORPUS_TOKENS
        and queries_per_day < LONG_CONTEXT_QUERIES_PER_DAY
    ):
        return "long_context"
    if multi_hop:
        return "graph"
    return "vector"
