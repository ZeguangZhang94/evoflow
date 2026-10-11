"""知识层测试：chunker / contextual prefix / VectorRAG / RRF / LightGraphRAG / router。

零依赖，LLM 路径全部走 ScriptedLLM。
"""

import pytest

from evoflow.knowledge import (
    Chunk,
    ContextualPrefixer,
    Document,
    LightGraphRAG,
    VectorRAG,
    chunk_text,
    recommend,
    rrf_fuse,
)
from evoflow.l1_memory.store import JsonlVectorStore
from evoflow.llm.client import ScriptedLLM

# ---------------------------------------------------------------- chunker


def test_chunk_text_short_and_empty():
    assert chunk_text("") == []
    assert chunk_text("  ") == []
    assert chunk_text("短文本", size=500) == ["短文本"]


def test_chunk_text_boundaries_and_overlap():
    text = "a" * 1000
    chunks = chunk_text(text, size=500, overlap=50)
    assert len(chunks) == 3  # 起点 0 / 450 / 900
    assert chunks[0] == text[0:500]
    assert chunks[1] == text[450:950]
    assert chunks[1][:50] == text[450:500]  # 重叠段一致
    assert chunks[2] == text[900:1000]


def test_chunk_text_drops_pure_overlap_tail():
    text = "b" * 950
    chunks = chunk_text(text, size=500, overlap=50)
    # 最后一个窗口 [900:950] 长度 50 <= overlap，纯重叠尾巴被丢弃
    assert len(chunks) == 2


def test_chunk_text_mixed_cjk_by_chars():
    text = "中文mixed" * 200  # 中英文混合按字符切
    chunks = chunk_text(text, size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert chunks[1][:10] == text[90:100]  # 重叠段与原文一致


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError):
        chunk_text("x", size=0)
    with pytest.raises(ValueError):
        chunk_text("x", size=100, overlap=100)
    with pytest.raises(ValueError):
        chunk_text("x", size=100, overlap=-1)


# ------------------------------------------------------- ContextualPrefixer


def test_prefixer_llm_path():
    client = ScriptedLLM(answers=["该片段出自财报第三章，讨论年度营收。"])
    prefixer = ContextualPrefixer(client)
    doc = Document(id="d1", text="年度财报全文……营收增长 20%……", metadata={})
    prefix = prefixer.prefix(doc, "营收增长 20%")
    assert prefix == "该片段出自财报第三章，讨论年度营收。"
    # 提示词同时包含文档与片段
    prompt = client.calls[0][0]["content"]
    assert "营收增长 20%" in prompt and "年度财报全文" in prompt


def test_prefixer_heuristic_fallback():
    prefixer = ContextualPrefixer(None)
    doc = Document(
        id="d1",
        text="本年度营收大幅增长。其余细节略。",
        metadata={"title": "年度报告"},
    )
    prefix = prefixer.prefix(doc, "任意片段")
    assert "年度报告" in prefix
    assert "本年度营收大幅增长" in prefix


# -------------------------------------------------------------- VectorRAG


def _docs() -> list[Document]:
    return [
        Document(
            id="d1",
            text="苹果是一种营养丰富的水果，富含维生素和膳食纤维。",
            metadata={"title": "水果"},
        ),
        Document(
            id="d2",
            text="量子计算机利用量子比特进行大规模并行计算。",
            metadata={"title": "计算"},
        ),
    ]


def test_vector_rag_add_returns_chunk_count():
    rag = VectorRAG(JsonlVectorStore(), chunk_size=10, overlap=2)
    n = rag.add(_docs())
    assert n == len(rag)
    assert n > 2  # 小窗口强制多 chunk


def test_vector_rag_query_returns_relevant_chunk():
    rag = VectorRAG(JsonlVectorStore(), hybrid=False)
    rag.add(_docs())
    hits = rag.query("苹果 水果", k=2)
    assert hits and all(isinstance(c, Chunk) for c in hits)
    assert hits[0].doc_id == "d1"


def test_vector_rag_augment_has_citation_marker():
    rag = VectorRAG(JsonlVectorStore())
    rag.add(_docs())
    ctx = rag.augment("苹果", k=2)
    assert "[来源:" in ctx
    assert "[来源:d1]" in ctx


def test_vector_rag_augment_respects_budget():
    rag = VectorRAG(JsonlVectorStore())
    rag.add(_docs())
    ctx = rag.augment("苹果", k=4, max_tokens=20)
    assert len(ctx) <= 40  # 至多放入一条短行即停
    assert ctx.count("[来源:") <= 1


def test_vector_rag_hybrid_rrf_prefers_exact_terms():
    # d2 在词频上反复命中查询词，纯向量未必占优；混合后应排第一
    d1 = Document(id="v1", text="苹果是一种营养丰富的水果。")
    d2 = Document(id="v2", text="苹果 苹果 苹果，苹果新品发布会宣布苹果全线升级。")
    rag = VectorRAG(JsonlVectorStore(), hybrid=True)
    rag.add([d1, d2])
    hits = rag.query("苹果", k=2)
    assert hits[0].doc_id == "v2"


def test_rrf_fuse_ordering():
    r1 = ["a", "b", "c"]
    r2 = ["b", "a", "d"]
    fused = rrf_fuse([r1, r2], k=60)
    ids = [i for i, _ in fused]
    # a、b 各在两榜前列且分数相同，c、d 各只在一榜，靠后
    assert set(ids[:2]) == {"a", "b"}
    assert set(ids[2:]) == {"c", "d"}
    # 单榜第一 > 双榜垫底
    fused2 = dict(rrf_fuse([["x"], ["y", "z", "x"]], k=60))
    assert fused2["x"] > fused2["y"] > fused2["z"]


# ---------------------------------------------------------- LightGraphRAG


def test_graph_rag_dual_layer_entity_hit():
    g = LightGraphRAG(JsonlVectorStore())
    g.add(
        [
            Document(id="g1", text="OpenAI 于 2024 年发布了新一代 GPT 模型。"),
            Document(id="g2", text="Python 是一种流行的编程语言。"),
        ]
    )
    hits = g.query("OpenAI", k=2)
    assert hits[0].doc_id == "g1"  # 低层实体精确命中优先


def test_graph_rag_vector_fallback_when_no_entity_match():
    g = LightGraphRAG(JsonlVectorStore())
    g.add([Document(id="g1", text="苹果是一种营养丰富的水果。")])
    # 查询无可抽取实体（全小写英文），退化为纯向量高层检索
    hits = g.query("hello world", k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "g1"


def test_graph_rag_llm_entity_extraction():
    client = ScriptedLLM(answers=["甲公司, 乙产品"])
    g = LightGraphRAG(JsonlVectorStore(), client=client)
    g.add([Document(id="g1", text="甲公司发布了乙产品。")])
    assert "甲公司" in g._entity_index
    assert "乙产品" in g._entity_index
    assert "甲公司发布了乙产品" in client.calls[0][0]["content"]


def test_graph_rag_incremental_add():
    g = LightGraphRAG(JsonlVectorStore())
    g.add([Document(id="g1", text="OpenAI 发布了新模型。")])
    assert len(g) == 1
    # O(1) 增量合并：直接追加，无需重建
    g.add([Document(id="g2", text="Tesla 公布了新款电动车。")])
    assert len(g) == 2
    assert g.query("Tesla", k=1)[0].doc_id == "g2"


def test_graph_rag_index_persistence_roundtrip(tmp_path):
    path = tmp_path / "entity_index.jsonl"
    g = LightGraphRAG(JsonlVectorStore(), index_path=path)
    g.add([Document(id="g1", text="OpenAI 发布了新模型。")])
    assert path.exists()

    g2 = LightGraphRAG(JsonlVectorStore(), index_path=path)
    n = g2.load_index()
    assert n > 0
    assert "OpenAI" in g2._entity_index
    assert "g1-c0" in g2._entity_index["OpenAI"]


def test_graph_rag_augment_has_citation_marker():
    g = LightGraphRAG(JsonlVectorStore())
    g.add([Document(id="g1", text="OpenAI 发布了新模型。")])
    assert "[来源:g1]" in g.augment("OpenAI")


# ---------------------------------------------------------------- router


def test_router_decision_boundaries():
    # 语料 <200K 且 <500 查询/天 → 长上下文
    assert recommend(199_999, 499) == "long_context"
    # 语料越界 → 不再长上下文
    assert recommend(200_000, 100) == "vector"
    # 查询量越界 → 不再长上下文
    assert recommend(100_000, 500) == "vector"
    # 多跳/全局 → graph（前提是不满足长上下文线）
    assert recommend(300_000, 100, multi_hop=True) == "graph"
    assert recommend(200_000, 600, multi_hop=True) == "graph"
    # 决策线按优先级：小语料即使多跳也先走长上下文
    assert recommend(100_000, 100, multi_hop=True) == "long_context"
