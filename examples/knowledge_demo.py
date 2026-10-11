"""EvoFlow 知识层（evoflow.knowledge）零依赖演示。

演示内容：

1. VectorRAG —— 混合检索（词频 + 向量，RRF 融合）+ Contextual Retrieval
   前缀；augment 输出带 ``[来源:doc_id]`` 引用标记的注入文本。
2. LightGraphRAG —— LightRAG 双层检索思想的零依赖简化实现：低层实体
   倒排 + 高层向量语义融合。
3. router.recommend —— 按语料规模 / 查询量 / 查询形态给出
   long_context / graph / vector 路由建议。

LLM 全部用 ``evoflow.llm.ScriptedLLM``（预置答案的假 LLM），无需联网、
无需 API Key。

运行：``python examples/knowledge_demo.py``（零依赖）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 未安装包时直接从 src 布局导入，保证裸环境可跑。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from evoflow import (
    Document,
    JsonlVectorStore,
    LightGraphRAG,
    ScriptedLLM,
    VectorRAG,
)
from evoflow.knowledge import recommend


def hr(title: str) -> None:
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ------------------------------------------------------------------ 演示语料
DOCS = [
    Document(
        id="doc-refund",
        text="退货退款政策：商品签收后 7 天内支持无理由退货，退回运费由买家承担；"
        "退款在仓库验收通过后 3 个工作日内原路退回支付账户。"
        "定制类商品不支持无理由退货。",
        metadata={"title": "退货退款政策"},
    ),
    Document(
        id="doc-api",
        text="开放平台 API 限流说明：每个 API Key 默认限流 60 次/分钟，"
        "触发限流返回 HTTP 429，指数退避重试；企业版可申请提升至 600 次/分钟。",
        metadata={"title": "API 限流说明"},
    ),
    Document(
        id="doc-deploy",
        text="部署手册：使用 docker compose up -d 启动全部服务；"
        "首次启动前需执行 alembic upgrade head 初始化数据库；"
        "配置文件位于 config/app.yaml，端口默认 8080。",
        metadata={"title": "部署手册"},
    ),
]


def main() -> None:
    # ---------------------------------------------------------- 1. VectorRAG
    hr("1. VectorRAG：混合检索 + Contextual Retrieval 前缀")
    # ScriptedLLM：为每个 chunk 预置一段"它在原文中的位置/背景"前缀。
    # 3 篇文档均不足 chunk_size（500 字符），各产 1 个 chunk → 3 次 LLM 调用。
    prefix_llm = ScriptedLLM(
        answers=[
            "本片段出自《退货退款政策》，说明退货时限与退款到账周期。",
            "本片段出自《API 限流说明》，说明默认限流阈值与超限行为。",
            "本片段出自《部署手册》，说明服务启动与数据库初始化步骤。",
        ],
        default="（默认前缀）",
    )
    vector_rag = VectorRAG(JsonlVectorStore(), client=prefix_llm)
    n = vector_rag.add(DOCS)
    print(f"  入库 {len(DOCS)} 篇文档 -> {n} 个 chunk（LLM 生成 contextual prefix）")
    sample_chunk = vector_rag.query("退款多久到账", k=1)[0]
    print(f"  chunk 前缀示例: {sample_chunk.context_prefix}")

    print("\n  查询「退款多久到账」的 augment 注入文本（带引用标记）：")
    for line in vector_rag.augment("退款多久到账", k=2).splitlines():
        print(f"    {line[:80]}{'...' if len(line) > 80 else ''}")

    # ---------------------------------------------------------- 2. LightGraphRAG
    hr("2. LightGraphRAG：实体倒排（低层）+ 向量语义（高层）双层检索")
    # ScriptedLLM：入库时抽实体（每 chunk 一次调用）；查询时再抽查询实体。
    entity_llm = ScriptedLLM(
        answers=[
            "退货，退款，运费",  # doc-refund 的实体
            "API Key，限流，HTTP 429",  # doc-api 的实体
            "docker compose，数据库，端口",  # doc-deploy 的实体
        ],
        default="退款",  # 查询「退款流程怎么走」时抽到的实体
    )
    graph_rag = LightGraphRAG(JsonlVectorStore(), client=entity_llm)
    n = graph_rag.add(DOCS)
    print(
        f"  入库 {len(DOCS)} 篇文档 -> {n} 个 chunk，"
        f"实体倒排索引 {len(graph_rag._entity_index)} 个实体"
    )
    print(f"  倒排索引示例: 实体「退款」-> {sorted(graph_rag._entity_index.get('退款', []))}")

    print("\n  查询「退款流程怎么走」的 augment 注入文本（带引用标记）：")
    for line in graph_rag.augment("退款流程怎么走", k=2).splitlines():
        print(f"    {line[:80]}{'...' if len(line) > 80 else ''}")

    # ---------------------------------------------------------- 3. 路由建议
    hr("3. router.recommend：知识层选型路由")
    scenarios = [
        ("小型 FAQ（8 万 token，50 查询/天，单跳）", 80_000, 50, False),
        ("企业知识库（500 万 token，含多跳/全局查询）", 5_000_000, 2_000, True),
        ("中型文档集（100 万 token，800 查询/天，单跳）", 1_000_000, 800, False),
    ]
    for label, tokens, qpd, multi_hop in scenarios:
        layer = recommend(corpus_tokens=tokens, queries_per_day=qpd, multi_hop=multi_hop)
        print(f"  {label} -> {layer}")

    hr("knowledge demo 完成")


if __name__ == "__main__":
    main()
