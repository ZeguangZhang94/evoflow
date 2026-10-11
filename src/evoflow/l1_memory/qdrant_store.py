"""L1 经验/记忆库 —— Qdrant 向量后端（可选 extras: ``evoflow[vec-qdrant]``）。

实现 :class:`VectorStore` 协议，把嵌入与检索卸载到 Qdrant 服务端。
**零依赖承诺不受影响**：本模块自身只用标准库；``qdrant_client`` 在构造
:class:`QdrantStore` 时才惰性 import，缺失时抛出带安装指引的 ImportError。
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evoflow.l1_memory.store import EmbedFn, hashing_embed

if TYPE_CHECKING:  # 仅类型标注，不产生运行时依赖
    import qdrant_client as _qc_types  # noqa: F401

# 模块加载期不强制依赖 qdrant_client；装了则顺带拿到 models（构造 Point 用），
# 没装时 _client 注入路径（测试/自定义客户端）退化为 dict 形式的 point。
try:
    from qdrant_client import models as _models
except ImportError:  # pragma: no cover - 取决于环境是否安装
    _models = None

_MISSING_MSG = (
    "QdrantStore 需要 qdrant-client，请 pip install evoflow[vec-qdrant] "
    "（或 pip install qdrant-client）后重试"
)


def _require_qdrant() -> Any:
    """惰性 import qdrant_client；缺失时给出明确的安装指引。"""
    try:
        import qdrant_client
    except ImportError as e:
        raise ImportError(_MISSING_MSG) from e
    return qdrant_client


class QdrantStore:
    """Qdrant 向量存储，语义对齐 :class:`VectorStore` 协议。

    :param collection: collection 名；不存在时按向量维度惰性创建（COSINE 距离）。
    :param url: Qdrant 服务地址。
    :param vector_size: 向量维度；缺省时在首次 ``add`` 时用 ``embed_fn``
        实际输出的维度创建 collection。
    :param api_key: 可选 API Key（Qdrant Cloud / 开启了鉴权的部署）。
    :param embed_fn: 可插拔嵌入函数，语义与 :class:`JsonlVectorStore` 一致；
        缺省用零依赖 :func:`hashing_embed`。注意换 embed_fn 后需保证
        collection 维度匹配，否则 Qdrant 会拒绝写入。
    :param _client: 私有测试参数：直接注入已构造的客户端（如 fake），
        传入后跳过惰性 import 与真实连接。
    """

    def __init__(
        self,
        collection: str,
        url: str = "http://localhost:6333",
        vector_size: int | None = None,
        api_key: str | None = None,
        embed_fn: EmbedFn | None = None,
        _client: Any = None,
    ) -> None:
        self.collection = collection
        self.embed_fn: EmbedFn = embed_fn or hashing_embed
        self._vector_size = vector_size
        self._collection_ready = False
        if _client is not None:
            self._client = _client
        else:
            qc = _require_qdrant()
            self._client = qc.QdrantClient(url=url, api_key=api_key)

    # ------------------------------------------------------------ internal

    def _ensure_collection(self, dim: int) -> None:
        """惰性创建 collection（首次写入时按实际向量维度建）。"""
        if self._collection_ready:
            return
        existing = {c.name for c in self._client.get_collections().collections}
        if self.collection not in existing:
            size = self._vector_size or dim
            if _models is not None:
                vectors_config = _models.VectorParams(
                    size=size, distance=_models.Distance.COSINE
                )
            else:
                vectors_config = {"size": size, "distance": "Cosine"}
            self._client.create_collection(
                collection_name=self.collection, vectors_config=vectors_config
            )
        self._collection_ready = True

    @staticmethod
    def _point_id(item_id: str) -> str:
        """把任意字符串 id 映射为 Qdrant 接受的确定性 UUID。"""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"evoflow:{item_id}"))

    # ------------------------------------------------------------- protocol

    def add(self, item_id: str, text: str, payload: dict) -> None:
        """嵌入 text 并 upsert 一条记录（同 item_id 覆盖，幂等）。"""
        vector = self.embed_fn(text)
        self._ensure_collection(len(vector))
        point_id = self._point_id(item_id)
        if _models is not None:
            point = _models.PointStruct(id=point_id, vector=vector, payload=dict(payload))
        else:
            point = {"id": point_id, "vector": vector, "payload": dict(payload)}
        self._client.upsert(collection_name=self.collection, points=[point])

    def search(self, query: str, k: int = 1) -> list[tuple[dict, float]]:
        """按相似度检索，返回 [(payload, score)] 降序，最多 k 条。

        兼容 qdrant-client 新旧两套 API：优先 ``query_points``（>=1.10），
        否则退回 ``search``。
        """
        qv = self.embed_fn(query)
        if hasattr(self._client, "query_points"):
            res = self._client.query_points(
                collection_name=self.collection, query=qv, limit=k, with_payload=True
            )
            points = res.points
        else:
            points = self._client.search(
                collection_name=self.collection,
                query_vector=qv,
                limit=k,
                with_payload=True,
            )
        return [(dict(p.payload or {}), float(p.score)) for p in points]

    def save(self, path: str | Path) -> None:
        """no-op：Qdrant 由服务端负责持久化（WAL + snapshot），本地不落盘。

        保留本方法仅为满足 :class:`VectorStore` 协议签名，调用方代码
        （如 save/load 流程）无需为后端写分支。如需本地备份请使用
        Qdrant 自带的 snapshot API。
        """
        return

    def load(self, path: str | Path) -> None:
        """no-op：数据在服务端，连接即可用，无需本地加载。见 :meth:`save`。"""
        return
