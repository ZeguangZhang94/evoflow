"""L1 Qdrant 向量后端测试：缺依赖报错 + fake client 协议调用（不装 qdrant_client）。"""

import sys
from types import SimpleNamespace

import pytest

from evoflow.l1_memory.qdrant_store import QdrantStore


class FakeQdrantClient:
    """最小 fake：只实现 QdrantStore 触达的旧版 search API（无 query_points）。"""

    def __init__(self):
        self.collections: list[str] = []
        self.created: list[dict] = []
        self.upserted: list[tuple[str, list]] = []
        self.search_calls: list[dict] = []

    def get_collections(self):
        return SimpleNamespace(
            collections=[SimpleNamespace(name=n) for n in self.collections]
        )

    def create_collection(self, collection_name, vectors_config):
        self.collections.append(collection_name)
        self.created.append(
            {"collection_name": collection_name, "vectors_config": vectors_config}
        )

    def upsert(self, collection_name, points):
        self.upserted.append((collection_name, points))

    def search(self, collection_name, query_vector, limit, with_payload):
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "limit": limit,
                "with_payload": with_payload,
            }
        )
        return [
            SimpleNamespace(payload={"content": "重置密码"}, score=0.91),
            SimpleNamespace(payload={"content": "申请退款"}, score=0.42),
        ][:limit]


def test_missing_qdrant_client_raises_clear_error(monkeypatch):
    # 模拟环境未安装 qdrant_client：sys.modules 置 None 会让 import 抛 ImportError
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    with pytest.raises(ImportError, match=r"pip install evoflow\[vec-qdrant\]"):
        QdrantStore("memories")


def test_add_creates_collection_and_upserts():
    fake = FakeQdrantClient()
    store = QdrantStore("memories", embed_fn=lambda t: [1.0, 0.0, 0.0], _client=fake)
    store.add("id-1", "重置密码", {"content": "重置密码"})

    assert fake.created[0]["collection_name"] == "memories"
    cfg = fake.created[0]["vectors_config"]
    size = cfg.size if hasattr(cfg, "size") else cfg["size"]
    assert size == 3  # 来自 embed_fn 输出维度

    coll, points = fake.upserted[0]
    assert coll == "memories"
    assert len(points) == 1
    point = points[0]
    payload = point.payload if hasattr(point, "payload") else point["payload"]
    assert payload == {"content": "重置密码"}


def test_add_is_idempotent_per_item_id():
    fake = FakeQdrantClient()
    store = QdrantStore("memories", _client=fake)
    store.add("id-1", "文本", {"v": 1})
    store.add("id-1", "文本变了", {"v": 2})
    ids = []
    for _coll, points in fake.upserted:
        p = points[0]
        ids.append(p.id if hasattr(p, "id") else p["id"])
    assert ids[0] == ids[1]  # 同 item_id → 同确定性 point id → 服务端覆盖


def test_add_skips_create_when_collection_exists():
    fake = FakeQdrantClient()
    fake.collections.append("memories")  # 已存在
    store = QdrantStore("memories", _client=fake)
    store.add("id-1", "文本", {})
    assert fake.created == []  # 没有重复建 collection
    assert len(fake.upserted) == 1


def test_search_returns_payload_score_tuples():
    fake = FakeQdrantClient()
    store = QdrantStore("memories", _client=fake)
    hits = store.search("密码", k=2)
    assert hits == [({"content": "重置密码"}, 0.91), ({"content": "申请退款"}, 0.42)]
    call = fake.search_calls[0]
    assert call["collection_name"] == "memories"
    assert call["limit"] == 2
    assert call["with_payload"] is True


def test_search_prefers_query_points_api():
    class NewApiFake(FakeQdrantClient):
        def query_points(self, collection_name, query, limit, with_payload):
            self.search_calls.append(
                {"collection_name": collection_name, "limit": limit, "api": "query_points"}
            )
            return SimpleNamespace(
                points=[SimpleNamespace(payload={"x": 1}, score=0.5)]
            )

    fake = NewApiFake()
    store = QdrantStore("memories", _client=fake)
    hits = store.search("q", k=1)
    assert fake.search_calls[0]["api"] == "query_points"
    assert hits == [({"x": 1}, 0.5)]


def test_save_load_are_noop(tmp_path):
    fake = FakeQdrantClient()
    store = QdrantStore("memories", _client=fake)
    path = tmp_path / "should_not_exist.jsonl"
    assert store.save(path) is None
    assert store.load(path) is None
    assert not path.exists()  # 服务端持久化，本地不落盘
