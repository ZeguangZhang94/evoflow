"""L1 用户画像记忆测试：增删查、渲染、语义检索、持久化。"""

from evoflow.l1_memory.profile import ProfileMemory
from evoflow.l1_memory.store import JsonlVectorStore


def test_update_get_render():
    mem = ProfileMemory()
    mem.update("u1", "饮食偏好", "不吃辣")
    mem.update("u1", "交易权限", "仅主板")
    profile = mem.get("u1")
    assert profile.facts == {"饮食偏好": "不吃辣", "交易权限": "仅主板"}
    assert profile.updated_at > 0

    text = mem.render("u1")
    assert text.startswith("[用户画像 u1]")
    assert "- 饮食偏好: 不吃辣" in text
    assert "- 交易权限: 仅主板" in text


def test_update_overwrites_same_key():
    mem = ProfileMemory()
    mem.update("u1", "城市", "北京")
    mem.update("u1", "城市", "上海")
    assert mem.get("u1").facts["城市"] == "上海"


def test_get_unknown_user_returns_empty_profile():
    mem = ProfileMemory()
    profile = mem.get("ghost")
    assert profile.user_id == "ghost"
    assert profile.facts == {}
    assert "暂无记录" in mem.render("ghost")


def test_render_max_items_truncates():
    mem = ProfileMemory()
    for i in range(5):
        mem.update("u1", f"k{i}", f"v{i}")
    text = mem.render("u1", max_items=2)
    assert "- k0: v0" in text
    assert "- k1: v1" in text
    assert "k2" not in text
    assert "其余 3 条" in text


def test_forget_removes_fact_and_store_record():
    store = JsonlVectorStore()
    mem = ProfileMemory(store=store)
    mem.update("u1", "饮食偏好", "不吃辣")
    mem.update("u1", "交易权限", "仅主板")
    assert len(store) == 2

    assert mem.forget("u1", "饮食偏好") is True
    assert "饮食偏好" not in mem.get("u1").facts
    assert len(store) == 1  # 语义索引同步删除（合规可删除）
    assert "饮食偏好" not in mem.render("u1")
    # 再删一次返回 False
    assert mem.forget("u1", "饮食偏好") is False


def test_recall_semantic_search_per_user():
    mem = ProfileMemory(store=JsonlVectorStore())
    mem.update("u1", "饮食偏好", "不吃辣，清淡为主")
    mem.update("u1", "交易权限", "只能买主板股票")
    mem.update("u2", "饮食偏好", "无辣不欢")

    hits = mem.recall("u1", "能吃辣吗")
    assert hits[0] == ("饮食偏好", "不吃辣，清淡为主")  # 最相关的事实排第一
    # 不会串用户
    assert all(v != "无辣不欢" for _k, v in hits)
    assert all(v != "无辣不欢" for _k, v in mem.recall("u1", "吃辣", k=5))


def test_recall_without_store_returns_empty():
    mem = ProfileMemory()
    mem.update("u1", "城市", "北京")
    assert mem.recall("u1", "住哪") == []


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "profiles.jsonl"
    mem = ProfileMemory()
    mem.update("u1", "城市", "北京")
    mem.update("u2", "语言", "简体中文")
    mem.save(path)

    mem2 = ProfileMemory()
    mem2.load(path)
    assert len(mem2) == 2
    assert mem2.get("u1").facts["城市"] == "北京"
    assert mem2.get("u2").facts["语言"] == "简体中文"


def test_load_missing_file_is_noop(tmp_path):
    mem = ProfileMemory()
    mem.load(tmp_path / "nope.jsonl")  # 不抛异常
    assert len(mem) == 0
